"""Jev Score calls for one joke at a time.

One call holds one full joke as `state` and one `Score` question. The
instructions are fixed. The three level descriptions are the criteria under
optimization. See https://docs.typesafe.ai/primitives/score.

The answer is a probability per level. `score` is the probability-weighted
level, from 0 (bad) to 2 (great). Results are cached on disk by the hash of
(model, instructions, levels, joke, repeat), so re-scoring the same candidate
costs nothing and the optimizer stays repeatable.

Error policy follows the parsing experiment: an authentication or permission
error aborts the whole run; a timeout, rate limit, 5xx or validation error
makes one result invalid and never becomes a number.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from joke_pref.criteria import Criteria
from joke_pref.data import LABELS

DEFAULT_MODEL = "jev-latest"
MODEL_ENV = "JEV_MODEL"
KEY_ENV = "JEV_API_KEY"
QUESTION_KEY = "liking"


class FatalScorerError(RuntimeError):
    """An error no per-item retry can fix, such as a bad API key."""


@dataclass(frozen=True)
class ScoreResult:
    probabilities: tuple[float, ...] | None  # one per level, or None when invalid
    score: float | None  # expected level, 0..len(LABELS)-1
    confidence: float | None
    latency_ms: float
    error: str | None = None
    input_tokens: int | None = None

    @property
    def valid(self) -> bool:
        return self.probabilities is not None

    @property
    def argmax(self) -> int | None:
        if self.probabilities is None:
            return None
        return max(range(len(self.probabilities)), key=self.probabilities.__getitem__)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScoreResult":
        probs = d.get("probabilities")
        return cls(
            probabilities=tuple(probs) if probs is not None else None,
            score=d.get("score"),
            confidence=d.get("confidence"),
            latency_ms=float(d.get("latency_ms", 0.0)),
            error=d.get("error"),
            input_tokens=d.get("input_tokens"),
        )


def invalid(reason: str, latency_ms: float = 0.0) -> ScoreResult:
    return ScoreResult(None, None, None, latency_ms, error=reason)


class ScoreCache:
    """A small sqlite key-value store. Safe for use from several threads."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS scores (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self._conn.commit()
        self._lock = threading.Lock()

    def get(self, key: str) -> ScoreResult | None:
        with self._lock:
            row = self._conn.execute("SELECT v FROM scores WHERE k = ?", (key,)).fetchone()
        return ScoreResult.from_dict(json.loads(row[0])) if row else None

    def put(self, key: str, result: ScoreResult) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO scores (k, v) VALUES (?, ?)",
                (key, json.dumps(result.as_dict())),
            )
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0])


def cache_key(model: str, criteria: Criteria, joke: str, repeat: int) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([model, criteria.key(), joke, repeat]).encode("utf-8"))
    return h.hexdigest()


class JevScorer:
    """Scores jokes against a `Criteria` with Jev's `Score` primitive.

    Pass `client` and `sdk` to inject fakes in tests. Otherwise the client is
    built from `JEV_API_KEY` and `JEV_MODEL`.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout: float = 30.0,
        max_workers: int = 8,
        repeats: int = 1,
        cache: ScoreCache | None = None,
        client: Any = None,
        sdk: Any = None,
    ):
        self.model = model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL
        self.timeout = timeout
        self.max_workers = max_workers
        self.repeats = max(1, repeats)
        self.cache = cache
        self.calls = 0
        self._lock = threading.Lock()
        if client is None or sdk is None:
            client, sdk = self._build_client()
        self._client = client
        self._sdk = sdk

    def _build_client(self) -> tuple[Any, Any]:
        try:
            import typesafe_sdk as sdk
        except ImportError as exc:  # pragma: no cover
            raise FatalScorerError("typesafe_sdk is not installed; run `uv sync`") from exc
        key = os.environ.get(KEY_ENV)
        if not key:
            raise FatalScorerError(
                f"No API key: set {KEY_ENV}. Refusing to call the network."
            )
        retry = sdk.RetryPolicy(max_retries=3, timeout=self.timeout)
        client = sdk.TypeSafeClient(api_key=key, timeout=self.timeout, retry=retry)
        return client, sdk

    # --- single call -------------------------------------------------------

    def _call(self, joke: str, criteria: Criteria) -> ScoreResult:
        sdk = self._sdk
        question = sdk.Score(instructions=criteria.instructions, criteria=list(criteria.levels))
        t0 = time.perf_counter()
        try:
            response = self._client.system_one(
                joke, {QUESTION_KEY: question}, model=self.model, timeout=self.timeout
            )
        except (sdk.TypeSafeAuthenticationError, sdk.TypeSafePermissionDeniedError) as exc:
            raise FatalScorerError(f"{type(exc).__name__}: {exc}") from exc
        except sdk.TypeSafeError as exc:
            return invalid(f"{type(exc).__name__}: {exc}", (time.perf_counter() - t0) * 1000.0)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self.calls += 1
        return parse_answer(response, latency_ms)

    def score_one(self, joke: str, criteria: Criteria) -> ScoreResult:
        """Score one joke, averaged over `repeats` calls when repeats > 1."""
        results = [self._cached(joke, criteria, r) for r in range(self.repeats)]
        return average(results)

    def _cached(self, joke: str, criteria: Criteria, repeat: int) -> ScoreResult:
        key = cache_key(self.model, criteria, joke, repeat) if self.cache is not None else None
        if key is not None:
            hit = self.cache.get(key)  # type: ignore[union-attr]
            if hit is not None and hit.valid:
                return hit
        result = self._call(joke, criteria)
        if key is not None and result.valid:
            self.cache.put(key, result)  # type: ignore[union-attr]
        return result

    # --- batch -------------------------------------------------------------

    def score_many(self, jokes: Iterable[str], criteria: Criteria) -> list[ScoreResult]:
        jokes = list(jokes)
        if not jokes:
            return []
        # Score each distinct joke once, so duplicates in one batch never
        # race past the cache.
        unique = list(dict.fromkeys(jokes))
        if self.max_workers <= 1 or len(unique) == 1:
            results = [self.score_one(j, criteria) for j in unique]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                results = list(pool.map(lambda j: self.score_one(j, criteria), unique))
        by_joke = dict(zip(unique, results))
        return [by_joke[j] for j in jokes]


def parse_answer(response: Any, latency_ms: float) -> ScoreResult:
    """Map an SDK `SystemOneResponse` onto a `ScoreResult`."""
    answers = getattr(response, "answers", None) or {}
    answer = answers.get(QUESTION_KEY)
    if answer is None or getattr(answer, "type", None) != "score":
        return invalid(f"response has no score answer under {QUESTION_KEY!r}", latency_ms)
    raw_probs = getattr(answer, "probabilities", None) or {}
    probs: list[float] = []
    for i in range(len(LABELS)):
        value = raw_probs.get(i, raw_probs.get(str(i)))
        if value is None:
            return invalid(f"probabilities missing level {i}: {raw_probs!r}", latency_ms)
        probs.append(float(value))
    total = sum(probs)
    if not 0.98 <= total <= 1.02 or any(p < 0 for p in probs):
        return invalid(f"probabilities do not sum to 1: {probs!r}", latency_ms)
    usage = getattr(response, "usage", None)
    tokens = getattr(usage, "input_tokens", None)
    return ScoreResult(
        probabilities=tuple(probs),
        score=float(answer.score),
        confidence=float(answer.confidence),
        latency_ms=latency_ms,
        input_tokens=int(tokens) if isinstance(tokens, int) else None,
    )


def average(results: list[ScoreResult]) -> ScoreResult:
    """Mean of the valid results. One result comes back unchanged."""
    valid_results = [r for r in results if r.valid]
    if len(results) == 1:
        return results[0]
    if not valid_results:
        return invalid("; ".join(r.error or "invalid" for r in results))
    n = len(valid_results)
    probs = tuple(
        sum(r.probabilities[i] for r in valid_results) / n  # type: ignore[index]
        for i in range(len(LABELS))
    )
    return ScoreResult(
        probabilities=probs,
        score=sum(r.score for r in valid_results) / n,  # type: ignore[misc]
        confidence=sum(r.confidence for r in valid_results) / n,  # type: ignore[misc]
        latency_ms=sum(r.latency_ms for r in results),
        input_tokens=sum(r.input_tokens or 0 for r in valid_results) or None,
    )
