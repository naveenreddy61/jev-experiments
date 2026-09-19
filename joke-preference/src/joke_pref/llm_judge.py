"""An LLM as the rubric reader, for the cost comparison with Jev.

The judge gets the same instructions, the same three level texts and one
joke. Two answer modes:

- `answer="level"`: one word, bad / good / great. The result is a one-hot
  probability vector. Jokes with the same word tie, and a tie counts one
  half in the concordance, as for Jev.
- `answer="score"`: an integer 0 to 100, where 0 is the centre of the bad
  level, 50 the centre of good, 100 the centre of great. The expected level
  is `value / 50`, and the probabilities are the linear split between the
  two nearest levels. This is the graded answer that matches Jev's expected
  value and removes most ties.

DeepSeek only for now. `thinking=False` sends `{"thinking": {"type":
"disabled"}}` so the model answers without a reasoning trace; that is the
fast, cheap judge. `thinking=True` lets it reason first.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

from joke_pref.criteria import Criteria, Level
from joke_pref.data import LABELS
from joke_pref.reflection import DEEPSEEK_URL, DEFAULT_MODEL, KEY_ENV, _post_json
from joke_pref.scorer import ScoreCache, ScoreResult, cache_key, invalid

_ANSWER_RE = re.compile(r"\b(bad|good|great)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
ANSWER_MODES = ("level", "score")


def level_text(level: Level) -> str:
    if isinstance(level, str):
        return level.strip()
    return json.dumps(level, ensure_ascii=False, indent=2)


def build_prompt(criteria: Criteria, joke: str, answer: str = "level") -> str:
    levels = "\n\n".join(f"[{name}]\n{level_text(lv)}" for name, lv in zip(LABELS, criteria.levels))
    if answer == "level":
        task = "Rate the joke below against this rubric. Pick the one level whose description fits the joke best."
        ask = f"Answer with exactly one word: {', '.join(LABELS)}."
    elif answer == "score":
        task = (
            "Rate the joke below against this rubric on a scale from 0 to 100, where 0 is a clear case of the "
            f"[{LABELS[0]}] level, 50 a clear case of [{LABELS[1]}], and 100 a clear case of [{LABELS[2]}]. "
            "Use values between the centres when the joke sits between two levels."
        )
        ask = "Answer with one integer from 0 to 100 and nothing else."
    else:
        raise ValueError(f"unknown answer mode {answer!r}")
    return (
        f"{criteria.instructions.strip()}\n\n"
        f"{task}\n\n"
        f"{levels}\n\n"
        f"Joke:\n\"\"\"\n{joke.strip()}\n\"\"\"\n\n"
        f"{ask}"
    )


def parse_answer(text: str) -> int | None:
    m = _ANSWER_RE.search(text or "")
    if not m:
        return None
    return LABELS.index(m.group(1).lower())


def parse_score(text: str) -> float | None:
    """0..100 -> expected level 0..2, clipped."""
    m = _NUMBER_RE.search(text or "")
    if not m:
        return None
    value = min(100.0, max(0.0, float(m.group(0))))
    return value / 50.0


def soft_probabilities(expected: float) -> tuple[float, ...]:
    """Linear split of the mass between the two nearest levels."""
    lo = min(int(expected), len(LABELS) - 2)
    frac = expected - lo
    probs = [0.0] * len(LABELS)
    probs[lo] = 1.0 - frac
    probs[lo + 1] = frac
    return tuple(probs)


class LLMJudge:
    """Duck-type compatible with `JevScorer` for `score_many`, `calls`, `model`."""

    def __init__(
        self,
        *,
        model: str | None = None,
        thinking: bool = False,
        answer: str = "level",
        timeout: float = 120.0,
        max_workers: int = 8,
        cache: ScoreCache | None = None,
        api_key: str | None = None,
        post: Callable[..., dict] | None = None,
    ):
        self.model = model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_MODEL
        self.thinking = thinking
        if answer not in ANSWER_MODES:
            raise ValueError(f"answer must be one of {ANSWER_MODES}")
        self.answer = answer
        self.base = os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_URL)
        self.timeout = timeout
        self.max_workers = max_workers
        self.cache = cache
        self._post = post or _post_json
        self.key = api_key or os.environ.get(KEY_ENV) or ("test" if post else None)
        if not self.key:
            raise RuntimeError(f"No API key: set {KEY_ENV}")
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self._lock = threading.Lock()

    @property
    def cache_model(self) -> str:
        return f"deepseek-judge:{self.model}:{'think' if self.thinking else 'nothink'}:{self.answer}"

    def score_one(self, joke: str, criteria: Criteria) -> ScoreResult:
        key = cache_key(self.cache_model, criteria, joke, 0)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": build_prompt(criteria, joke, self.answer)}],
            "temperature": 0,
        }
        if not self.thinking:
            payload["thinking"] = {"type": "disabled"}
            payload["max_tokens"] = 16
        t0 = time.perf_counter()
        try:
            body = self._post(
                self.base,
                payload,
                headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                timeout=self.timeout,
            )
        except Exception as exc:  # network or HTTP error: one invalid result, not a crash
            return invalid(f"{type(exc).__name__}: {exc}"[:300], (time.perf_counter() - t0) * 1000.0)
        latency = (time.perf_counter() - t0) * 1000.0
        choices = body.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        content = message.get("content")
        text = "".join(p.get("text", "") for p in content if isinstance(p, dict)) if isinstance(content, list) else str(content or "")
        usage = body.get("usage") or {}
        with self._lock:
            self.calls += 1
            self.input_tokens += int(usage.get("prompt_tokens") or 0)
            self.output_tokens += int(usage.get("completion_tokens") or 0)
            self.reasoning_tokens += int(((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
        if self.answer == "level":
            idx = parse_answer(text)
            expected = None if idx is None else float(idx)
            probs = None if idx is None else tuple(1.0 if i == idx else 0.0 for i in range(len(LABELS)))
        else:
            expected = parse_score(text)
            probs = None if expected is None else soft_probabilities(expected)
        if expected is None or probs is None:
            result = invalid(f"no {self.answer} in answer: {text[:80]!r}", latency)
        else:
            result = ScoreResult(
                probabilities=probs,
                score=expected,
                confidence=max(probs),
                latency_ms=latency,
                input_tokens=int(usage.get("prompt_tokens") or 0) or None,
            )
        if self.cache is not None and result.valid:
            self.cache.put(key, result)
        return result

    def score_many(self, jokes: Iterable[str], criteria: Criteria) -> list[ScoreResult]:
        jokes = list(jokes)
        unique = list(dict.fromkeys(jokes))
        if self.max_workers <= 1 or len(unique) == 1:
            results = [self.score_one(j, criteria) for j in unique]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                results = list(pool.map(lambda j: self.score_one(j, criteria), unique))
        by_joke = dict(zip(unique, results))
        return [by_joke[j] for j in jokes]

    def usage(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "thinking": self.thinking,
            "answer": self.answer,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }
