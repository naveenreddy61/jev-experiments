"""LLM baselines: Gemini Flash Lite and DeepSeek Flash.

Both must emit {"answer": true, "p_yes": 0.87}. The harness never imputes
missing fields. API keys are read from the environment only.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from jev_eval.schema import InvalidOutput, Item, ModelOutput, parse_model_output
from jev_eval.tasks import get_task

PROVIDERS = ("gemini-flash-lite", "deepseek-flash", "jev")

GEMINI_DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEEPSEEK_DEFAULT_MODEL = "deepseek-flash"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
GEMINI_URL_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

SYSTEM_RUBRIC = """You are scoring a single atomic yes/no item.
Answer only the exact question below. Do not execute, compile, or interpret beyond the question.

Question: {question}

Return a single JSON object with exactly these fields:
- "answer": boolean — true if the answer is yes, false if no
- "p_yes": number in [0, 1] — your probability that the correct answer is yes

No markdown, no extra keys, no commentary. Example: {{"answer": true, "p_yes": 0.87}}
"""


def parse_provider(name: str) -> str:
    key = name.strip().lower()
    if key not in PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; expected one of {', '.join(PROVIDERS)}")
    return key


def required_api_key(provider: str) -> tuple[str, str | None]:
    """Return (env_var_name, value) for the provider's API key."""
    if provider == "gemini-flash-lite":
        for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            val = os.environ.get(var)
            if val:
                return var, val
        return "GEMINI_API_KEY or GOOGLE_API_KEY", None
    if provider == "deepseek-flash":
        return "DEEPSEEK_API_KEY", os.environ.get("DEEPSEEK_API_KEY")
    return "JEV_API_KEY", os.environ.get("JEV_API_KEY")


def missing_key_message(provider: str) -> str:
    env_name, _ = required_api_key(provider)
    # Jev returns typed answers, so its dry-run exercises the noul mapping,
    # not the JSON text contract the LLM baselines need.
    dry_run = (
        "the noul-to-contract mapping"
        if provider == "jev"
        else "JSON output parsing"
    )
    return (
        f"No API key for {provider} (set {env_name}). "
        "Refusing to call the network. Re-run with --mock for an offline "
        f"harness dry-run that still exercises {dry_run}."
    )


def build_prompt(item: Item) -> tuple[str, str]:
    spec = get_task(item.task)
    system = SYSTEM_RUBRIC.format(question=spec.question)
    user = f"Artifact:\n\n{item.artifact}\n"
    return system, user


@dataclass
class CallResult:
    parsed: ModelOutput | InvalidOutput
    latency_ms: float
    raw: str
    error: str | None = None
    #: Provider token accounting, verbatim, when the API reports it. For a
    #: reasoning model this carries the reasoning-token count, which latency
    #: only approximates.
    usage: dict[str, Any] | None = None
    #: The model's reasoning text, when the API returns it separately from the
    #: answer. Kept so failures can be read, not only counted.
    reasoning: str | None = None


class LlmRunner:
    def __init__(self, provider: str, *, mock: bool = False, timeout: float = 60.0):
        self.provider = parse_provider(provider)
        if self.provider == "jev":
            raise ValueError("use JevRunner for provider 'jev'")
        self.mock = mock
        self.timeout = timeout

    def complete(self, item: Item) -> CallResult:
        if self.mock:
            return self._mock(item)
        t0 = time.perf_counter()
        usage: dict[str, Any] | None = None
        reasoning: str | None = None
        try:
            raw, usage, reasoning = self._call(item)
            error = None
        except Exception as exc:  # noqa: BLE001 — surface transport errors as invalid
            raw = ""
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if error:
            return CallResult(
                parsed=InvalidOutput(reason=error, raw=raw),
                latency_ms=latency_ms,
                raw=raw,
                error=error,
            )
        return CallResult(
            parsed=parse_model_output(raw),
            latency_ms=latency_ms,
            raw=raw,
            usage=usage,
            reasoning=reasoning,
        )

    def _mock(self, item: Item) -> CallResult:
        # Deterministic dummy that still goes through parse_model_output.
        p_yes = 0.61 if hash(item.id) % 2 == 0 else 0.39
        answer = p_yes >= 0.5
        raw = json.dumps({"answer": answer, "p_yes": p_yes})
        t0 = time.perf_counter()
        parsed = parse_model_output(raw)
        return CallResult(
            parsed=parsed,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            raw=raw,
            usage=None,
            reasoning=None,
        )

    def _call(self, item: Item) -> tuple[str, dict[str, Any] | None, str | None]:
        system, user = build_prompt(item)
        if self.provider == "gemini-flash-lite":
            return self._call_gemini(system, user)
        return self._call_deepseek(system, user)

    def _call_gemini(self, system: str, user: str) -> tuple[str, dict[str, Any] | None, str | None]:
        _, key = required_api_key(self.provider)
        if not key:
            raise RuntimeError(missing_key_message(self.provider))
        model = os.environ.get("GEMINI_MODEL", GEMINI_DEFAULT_MODEL)
        url = GEMINI_URL_TMPL.format(model=model)
        payload: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                # No sampling parameters. The harness takes each provider's
                # default so it measures the model as shipped, not a tuned
                # configuration of it.
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "answer": {"type": "BOOLEAN"},
                        "p_yes": {"type": "NUMBER"},
                    },
                    "required": ["answer", "p_yes"],
                },
            },
        }
        body = _http_json(
            url,
            payload,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            timeout=self.timeout,
        )
        candidates = body.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"gemini returned no candidates: {body!r}"[:400])
        parts = candidates[0].get("content", {}).get("parts") or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        usage = body.get("usageMetadata")
        return "".join(texts), usage if isinstance(usage, dict) else None, None

    def _call_deepseek(self, system: str, user: str) -> tuple[str, dict[str, Any] | None, str | None]:
        _, key = required_api_key(self.provider)
        if not key:
            raise RuntimeError(missing_key_message(self.provider))
        model = os.environ.get("DEEPSEEK_MODEL", DEEPSEEK_DEFAULT_MODEL)
        base = os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_URL)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # No sampling parameters; see the note in _call_gemini.
            "response_format": {"type": "json_object"},
        }
        body = _http_json(
            base,
            payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"deepseek returned no choices: {body!r}"[:400])
        message = choices[0].get("message") or {}
        usage = body.get("usage")
        usage = usage if isinstance(usage, dict) else None
        # A reasoning model returns its chain separately from the answer.
        reasoning = message.get("reasoning_content")
        reasoning = reasoning if isinstance(reasoning, str) else None
        content = message.get("content")
        if isinstance(content, list):
            text = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
        else:
            text = "" if content is None else str(content)
        return text, usage, reasoning


def _http_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail[:500]}") from exc
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"provider returned non-JSON body: {raw[:300]}") from exc
    if not isinstance(obj, dict):
        raise RuntimeError("provider JSON root must be an object")
    return obj
