"""The reflection language model for GEPA: DeepSeek through its native API.

GEPA calls this a few dozen times per run. It reads the current level
description plus examples with feedback, and writes a new description. It
is the only LLM in the project, and it never runs at inference time.

Same endpoint and key names as the parsing experiment: `DEEPSEEK_API_KEY`,
`DEEPSEEK_MODEL` (default `deepseek-flash`), `DEEPSEEK_BASE_URL`.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-flash"
KEY_ENV = "DEEPSEEK_API_KEY"


class DeepSeekReflectionLM:
    """Callable `str | list[messages] -> str`, the protocol gepa expects."""

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout: float = 180.0,
        log_path: str | Path | None = None,
        api_key: str | None = None,
    ):
        self.model = model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_MODEL
        self.base = os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_URL)
        self.timeout = timeout
        self.log_path = Path(log_path) if log_path else None
        self.key = api_key or os.environ.get(KEY_ENV)
        if not self.key:
            raise RuntimeError(f"No API key: set {KEY_ENV}")
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        payload = {"model": self.model, "messages": messages}
        t0 = time.perf_counter()
        body = _post_json(
            self.base,
            payload,
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            timeout=self.timeout,
        )
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"deepseek returned no choices: {str(body)[:400]}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        text = "".join(p.get("text", "") for p in content if isinstance(p, dict)) if isinstance(content, list) else str(content or "")
        usage = body.get("usage") or {}
        with self._lock:
            self.calls += 1
            self.input_tokens += int(usage.get("prompt_tokens") or 0)
            self.output_tokens += int(usage.get("completion_tokens") or 0)
        if self.log_path:
            self._log(messages, text, message.get("reasoning_content"), usage, (time.perf_counter() - t0) * 1000.0)
        return text

    def _log(self, messages: list, text: str, reasoning: Any, usage: dict, latency_ms: float) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
        row = {
            "model": self.model,
            "messages": messages,
            "content": text,
            "reasoning": reasoning if isinstance(reasoning, str) else None,
            "usage": usage,
            "latency_ms": latency_ms,
        }
        with self._lock, open(self.log_path, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _post_json(url: str, payload: dict, *, headers: dict[str, str], timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail[:500]}") from exc
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise RuntimeError("provider JSON root must be an object")
    return obj
