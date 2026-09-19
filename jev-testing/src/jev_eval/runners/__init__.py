"""Model runners. Every provider must return {answer, p_yes} or be marked invalid."""

from __future__ import annotations

from jev_eval.runners.jev import JevRunner
from jev_eval.runners.llm import LlmRunner, parse_provider

PROVIDERS = ("gemini-flash-lite", "deepseek-flash", "jev")


def make_runner(provider: str, *, mock: bool = False, timeout: float = 60.0):
    name = parse_provider(provider)
    if name == "jev":
        return JevRunner(mock=mock, timeout=timeout)
    return LlmRunner(provider=name, mock=mock, timeout=timeout)
