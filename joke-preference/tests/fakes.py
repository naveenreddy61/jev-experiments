"""A fake `typesafe_sdk` namespace and a fake scorer for offline tests.

Copied in spirit from parsing-experiments/tests/test_runners.py: the tests
never touch the network and never need the real SDK.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from joke_pref.data import LABELS, Premise, parse_premise


class FakeTypeSafeError(Exception):
    pass


class FakeAuthError(FakeTypeSafeError):
    pass


class FakePermissionError(FakeTypeSafeError):
    pass


class FakeRateLimitError(FakeTypeSafeError):
    pass


class FakeScore:
    def __init__(self, *, instructions, criteria):
        self.instructions = instructions
        self.criteria = criteria


class FakeRetryPolicy:
    def __init__(self, **kw):
        self.kw = kw


def fake_response(probs: dict, *, key: str = "liking", input_tokens: int = 50):
    score = sum(int(k) * v for k, v in probs.items())
    answer = SimpleNamespace(
        type="score",
        score=score,
        confidence=max(probs.values()),
        probabilities=dict(probs),
        legend={},
    )
    return SimpleNamespace(
        model="jev-latest",
        answers={key: answer},
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=5),
    )


class FakeClient:
    """Returns a probability distribution chosen by a function of the joke."""

    def __init__(self, decide=None, error: Exception | None = None):
        self.decide = decide or (lambda joke, question: {0: 0.2, 1: 0.5, 2: 0.3})
        self.error = error
        self.calls: list[tuple[str, FakeScore]] = []

    def system_one(self, state, questions, *, model=None, timeout=None):
        question = questions["liking"]
        self.calls.append((state, question))
        if self.error is not None:
            raise self.error
        return fake_response(self.decide(state, question))


FAKE_SDK = SimpleNamespace(
    Score=FakeScore,
    RetryPolicy=FakeRetryPolicy,
    TypeSafeError=FakeTypeSafeError,
    TypeSafeAuthenticationError=FakeAuthError,
    TypeSafePermissionDeniedError=FakePermissionError,
    TypeSafeRateLimitError=FakeRateLimitError,
)


SAMPLE_PREMISES = [
    {
        "id": "t-001",
        "premise": "Why did the chicken cross the road?",
        "format": "question",
        "domain": "animals",
        "punchlines": [
            {"text": "To get to the other side.", "style": "anti-joke"},
            {"text": "It was fleeing the consequences of its actions.", "style": "dark"},
            {"text": "Because the road was there, and the chicken had a plan.", "style": "deadpan"},
        ],
    },
    {
        "id": "t-002",
        "premise": "My therapist says I have a preoccupation with vengeance.",
        "format": "one-liner",
        "domain": "health",
        "punchlines": [
            {"text": "We'll see about that.", "style": "misdirection"},
            {"text": "I told her she'd regret saying that.", "style": "callback"},
        ],
    },
    {
        "id": "t-003",
        "premise": "What's the difference between a cat and a comma?",
        "format": "difference",
        "domain": "language",
        "punchlines": [
            {"text": "One has claws at the end of its paws; the other is a pause at the end of a clause.", "style": "pun"},
            {"text": "You can't ignore a cat.", "style": "observational"},
            {"text": "A comma has never knocked a glass off a table on purpose.", "style": "absurdist"},
            {"text": "Nothing. Both show up uninvited.", "style": "deadpan"},
        ],
    },
]


def write_sample_premises(directory: str | Path | None = None) -> Path:
    directory = Path(directory) if directory else Path(tempfile.mkdtemp())
    path = directory / "premises.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for obj in SAMPLE_PREMISES:
            fh.write(json.dumps(obj) + "\n")
    return path


def sample_premises() -> list[Premise]:
    return [parse_premise(o) for o in SAMPLE_PREMISES]


def label_for(joke: str) -> int:
    """A deterministic "taste": dark and misdirection lines are great,
    anti-jokes are bad, the rest good."""
    lower = joke.lower()
    if "consequences" in lower or "regret" in lower or "uninvited" in lower:
        return 2
    if "other side" in lower or "you can't ignore" in lower:
        return 0
    return 1


def oracle_decide(joke: str, question) -> dict:
    """A fake Jev that agrees with `label_for` when the criteria mention
    'dark', and answers uniformly otherwise."""
    text = json.dumps(question.criteria).lower()
    if "dark" not in text:
        return {0: 0.34, 1: 0.33, 2: 0.33}
    lab = label_for(joke)
    probs = {i: 0.1 for i in range(len(LABELS))}
    probs[lab] = 0.8
    return probs
