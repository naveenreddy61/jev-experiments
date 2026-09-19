from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from jev_eval.runners import make_runner
from jev_eval.runners.jev import QUESTION_KEY, YES_THRESHOLD, FatalRunnerError, JevRunner
from jev_eval.schema import InvalidOutput, Item, ModelOutput
from jev_eval.tasks import get_task


def _item() -> Item:
    return Item(
        id="python.parse.00001",
        task="python.parse",
        difficulty="easy",
        artifact="x = 1\n",
        question="Does this parse as Python 3.12?",
        answer=True,
        fault_tags=(),
        oracle_meta={"tool": "ast.parse", "version_note": "test", "exit_code": 0},
    )


class RunnerMockTests(unittest.TestCase):
    def test_llm_mock_parses_contract(self) -> None:
        runner = make_runner("gemini-flash-lite", mock=True)
        result = runner.complete(_item())
        self.assertIsInstance(result.parsed, ModelOutput)
        assert isinstance(result.parsed, ModelOutput)
        self.assertIsInstance(result.parsed.answer, bool)
        self.assertGreaterEqual(result.parsed.p_yes, 0.0)
        self.assertLessEqual(result.parsed.p_yes, 1.0)

    def test_deepseek_mock(self) -> None:
        runner = make_runner("deepseek-flash", mock=True)
        result = runner.complete(_item())
        self.assertIsInstance(result.parsed, ModelOutput)

    def test_jev_mock(self) -> None:
        runner = make_runner("jev", mock=True)
        result = runner.complete(_item())
        self.assertIsInstance(result.parsed, ModelOutput)




if __name__ == "__main__":
    unittest.main()


# --- Jev adapter -----------------------------------------------------------
#
# These tests never touch the network and never need typesafe_sdk installed.
# A fake SDK namespace supplies the question type, the answer shape and the
# exception hierarchy that JevRunner refers to.


class _FakeNoul:
    def __init__(self, instructions=None, criteria=None):
        self.instructions = instructions
        self.criteria = criteria


class _FakeNoulAnswer:
    type = "noul"

    def __init__(self, noul: float):
        self.noul = noul

    def model_dump(self) -> dict:
        return {"type": "noul", "noul": self.noul}


class _FakeUsage:
    def model_dump(self) -> dict:
        return {"input_tokens": 10, "output_tokens": 2}


class _FakeResponse:
    def __init__(self, answers: dict):
        self.model = "jev-1.13.0"
        self.answers = answers
        self.usage = _FakeUsage()


class _FakeTypeSafeError(Exception):
    pass


class _FakeAuthError(_FakeTypeSafeError):
    pass


class _FakePermissionError(_FakeTypeSafeError):
    pass


class _FakeRateLimitError(_FakeTypeSafeError):
    pass


def _fake_sdk() -> SimpleNamespace:
    return SimpleNamespace(
        Noul=_FakeNoul,
        TypeSafeError=_FakeTypeSafeError,
        TypeSafeAuthenticationError=_FakeAuthError,
        TypeSafePermissionDeniedError=_FakePermissionError,
        TypeSafeRateLimitError=_FakeRateLimitError,
    )


class _FakeClient:
    """Returns a canned answer, or raises a canned exception."""

    def __init__(self, noul=None, raises=None):
        self.noul = noul
        self.raises = raises
        self.calls: list[dict] = []

    def system_one(self, state, questions, *, model=None, timeout=None):
        self.calls.append(
            {"state": state, "questions": questions, "model": model, "timeout": timeout}
        )
        if self.raises is not None:
            raise self.raises
        return _FakeResponse({QUESTION_KEY: _FakeNoulAnswer(self.noul)})


def _wired_runner(noul=None, raises=None) -> tuple[JevRunner, _FakeClient]:
    """A live-path JevRunner with the SDK and client swapped for fakes."""
    runner = JevRunner(mock=True)  # mock=True skips the real client build
    client = _FakeClient(noul=noul, raises=raises)
    runner.mock = False
    runner._client = client
    runner._sdk = _fake_sdk()
    return runner, client


class JevAdapterTests(unittest.TestCase):
    def test_noul_maps_straight_to_p_yes(self) -> None:
        runner, _ = _wired_runner(noul=0.22)
        parsed = runner.complete(_item()).parsed
        self.assertIsInstance(parsed, ModelOutput)
        assert isinstance(parsed, ModelOutput)
        # p_yes is the noul itself: no inversion, no confidence conversion.
        self.assertAlmostEqual(parsed.p_yes, 0.22)
        self.assertFalse(parsed.answer)

    def test_high_noul_is_yes(self) -> None:
        runner, _ = _wired_runner(noul=0.72)
        parsed = runner.complete(_item()).parsed
        assert isinstance(parsed, ModelOutput)
        self.assertAlmostEqual(parsed.p_yes, 0.72)
        self.assertTrue(parsed.answer)

    def test_threshold_boundary_is_inclusive(self) -> None:
        runner, _ = _wired_runner(noul=YES_THRESHOLD)
        parsed = runner.complete(_item()).parsed
        assert isinstance(parsed, ModelOutput)
        self.assertTrue(parsed.answer, "noul exactly at the threshold counts as yes")

    def test_sends_locked_question_and_raw_artifact(self) -> None:
        item = _item()
        runner, client = _wired_runner(noul=0.9)
        runner.complete(item)
        sent = client.calls[0]
        # State is the artifact itself; no JSON rubric wrapper.
        self.assertEqual(sent["state"], item.artifact)
        question = sent["questions"][QUESTION_KEY]
        self.assertEqual(question.instructions, get_task(item.task).question)
        # v1 keeps the locked question unmodified: no criteria.
        self.assertIsNone(question.criteria)

    def test_cli_timeout_reaches_the_call(self) -> None:
        runner, client = _wired_runner(noul=0.9)
        runner.timeout = 42.0
        runner.complete(_item())
        self.assertEqual(client.calls[0]["timeout"], 42.0)

    def test_rate_limit_is_invalid_not_fatal(self) -> None:
        runner, _ = _wired_runner(raises=_FakeRateLimitError("slow down"))
        result = runner.complete(_item())
        self.assertIsInstance(result.parsed, InvalidOutput)
        self.assertEqual(result.error, "_FakeRateLimitError")

    def test_auth_error_aborts_the_run(self) -> None:
        runner, _ = _wired_runner(raises=_FakeAuthError("bad key"))
        with self.assertRaises(FatalRunnerError):
            runner.complete(_item())

    def test_permission_error_aborts_the_run(self) -> None:
        runner, _ = _wired_runner(raises=_FakePermissionError("no access"))
        with self.assertRaises(FatalRunnerError):
            runner.complete(_item())

    def test_missing_answer_is_invalid(self) -> None:
        runner, client = _wired_runner(noul=0.5)
        client.system_one = lambda *a, **k: _FakeResponse({})
        result = runner.complete(_item())
        self.assertIsInstance(result.parsed, InvalidOutput)
        self.assertEqual(result.error, "missing_noul")

    def test_out_of_range_noul_is_invalid(self) -> None:
        runner, _ = _wired_runner(noul=1.5)
        result = runner.complete(_item())
        self.assertIsInstance(result.parsed, InvalidOutput)
        self.assertEqual(result.error, "noul_out_of_range")

    def test_missing_key_is_fatal(self) -> None:
        # Pin the cause: _build_client imports the SDK before it checks the
        # key, so a bare assertRaises would also pass on a machine without
        # typesafe_sdk installed, for the wrong reason.
        with mock.patch.dict(os.environ, {"JEV_API_KEY": ""}, clear=False):
            with self.assertRaises(FatalRunnerError) as cm:
                JevRunner(mock=False)
        self.assertIn("No API key", str(cm.exception))
