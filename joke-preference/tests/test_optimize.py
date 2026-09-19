from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from joke_pref.criteria import Criteria
from joke_pref.runs import optimize
from joke_pref.scorer import JevScorer
from tests.fakes import FAKE_SDK, FakeClient, oracle_decide
from tests.test_adapter import _items


class FakeReflectionLM:
    """Always proposes the text the oracle scorer rewards."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt) -> str:
        self.calls += 1
        return "```\nThe joke is dark humor and the reader likes it.\n```"


class OptimizeSmokeTest(unittest.TestCase):
    def test_gepa_runs_offline_and_calls_reflection(self) -> None:
        scorer = JevScorer(client=FakeClient(decide=oracle_decide), sdk=FAKE_SDK, max_workers=1)
        items = _items()
        lm = FakeReflectionLM()
        seed = Criteria("How much will this reader enjoy this joke?", ("a", "b", "c"))
        with tempfile.TemporaryDirectory() as d:
            best, result = optimize(
                scorer,
                train=items,
                val=items,
                seed_criteria=seed,
                reflection_lm=lm,
                max_metric_calls=30,
                run_dir=Path(d) / "run",
                reflection_minibatch_size=2,
                log=lambda _msg: None,
            )
            self.assertTrue((Path(d) / "run" / "best_criteria.json").exists())
        self.assertGreater(lm.calls, 0, "GEPA never asked the reflection model for a proposal")
        self.assertGreater(result.num_candidates, 1)
        self.assertEqual(best.instructions, seed.instructions)


if __name__ == "__main__":
    unittest.main()
