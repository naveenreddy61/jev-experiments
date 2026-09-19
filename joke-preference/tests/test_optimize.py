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
    """Always proposes the text the oracle scorer rewards, in both the
    per-level format and the joint JSON format."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, prompt) -> str:
        self.calls += 1
        self.prompts.append(prompt if isinstance(prompt, str) else str(prompt))
        if "Answer with one JSON object" in str(prompt):
            return (
                "```json\n"
                '{"level_bad": "Light and safe.", "level_good": {"what": "mild dark humor", "examples": ["x"]}, '
                '"level_great": "The joke is dark humor and the reader likes it."}\n```'
            )
        return "```\nThe joke is dark humor and the reader likes it.\n```"


class OptimizeSmokeTest(unittest.TestCase):
    def _run(self, joint: bool):
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
                joint=joint,
                log=lambda _msg: None,
            )
            self.assertTrue((Path(d) / "run" / "best_criteria.json").exists())
        self.assertGreater(lm.calls, 0, "GEPA never asked the reflection model for a proposal")
        self.assertGreater(result.num_candidates, 1)
        self.assertEqual(best.instructions, seed.instructions)
        return best, lm

    def test_per_level_proposer(self) -> None:
        best, lm = self._run(joint=False)
        self.assertNotIn("Answer with one JSON object", lm.prompts[0])

    def test_joint_proposer(self) -> None:
        best, lm = self._run(joint=True)
        self.assertIn("Answer with one JSON object", lm.prompts[0])
        self.assertIn("[bad]", lm.prompts[0])
        self.assertEqual(best.levels[1], {"what": "mild dark humor", "examples": ["x"]})
        self.assertIn("dark", best.levels[2])


class JointParseTests(unittest.TestCase):
    def test_parse_and_shuffle(self) -> None:
        from joke_pref.adapter import parse_joint_proposal
        from joke_pref.data import shuffle_labels

        self.assertIsNone(parse_joint_proposal("no json here"))
        got = parse_joint_proposal('text {"level_bad": "a", "level_great": {"what": "w"}} tail')
        self.assertEqual(got["level_bad"], "a")
        self.assertIn('"what": "w"', got["level_great"])
        self.assertNotIn("level_good", got)

        items = _items()
        shuffled = shuffle_labels(items, seed=1)
        for a, b in zip(items, shuffled):
            self.assertEqual(sorted(a.labels.values()), sorted(b.labels.values()))
            self.assertEqual(set(a.labels), set(b.labels))
        self.assertNotEqual([i.labels for i in items], [i.labels for i in shuffled])


if __name__ == "__main__":
    unittest.main()
