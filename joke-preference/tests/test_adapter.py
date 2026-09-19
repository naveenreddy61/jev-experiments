from __future__ import annotations

import unittest

from joke_pref.adapter import JokeCriteriaAdapter, reflection_prompt_templates
from joke_pref.criteria import COMPONENTS, Criteria
from joke_pref.data import LABELS, joke_text, labeled_premises
from joke_pref.runs import evaluate
from joke_pref.scorer import JevScorer
from tests.fakes import FAKE_SDK, FakeClient, label_for, oracle_decide, sample_premises


def _items():
    premises = sample_premises()
    labels = {
        pl.id: LABELS[label_for(joke_text(p, pl))] for p in premises for pl in p.punchlines
    }
    return labeled_premises(premises, labels)


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = JevScorer(client=FakeClient(decide=oracle_decide), sdk=FAKE_SDK, max_workers=1)
        self.adapter = JokeCriteriaAdapter(self.scorer, "How much will this reader enjoy this joke?")
        self.items = _items()

    def test_evaluate_scores_per_premise(self) -> None:
        good = Criteria(self.adapter.instructions, ("safe", "ok", "dark humor")).to_candidate()
        flat = Criteria(self.adapter.instructions, ("a", "b", "c")).to_candidate()
        eb_good = self.adapter.evaluate(self.items, good, capture_traces=True)
        eb_flat = self.adapter.evaluate(self.items, flat)
        self.assertEqual(len(eb_good.scores), 3)
        self.assertGreater(sum(eb_good.scores), sum(eb_flat.scores))
        self.assertIsNone(eb_flat.trajectories)
        self.assertEqual(len(eb_good.trajectories), 3)
        self.assertEqual(set(eb_good.outputs[0]), {"t-001.0", "t-001.1", "t-001.2"})

    def test_reflective_dataset_shape(self) -> None:
        cand = Criteria(self.adapter.instructions, ("a", "b", "c")).to_candidate()
        eb = self.adapter.evaluate(self.items, cand, capture_traces=True)
        ds = self.adapter.make_reflective_dataset(cand, eb, ["level_great"])
        self.assertEqual(list(ds), ["level_great"])
        rec = ds["level_great"][0]
        self.assertEqual(set(rec), {"Inputs", "Generated Outputs", "Feedback"})
        self.assertEqual(rec["Inputs"]["setup"], "Why did the chicken cross the road?")
        self.assertIn("reader_label", rec["Inputs"]["punchlines"][0])
        self.assertIn("p_great", next(iter(rec["Generated Outputs"].values())))
        self.assertIn("great", rec["Feedback"])
        self.assertIn("Premise score", rec["Feedback"])

    def test_templates_have_placeholders(self) -> None:
        t = reflection_prompt_templates()
        self.assertEqual(set(t), set(COMPONENTS))
        for text in t.values():
            self.assertIn("<curr_param>", text)
            self.assertIn("<side_info>", text)

    def test_evaluate_run_summary(self) -> None:
        crit = Criteria(self.adapter.instructions, ("safe", "ok", "dark humor"))
        rows, evals, summary = evaluate(self.scorer, self.items, crit)
        self.assertEqual(summary["n_punchlines"], 9)
        self.assertEqual(summary["exact_accuracy"], 1.0)
        self.assertEqual(summary["concordance"], 1.0)
        self.assertEqual(summary["top1_hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
