from __future__ import annotations

import unittest

from joke_pref.metric import (
    Row,
    concordance,
    pairs,
    premise_score,
    spearman,
    summarize,
    top1_hit_rate,
)
from joke_pref.scorer import ScoreResult, invalid


def res(p0: float, p1: float, p2: float) -> ScoreResult:
    return ScoreResult((p0, p1, p2), p1 + 2 * p2, max(p0, p1, p2), 10.0)


class PairTests(unittest.TestCase):
    def test_pairs_skip_equal_labels(self) -> None:
        ps = pairs([0.2, 1.5, 1.0], [0, 2, 2], ["a", "b", "c"])
        self.assertEqual(len(ps), 2)
        self.assertEqual({(p.hi_id, p.lo_id) for p in ps}, {("b", "a"), ("c", "a")})
        self.assertEqual(concordance(ps), 1.0)

    def test_ties_half_credit(self) -> None:
        ps = pairs([1.0, 1.0], [0, 2], ["a", "b"])
        self.assertEqual(concordance(ps), 0.5)
        self.assertIsNone(concordance([]))


class PremiseScoreTests(unittest.TestCase):
    def test_perfect(self) -> None:
        results = [res(0.9, 0.05, 0.05), res(0.05, 0.05, 0.9)]
        ev = premise_score(results, [0, 2], ["a", "b"])
        self.assertEqual(ev.concordance, 1.0)
        self.assertAlmostEqual(ev.agreement, 0.9)
        self.assertAlmostEqual(ev.score, 0.6 * 1.0 + 0.4 * 0.9)
        self.assertEqual(ev.inverted, ())

    def test_inverted(self) -> None:
        results = [res(0.05, 0.05, 0.9), res(0.9, 0.05, 0.05)]
        ev = premise_score(results, [0, 2], ["a", "b"])
        self.assertEqual(ev.concordance, 0.0)
        self.assertEqual(len(ev.inverted), 1)
        self.assertEqual(ev.inverted[0].hi_id, "b")

    def test_same_labels_use_agreement_only(self) -> None:
        results = [res(0.1, 0.8, 0.1), res(0.1, 0.6, 0.3)]
        ev = premise_score(results, [1, 1], ["a", "b"])
        self.assertIsNone(ev.concordance)
        self.assertAlmostEqual(ev.score, 0.7)

    def test_invalid_counts_zero(self) -> None:
        ev = premise_score([invalid("boom"), res(0.1, 0.1, 0.8)], [0, 2], ["a", "b"])
        self.assertAlmostEqual(ev.agreement, 0.4)
        self.assertEqual(ev.concordance, 1.0)  # invalid predicted as -1 sorts lowest


class SummaryTests(unittest.TestCase):
    def test_spearman(self) -> None:
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(spearman([1, 2], [1, 2]))

    def test_summarize_and_top1(self) -> None:
        rows = [
            Row("p1", "p1.0", 0, res(0.8, 0.1, 0.1)),
            Row("p1", "p1.1", 2, res(0.1, 0.2, 0.7)),
            Row("p2", "p2.0", 1, res(0.1, 0.7, 0.2)),
            Row("p2", "p2.1", 2, res(0.6, 0.3, 0.1)),
            Row("p3", "p3.0", 1, invalid("x")),
        ]
        evals = [
            premise_score([r.result for r in rows[:2]], [0, 2], ["p1.0", "p1.1"]),
            premise_score([r.result for r in rows[2:4]], [1, 2], ["p2.0", "p2.1"]),
            premise_score([rows[4].result], [1], ["p3.0"]),
        ]
        s = summarize(rows, evals)
        self.assertEqual(s["n_punchlines"], 5)
        self.assertAlmostEqual(s["invalid_rate"], 0.2)
        self.assertAlmostEqual(s["exact_accuracy"], 0.75)
        self.assertEqual(s["label_counts"], {"bad": 1, "good": 1, "great": 2})
        self.assertEqual(s["confusion_rows_true_cols_pred"][2], [1, 0, 1])
        self.assertEqual(top1_hit_rate(rows), 0.5)


if __name__ == "__main__":
    unittest.main()
