"""Offline tests for the Jester evaluation path. Synthetic ratings only."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from joke_pref.data import joke_text
from joke_pref.jester import Joke, Ratings, item_split, tercile, tercile_cuts, user_items
from joke_pref.jester_eval import (
    concordance_rows,
    evaluate_rubric,
    loo_item_mean,
    selected_users,
    tercile_labels,
    user_mask,
)
from joke_pref.metric import concordance as ref_concordance
from joke_pref.metric import pairs as ref_pairs
from joke_pref.scorer import ScoreResult

SEED = 20260919


def _ratings(n_users: int = 30) -> Ratings:
    rng = np.random.default_rng(1)
    taste = rng.normal(size=(n_users, 3))
    item = rng.normal(size=(3, 100))
    x = np.clip(taste @ item * 3 + rng.normal(scale=2, size=(n_users, 100)), -10, 10)
    return Ratings(matrix=x, user_ids=np.arange(101, 101 + n_users), joke_ids=np.arange(1, 101))


def _jokes() -> list[Joke]:
    return [Joke(id=i, text=f"Joke number {i}. Why? Because {i}.") for i in range(1, 101)]


def _result(probs: tuple[float, float, float]) -> ScoreResult:
    score = sum(p * i for i, p in enumerate(probs))
    return ScoreResult(probabilities=probs, score=score, confidence=max(probs), latency_ms=1.0)


class MetricTests(unittest.TestCase):
    def test_concordance_matches_reference(self) -> None:
        rng = np.random.default_rng(2)
        preds = rng.normal(size=(6, 12))
        truth = rng.integers(0, 3, size=(6, 12)).astype(float)
        preds[0, :4] = 0.0  # ties
        fast = concordance_rows(preds, truth)
        ids = [str(i) for i in range(12)]
        for row in range(6):
            slow = ref_concordance(ref_pairs(preds[row].tolist(), truth[row].tolist(), ids))
            self.assertAlmostEqual(slow, fast[row])

    def test_tercile_labels_use_train_cuts(self) -> None:
        train = np.array([[0.0, 1, 2, 3, 4, 5, 6, 7, 8]])
        target = np.array([[-5.0, 2.0, 4.0, 6.0, 20.0]])
        labels = tercile_labels(train, target)
        self.assertEqual(labels.tolist(), [[0, 0, 1, 2, 2]])
        cuts = tercile_cuts(train[0])
        self.assertEqual([tercile(v, cuts) for v in target[0]], [0, 0, 1, 2, 2])


class EvaluateRubricTests(unittest.TestCase):
    def test_rubric_equal_to_crowd_gives_crowd_concordance(self) -> None:
        ratings = _ratings()
        jokes = _jokes()
        mean = ratings.matrix.mean(axis=0)
        # a rubric whose expected level follows the item mean exactly
        results = []
        for m in mean:
            p_great = (m + 10) / 20
            results.append(_result((1 - p_great, 0.0, p_great)))
        # the crowd predictor is the same item mean, so the two columns agree
        test_cols = ratings.columns(item_split(SEED)[1])
        crowd = np.tile(mean[test_cols][None, :], (30, 1))
        ev = evaluate_rubric(results, jokes, ratings, seed=SEED, k=5, crowd_pred=crowd)
        self.assertEqual(len(ev.test_ids), 40)
        self.assertEqual(ev.truth.shape, (30, 40))
        self.assertEqual(ev.n_invalid, 0)
        self.assertTrue(np.allclose(ev.rubric_conc, ev.crowd_conc, equal_nan=True))
        self.assertGreater(float(np.nanmedian(ev.knn_conc)), 0.5)
        table = ev.table()
        self.assertEqual(table["n_users"], 30)
        self.assertTrue(0.0 <= table["rubric_agreement_mean"] <= 1.0)

    def test_invalid_results_are_counted(self) -> None:
        ratings = _ratings(10)
        results = [_result((0.2, 0.5, 0.3)) for _ in range(99)]
        results.append(ScoreResult(None, None, None, 0.0, error="boom"))
        ev = evaluate_rubric(results, _jokes(), ratings, seed=SEED, k=3)
        self.assertEqual(ev.n_invalid, 1 if 100 in ev.test_ids else 0)
        # a constant prediction gives 0.5 concordance everywhere
        self.assertTrue(np.allclose(ev.rubric_conc[~np.isnan(ev.rubric_conc)], 0.5))

    def test_loo_item_mean_excludes_self(self) -> None:
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        loo = loo_item_mean(x)
        self.assertAlmostEqual(loo[0, 0], 4.0)
        self.assertAlmostEqual(loo[2, 1], 3.0)


class UserItemsTests(unittest.TestCase):
    def test_groups_cover_split_with_tercile_labels(self) -> None:
        ratings = _ratings(5)
        jokes = _jokes()
        train = user_items(ratings, jokes, 103, "train", seed=SEED)
        test = user_items(ratings, jokes, 103, "test", seed=SEED)
        self.assertEqual(len(train), 12)
        self.assertEqual(len(test), 8)
        self.assertTrue(all(len(it.premise.punchlines) == 5 for it in train + test))
        train_ids, test_ids = item_split(SEED)
        seen = sorted(int(pl.style.split("-")[1]) for it in train for pl in it.premise.punchlines)
        self.assertEqual(seen, sorted(train_ids))
        seen = sorted(int(pl.style.split("-")[1]) for it in test for pl in it.premise.punchlines)
        self.assertEqual(seen, sorted(test_ids))
        # labels come from the user's own train terciles
        row = ratings.matrix[2]
        cuts = tercile_cuts(row[ratings.columns(train_ids)])
        for it in train + test:
            for pl, lab in it.items():
                jid = int(pl.style.split("-")[1])
                self.assertEqual(lab, tercile(float(row[ratings.column(jid)]), cuts))
        # the empty setup does not add a blank line
        pl = train[0].premise.punchlines[0]
        self.assertEqual(joke_text(train[0].premise, pl), pl.text)
        # same seed, same grouping; different user, different grouping
        again = user_items(ratings, jokes, 103, "train", seed=SEED)
        self.assertEqual([it.id for it in again], [it.id for it in train])
        self.assertEqual([pl.text for it in again for pl in it.premise.punchlines], [pl.text for it in train for pl in it.premise.punchlines])
        other = user_items(ratings, jokes, 104, "train", seed=SEED)
        self.assertNotEqual([pl.text for it in other for pl in it.premise.punchlines], [pl.text for it in train for pl in it.premise.punchlines])

    def test_odd_group_size_merges_lone_joke(self) -> None:
        ratings = _ratings(3)
        items = user_items(ratings, _jokes(), 101, "test", seed=SEED, group_size=13)
        self.assertEqual(sum(len(it.premise.punchlines) for it in items), 40)
        self.assertTrue(all(len(it.premise.punchlines) >= 2 for it in items))


class SelectionTests(unittest.TestCase):
    def test_selected_users_top_plus_random(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "user_stats.csv"
            lines = ["user_id,gap_train_only"] + [f"{u},{(u * 37) % 100 / 100:.2f}" for u in range(1, 101)]
            path.write_text("\n".join(lines) + "\n")
            chosen = selected_users(path, n_top=4, n_random=3, seed=1)
            self.assertEqual(len(chosen), 7)
            self.assertEqual(len(set(chosen)), 7)
            top = sorted(range(1, 101), key=lambda u: -((u * 37) % 100))[:4]
            self.assertTrue(set(top) <= set(chosen))
            self.assertEqual(selected_users(path, n_top=4, n_random=3, seed=1), chosen)
        ratings = _ratings(5)
        self.assertEqual(user_mask(ratings, [101, 105]).tolist(), [True, False, False, False, True])


if __name__ == "__main__":
    unittest.main()
