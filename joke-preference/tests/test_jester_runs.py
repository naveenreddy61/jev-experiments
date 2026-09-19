"""Offline smoke test for the per-user Jester optimization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from joke_pref.criteria import Criteria
from joke_pref.jester import Joke, Ratings
from joke_pref.jester_runs import References, append_row, done_users, optimize_user, partner_of, split_groups
from joke_pref.scorer import JevScorer
from tests.fakes import FAKE_SDK, FakeClient, oracle_decide
from tests.test_optimize import FakeReflectionLM

SEED = 20260919


def _ratings(n_users: int = 12) -> Ratings:
    rng = np.random.default_rng(3)
    x = np.clip(rng.normal(scale=4, size=(n_users, 100)), -10, 10)
    return Ratings(matrix=x, user_ids=np.arange(1, n_users + 1), joke_ids=np.arange(1, 101))


class JesterRunsTest(unittest.TestCase):
    def test_optimize_user_writes_row(self) -> None:
        ratings = _ratings()
        jokes = [Joke(id=i, text=f"Joke {i}. The reader likes dark humor here." if i % 2 else f"Joke {i}. Light.") for i in range(1, 101)]
        scorer = JevScorer(client=FakeClient(decide=oracle_decide), sdk=FAKE_SDK, max_workers=1)
        seed = Criteria("How much will this reader enjoy this joke?", ("a", "b", "c"))
        refs = References.build(ratings, seed=SEED, k=3)
        selected = [1, 2, 3, 4]
        with tempfile.TemporaryDirectory() as d:
            row, ev = optimize_user(
                scorer,
                ratings=ratings,
                jokes=jokes,
                user_id=2,
                seed_criteria=seed,
                reflection_lm=FakeReflectionLM(),
                run_dir=Path(d) / "u2",
                references=refs,
                selected=selected,
                generic={2: 0.5},
                max_metric_calls=30,
                reflection_minibatch_size=2,
                seed=SEED,
                log=lambda _m: None,
            )
            self.assertTrue((Path(d) / "u2" / "best_criteria.json").exists())
            self.assertTrue((Path(d) / "u2" / "users.csv").exists())
            self.assertEqual(row["user_id"], 2)
            self.assertEqual(row["generic"], 0.5)
            self.assertIn(row["partner_id"], {1, 3, 4})
            self.assertGreater(row["num_candidates"], 1)
            self.assertEqual(ev.truth.shape, (12, 40))
            table = Path(d) / "users.csv"
            append_row(table, row)
            append_row(table, {**row, "user_id": 3})
            self.assertEqual(done_users(table), {2, 3})

    def test_split_and_partner(self) -> None:
        items = list(range(12))
        train, val = split_groups(items)  # type: ignore[arg-type]
        self.assertEqual((len(train), len(val)), (9, 3))
        selected = [10, 20, 30, 40, 50]
        partners = {u: partner_of(u, selected) for u in selected}
        self.assertEqual(sorted(partners.values()), selected)  # a permutation
        self.assertTrue(all(u != p for u, p in partners.items()))


if __name__ == "__main__":
    unittest.main()
