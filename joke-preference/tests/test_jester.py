"""Offline tests for the Jester loaders. No download happens here."""

from __future__ import annotations

import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from joke_pref.jester import (
    GAUGE_JOKES,
    N_JOKES,
    complete_users,
    item_split,
    load_jokes,
    load_ratings,
)

SYNTHETIC = [
    (1, 1, -5.0),
    (1, 2, 2.5),
    (1, 3, 9.0),
    (2, 1, 0.0),
    (2, 3, -1.25),
    (3, 2, 10.0),
]


class JesterLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.ratings_path = self.dir / "ratings.csv.gz"
        with gzip.open(self.ratings_path, "wt") as f:
            f.write("user_id,joke_id,rating\n")
            for u, j, r in SYNTHETIC:
                f.write(f"{u},{j},{r}\n")
        self.jokes_path = self.dir / "jokes.jsonl"
        with self.jokes_path.open("w") as f:
            for i, text in enumerate(["first joke", "second joke", "third joke"], 1):
                f.write(json.dumps({"id": i, "text": text}) + "\n")

    def test_jokes_load_in_id_order(self) -> None:
        jokes = load_jokes(self.jokes_path)
        self.assertEqual([j.id for j in jokes], [1, 2, 3])
        self.assertEqual(jokes[0].text, "first joke")
        self.assertEqual(jokes[0].n_words, 2)

    def test_matrix_shape_and_missing_values(self) -> None:
        r = load_ratings(self.ratings_path, n_jokes=3)
        self.assertEqual(r.matrix.shape, (3, 3))
        self.assertEqual(r.user_ids.tolist(), [1, 2, 3])
        self.assertEqual(r.joke_ids.tolist(), [1, 2, 3])
        self.assertEqual(r.matrix[0].tolist(), [-5.0, 2.5, 9.0])
        self.assertTrue(math.isnan(r.matrix[1, 1]))
        self.assertEqual(int(np.isnan(r.matrix).sum()), 3)

    def test_complete_users_keeps_only_full_rows(self) -> None:
        r = complete_users(load_ratings(self.ratings_path, n_jokes=3))
        self.assertEqual(r.user_ids.tolist(), [1])
        self.assertEqual(r.matrix.shape, (1, 3))

    def test_column_lookup(self) -> None:
        r = load_ratings(self.ratings_path, n_jokes=3)
        self.assertEqual(r.column(2), 1)
        self.assertEqual(r.columns([3, 1]).tolist(), [2, 0])
        with self.assertRaises(KeyError):
            r.column(9)

    def test_bad_header_is_rejected(self) -> None:
        bad = self.dir / "bad.csv.gz"
        with gzip.open(bad, "wt") as f:
            f.write("a,b,c\n1,1,0.0\n")
        with self.assertRaises(ValueError):
            load_ratings(bad, n_jokes=3)

    def test_missing_file_is_reported(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_ratings(self.dir / "nope.csv.gz")


class ItemSplitTest(unittest.TestCase):
    def test_split_is_fixed_and_covers_every_joke(self) -> None:
        train, test = item_split(seed=7)
        self.assertEqual(len(train), 60)
        self.assertEqual(len(test), 40)
        self.assertEqual(sorted(train + test), list(range(1, N_JOKES + 1)))
        self.assertTrue(set(GAUGE_JOKES) <= set(train))
        self.assertEqual(set(GAUGE_JOKES) & set(test), set())

    def test_split_is_deterministic(self) -> None:
        self.assertEqual(item_split(seed=7), item_split(seed=7))
        self.assertNotEqual(item_split(seed=7), item_split(seed=8))


if __name__ == "__main__":
    unittest.main()
