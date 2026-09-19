from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from joke_pref.data import (
    append_label,
    joke_text,
    labeled_premises,
    load_labels,
    load_premises,
    make_splits,
    validate_user,
)
from tests.fakes import write_sample_premises


class PremiseTests(unittest.TestCase):
    def test_load_and_ids(self) -> None:
        premises = load_premises(write_sample_premises())
        self.assertEqual([p.id for p in premises], ["t-001", "t-002", "t-003"])
        self.assertEqual(premises[0].punchlines[1].id, "t-001.1")
        self.assertEqual(premises[0].punchline("t-001.2").style, "deadpan")

    def test_joke_text_has_blank_line(self) -> None:
        p = load_premises(write_sample_premises())[1]
        self.assertEqual(
            joke_text(p, p.punchlines[0]),
            "My therapist says I have a preoccupation with vengeance.\n\nWe'll see about that.",
        )

    def test_rejects_duplicate_id(self) -> None:
        d = Path(tempfile.mkdtemp())
        path = write_sample_premises(d)
        with open(path, "a") as fh:
            fh.write(path.read_text().splitlines()[0] + "\n")
        with self.assertRaises(ValueError):
            load_premises(path)


class LabelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())

    def test_latest_wins_and_skip_removes(self) -> None:
        append_label(self.dir, "Nav", "t-001.0", "bad")
        append_label(self.dir, "nav", "t-001.0", "great")
        append_label(self.dir, "nav", "t-001.1", "good")
        append_label(self.dir, "nav", "t-001.1", "skip")
        self.assertEqual(load_labels(self.dir, "nav"), {"t-001.0": "great"})
        self.assertTrue((self.dir / "nav.jsonl").exists())

    def test_user_validation(self) -> None:
        self.assertEqual(validate_user(" Naveen "), "naveen")
        for bad in ("", "../x", "a b", "x" * 40):
            with self.assertRaises(ValueError):
                validate_user(bad)
        with self.assertRaises(ValueError):
            append_label(self.dir, "nav", "t-001.0", "meh")

    def test_labeled_premises_keeps_only_labeled(self) -> None:
        premises = load_premises(write_sample_premises())
        labels = {"t-001.0": "bad", "t-001.2": "great", "t-003.1": "good"}
        items = labeled_premises(premises, labels)
        self.assertEqual([it.id for it in items], ["t-001", "t-003"])
        self.assertEqual([(p.id, lab) for p, lab in items[0].items()], [("t-001.0", 0), ("t-001.2", 2)])
        self.assertEqual(len(labeled_premises(premises, labels, min_labeled=2)), 1)


class SplitTests(unittest.TestCase):
    def test_deterministic_and_disjoint(self) -> None:
        ids = [f"p-{i:03d}" for i in range(100)]
        a = make_splits(ids, seed=0)
        b = make_splits(reversed(ids), seed=0)
        self.assertEqual(a, b)
        self.assertNotEqual(a, make_splits(ids, seed=1))
        self.assertEqual(len(a.train) + len(a.val) + len(a.test), 100)
        self.assertEqual(len(set(a.train) & set(a.test)), 0)
        self.assertEqual(len(a.train), 58)
        self.assertEqual(len(a.val), 17)
        self.assertEqual(len(a.ids("all")), 100)


if __name__ == "__main__":
    unittest.main()
