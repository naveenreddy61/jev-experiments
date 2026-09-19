from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from jev_eval.schema import load_items
from jev_eval.tasks import TASK_IDS

SEED = Path(__file__).resolve().parents[1] / "data" / "seed"


class SeedSchemaTests(unittest.TestCase):
    def test_seed_files_match_schema_and_counts(self) -> None:
        paths = sorted(SEED.glob("*.jsonl"))
        self.assertEqual({p.name for p in paths}, {f"{tid}.jsonl" for tid in TASK_IDS})
        items = load_items(paths)
        self.assertEqual(len(items), 200)
        by_task = Counter(it.task for it in items)
        for tid in TASK_IDS:
            self.assertEqual(by_task[tid], 50, tid)
            subset = [it for it in items if it.task == tid]
            n_yes = sum(1 for it in subset if it.answer)
            self.assertEqual(n_yes, 25, tid)
            self.assertTrue(all(it.fault_tags or it.answer for it in subset))
            self.assertGreaterEqual(len({it.difficulty for it in subset}), 3)
