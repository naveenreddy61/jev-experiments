from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jev_eval.cli import main


def _run(*argv: str) -> int:
    return main(list(argv))


class DifficultyFilterTests(unittest.TestCase):
    """`run --difficulty` selects a subset without touching the data files."""

    def _run_into(self, tmp: str, *extra: str) -> tuple[int, Path]:
        out = Path(tmp) / "out"
        code = _run(
            "run", "--provider", "jev", "--mock",
            "--data", "data/seed", "--out", str(out), *extra,
        )
        return code, out

    def test_hard_only_is_a_strict_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code_all, out_all = self._run_into(tmp)
            self.assertEqual(code_all, 0)
            n_all = json.loads((out_all / "run_meta.json").read_text())["n"]

        with tempfile.TemporaryDirectory() as tmp:
            code, out = self._run_into(tmp, "--difficulty", "hard")
            self.assertEqual(code, 0)
            meta = json.loads((out / "run_meta.json").read_text())
            rows = [json.loads(l) for l in (out / "predictions.jsonl").read_text().splitlines()]

        self.assertEqual(meta["difficulty"], ["hard"])
        self.assertLess(meta["n"], n_all)
        self.assertEqual(meta["n"], len(rows))

    def test_two_difficulties_combine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, out_hard = self._run_into(tmp, "--difficulty", "hard")
            n_hard = json.loads((out_hard / "run_meta.json").read_text())["n"]
        with tempfile.TemporaryDirectory() as tmp:
            _, out_both = self._run_into(tmp, "--difficulty", "hard", "--difficulty", "easy")
            n_both = json.loads((out_both / "run_meta.json").read_text())["n"]
        self.assertGreater(n_both, n_hard)

    def test_unfiltered_run_records_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, out = self._run_into(tmp)
            self.assertEqual(json.loads((out / "run_meta.json").read_text())["difficulty"], "all")


if __name__ == "__main__":
    unittest.main()
