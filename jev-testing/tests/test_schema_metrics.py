from __future__ import annotations

import unittest

from jev_eval.metrics import brier_score, expected_calibration_error, score_rows
from jev_eval.schema import SchemaError, confidence, item_from_dict


def _item(**overrides):
    base = {
        "id": "python.parse.00001",
        "task": "python.parse",
        "difficulty": "easy",
        "artifact": "x = 1\n",
        "question": "Does this parse as Python 3.12?",
        "answer": True,
        "fault_tags": [],
        "oracle_meta": {"tool": "ast.parse", "version_note": "test", "exit_code": 0},
    }
    base.update(overrides)
    return base


class SchemaTests(unittest.TestCase):
    def test_valid_yes(self) -> None:
        item = item_from_dict(_item())
        self.assertTrue(item.answer)
        self.assertEqual(item.fault_tags, ())

    def test_valid_no_requires_tags(self) -> None:
        item = item_from_dict(_item(id="python.parse.00002", answer=False, fault_tags=["indent_error"]))
        self.assertFalse(item.answer)
        self.assertEqual(item.fault_tags, ("indent_error",))

    def test_yes_with_tags_rejected(self) -> None:
        with self.assertRaises(SchemaError):
            item_from_dict(_item(fault_tags=["indent_error"]))

    def test_no_without_tags_rejected(self) -> None:
        with self.assertRaises(SchemaError):
            item_from_dict(_item(answer=False, fault_tags=[]))

    def test_wrong_question_rejected(self) -> None:
        with self.assertRaises(SchemaError):
            item_from_dict(_item(question="Will this run?"))

    def test_answer_must_be_bool(self) -> None:
        with self.assertRaises(SchemaError):
            item_from_dict(_item(answer="true"))


class MetricsTests(unittest.TestCase):
    def test_confidence(self) -> None:
        self.assertAlmostEqual(confidence(0.87), 0.87)
        self.assertAlmostEqual(confidence(0.1), 0.9)

    def test_accuracy_ignores_invalid(self) -> None:
        rows = [
            {"task": "python.parse", "gold": True, "answer": True, "p_yes": 0.9, "invalid": False, "latency_ms": 10},
            {"task": "python.parse", "gold": False, "answer": True, "p_yes": 0.8, "invalid": False, "latency_ms": 12},
            {"task": "python.parse", "gold": True, "answer": None, "p_yes": None, "invalid": True, "latency_ms": 5},
        ]
        summary = score_rows(rows)
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["n_invalid"], 1)
        self.assertAlmostEqual(summary["invalid_rate"], 1 / 3)
        self.assertAlmostEqual(summary["accuracy"], 0.5)
        self.assertIsNotNone(summary["calibration"]["brier"])

    def test_brier_and_ece_perfect(self) -> None:
        golds = [True, True, False, False]
        ps = [1.0, 1.0, 0.0, 0.0]
        self.assertEqual(brier_score(golds, ps), 0.0)
        ece, table = expected_calibration_error(golds, ps, bins=2)
        self.assertIsNotNone(ece)
        self.assertAlmostEqual(ece, 0.0)
        self.assertEqual(len(table), 2)

    def test_empty_rows(self) -> None:
        summary = score_rows([])
        self.assertEqual(summary["n"], 0)
        self.assertIsNone(summary["accuracy"])
        self.assertEqual(summary["invalid_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
