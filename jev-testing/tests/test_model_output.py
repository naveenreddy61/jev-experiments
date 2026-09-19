from __future__ import annotations

import json
import unittest

from jev_eval.schema import InvalidOutput, ModelOutput, parse_model_output


class ParseModelOutputTests(unittest.TestCase):
    def test_valid_object(self) -> None:
        out = parse_model_output('{"answer": true, "p_yes": 0.87}')
        self.assertIsInstance(out, ModelOutput)
        assert isinstance(out, ModelOutput)
        self.assertTrue(out.answer)
        self.assertAlmostEqual(out.p_yes, 0.87)

    def test_valid_false(self) -> None:
        out = parse_model_output('{"answer": false, "p_yes": 0.0}')
        self.assertIsInstance(out, ModelOutput)
        assert isinstance(out, ModelOutput)
        self.assertFalse(out.answer)
        self.assertEqual(out.p_yes, 0.0)

    def test_integer_p_yes_ok(self) -> None:
        out = parse_model_output('{"answer": true, "p_yes": 1}')
        self.assertIsInstance(out, ModelOutput)
        assert isinstance(out, ModelOutput)
        self.assertEqual(out.p_yes, 1.0)

    def test_fenced_json(self) -> None:
        raw = "```json\n{\"answer\": false, \"p_yes\": 0.12}\n```"
        out = parse_model_output(raw)
        self.assertIsInstance(out, ModelOutput)
        assert isinstance(out, ModelOutput)
        self.assertFalse(out.answer)
        self.assertAlmostEqual(out.p_yes, 0.12)

    def test_prose_then_object(self) -> None:
        raw = 'Sure.\n{"answer": true, "p_yes": 0.5}\n'
        out = parse_model_output(raw)
        self.assertIsInstance(out, ModelOutput)

    def test_empty_invalid(self) -> None:
        out = parse_model_output("  ")
        self.assertIsInstance(out, InvalidOutput)
        assert isinstance(out, InvalidOutput)
        self.assertEqual(out.reason, "empty output")

    def test_garbage_invalid(self) -> None:
        out = parse_model_output("not json at all")
        self.assertIsInstance(out, InvalidOutput)
        assert isinstance(out, InvalidOutput)
        self.assertEqual(out.reason, "output is not JSON")

    def test_missing_p_yes_not_imputed(self) -> None:
        out = parse_model_output('{"answer": true}')
        self.assertIsInstance(out, InvalidOutput)
        assert isinstance(out, InvalidOutput)
        self.assertIn("p_yes", out.reason)

    def test_missing_answer_not_imputed(self) -> None:
        out = parse_model_output('{"p_yes": 0.4}')
        self.assertIsInstance(out, InvalidOutput)

    def test_string_answer_invalid(self) -> None:
        out = parse_model_output('{"answer": "true", "p_yes": 0.9}')
        self.assertIsInstance(out, InvalidOutput)
        assert isinstance(out, InvalidOutput)
        self.assertIn("boolean", out.reason)

    def test_bool_p_yes_invalid(self) -> None:
        # JSON true must not be accepted as 1.0
        out = parse_model_output('{"answer": true, "p_yes": true}')
        self.assertIsInstance(out, InvalidOutput)

    def test_p_yes_out_of_range(self) -> None:
        out = parse_model_output('{"answer": true, "p_yes": 1.2}')
        self.assertIsInstance(out, InvalidOutput)
        assert isinstance(out, InvalidOutput)
        self.assertIn("[0, 1]", out.reason)

    def test_none_invalid(self) -> None:
        out = parse_model_output(None)
        self.assertIsInstance(out, InvalidOutput)

    def test_to_dict_contract(self) -> None:
        out = parse_model_output('{"answer": true, "p_yes": 0.87}')
        assert isinstance(out, ModelOutput)
        self.assertEqual(out.to_dict(), {"answer": True, "p_yes": 0.87})
        self.assertEqual(json.dumps(out.to_dict()), '{"answer": true, "p_yes": 0.87}')


if __name__ == "__main__":
    unittest.main()
