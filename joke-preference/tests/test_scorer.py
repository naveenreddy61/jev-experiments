from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from joke_pref.criteria import COMPONENTS, Criteria, load_criteria, parse_level, save_criteria
from joke_pref.scorer import FatalScorerError, JevScorer, ScoreCache, average, parse_answer
from tests.fakes import FAKE_SDK, FakeAuthError, FakeClient, FakeRateLimitError, fake_response

CRIT = Criteria("How much will this reader enjoy this joke?", ("flat", "works", "lands hard"))


class CriteriaTests(unittest.TestCase):
    def test_roundtrip_candidate(self) -> None:
        cand = CRIT.to_candidate()
        self.assertEqual(list(cand), list(COMPONENTS))
        back = Criteria.from_candidate(cand, CRIT.instructions)
        self.assertEqual(back, CRIT)

    def test_object_level(self) -> None:
        obj = {"what": "a pun", "examples": ["a", "b"]}
        c = Criteria(CRIT.instructions, ("flat", obj, "great"))
        cand = c.to_candidate()
        self.assertTrue(cand["level_good"].startswith("{"))
        self.assertEqual(Criteria.from_candidate(cand, c.instructions).levels[1], obj)
        self.assertEqual(parse_level("  {not json  "), "{not json")
        self.assertEqual(parse_level("{}"), "{}")

    def test_validation_and_file(self) -> None:
        with self.assertRaises(ValueError):
            Criteria("q", ("a", "b"))
        with self.assertRaises(ValueError):
            Criteria("q", ("a", " ", "c"))
        d = Path(tempfile.mkdtemp())
        save_criteria(CRIT, d / "c.json")
        self.assertEqual(load_criteria(d / "c.json"), CRIT)


class ParseAnswerTests(unittest.TestCase):
    def test_int_and_str_keys(self) -> None:
        r = parse_answer(fake_response({0: 0.2, 1: 0.5, 2: 0.3}), 12.0)
        self.assertEqual(r.probabilities, (0.2, 0.5, 0.3))
        self.assertAlmostEqual(r.score, 1.1)
        self.assertEqual(r.argmax, 1)
        self.assertEqual(r.input_tokens, 50)
        r2 = parse_answer(fake_response({"0": 0.0, "1": 0.0, "2": 1.0}), 1.0)
        self.assertEqual(r2.probabilities, (0.0, 0.0, 1.0))

    def test_invalid_shapes(self) -> None:
        self.assertFalse(parse_answer(fake_response({0: 0.5, 1: 0.5}), 1.0).valid)
        self.assertFalse(parse_answer(fake_response({0: 0.9, 1: 0.9, 2: 0.9}), 1.0).valid)
        self.assertFalse(parse_answer(fake_response({0: 1.0, 1: 0, 2: 0}, key="other"), 1.0).valid)


class ScorerTests(unittest.TestCase):
    def test_sends_score_question_with_levels(self) -> None:
        client = FakeClient()
        scorer = JevScorer(client=client, sdk=FAKE_SDK, max_workers=1)
        r = scorer.score_one("setup\n\npunch", CRIT)
        self.assertTrue(r.valid)
        state, question = client.calls[0]
        self.assertEqual(state, "setup\n\npunch")
        self.assertEqual(question.instructions, CRIT.instructions)
        self.assertEqual(question.criteria, ["flat", "works", "lands hard"])
        self.assertEqual(scorer.calls, 1)

    def test_auth_error_is_fatal(self) -> None:
        scorer = JevScorer(client=FakeClient(error=FakeAuthError("bad key")), sdk=FAKE_SDK)
        with self.assertRaises(FatalScorerError):
            scorer.score_one("j", CRIT)

    def test_rate_limit_is_invalid_not_fatal(self) -> None:
        scorer = JevScorer(client=FakeClient(error=FakeRateLimitError("slow down")), sdk=FAKE_SDK)
        r = scorer.score_one("j", CRIT)
        self.assertFalse(r.valid)
        self.assertIn("FakeRateLimitError", r.error or "")

    def test_cache_hits_skip_the_network(self) -> None:
        client = FakeClient()
        cache = ScoreCache(Path(tempfile.mkdtemp()) / "c.sqlite")
        scorer = JevScorer(client=client, sdk=FAKE_SDK, cache=cache, max_workers=4)
        scorer.score_many(["a", "b", "a"], CRIT)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(cache), 2)
        scorer.score_many(["a", "b"], CRIT)
        self.assertEqual(len(client.calls), 2)
        other = Criteria(CRIT.instructions, ("x", "y", "z"))
        scorer.score_one("a", other)
        self.assertEqual(len(client.calls), 3)

    def test_repeats_average(self) -> None:
        seq = iter([{0: 1.0, 1: 0.0, 2: 0.0}, {0: 0.0, 1: 0.0, 2: 1.0}])
        client = FakeClient(decide=lambda j, q: next(seq))
        scorer = JevScorer(client=client, sdk=FAKE_SDK, repeats=2, max_workers=1)
        r = scorer.score_one("j", CRIT)
        self.assertEqual(r.probabilities, (0.5, 0.0, 0.5))
        self.assertAlmostEqual(r.score, 1.0)

    def test_average_all_invalid(self) -> None:
        from joke_pref.scorer import invalid

        self.assertFalse(average([invalid("a"), invalid("b")]).valid)


if __name__ == "__main__":
    unittest.main()
