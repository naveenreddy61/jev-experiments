"""Offline tests for the LLM judge. The HTTP call is a fake."""

from __future__ import annotations

import unittest

from joke_pref.criteria import Criteria
from joke_pref.llm_judge import LLMJudge, build_prompt, parse_answer, parse_score, soft_probabilities

CRIT = Criteria("How much will this reader enjoy this joke?", ("dull", {"what": "fine"}, "sharp"))


def fake_post(url, payload, *, headers, timeout):
    prompt = payload["messages"][0]["content"]
    assert "[bad]\ndull" in prompt and '"what": "fine"' in prompt and "[great]\nsharp" in prompt
    assert payload.get("thinking") == {"type": "disabled"}
    word = "Great" if "pun" in prompt else ("good" if "story" in prompt else "meh")
    return {
        "choices": [{"message": {"content": word}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 2, "completion_tokens_details": {"reasoning_tokens": 0}},
    }


class JudgeTests(unittest.TestCase):
    def test_parse(self) -> None:
        self.assertEqual(parse_answer("Great."), 2)
        self.assertEqual(parse_answer("I would say: good"), 1)
        self.assertEqual(parse_answer("BAD"), 0)
        self.assertIsNone(parse_answer("hmm"))
        self.assertIn("Answer with exactly one word: bad, good, great.", build_prompt(CRIT, "x"))

    def test_score_mode(self) -> None:
        self.assertEqual(parse_score("Answer: 75"), 1.5)
        self.assertEqual(parse_score("120"), 2.0)
        self.assertIsNone(parse_score("none"))
        self.assertEqual(soft_probabilities(1.5), (0.0, 0.5, 0.5))
        self.assertEqual(soft_probabilities(2.0), (0.0, 0.0, 1.0))
        self.assertEqual(soft_probabilities(0.0), (1.0, 0.0, 0.0))
        self.assertIn("0 to 100", build_prompt(CRIT, "x", "score"))

        def post(url, payload, *, headers, timeout):
            return {"choices": [{"message": {"content": "80"}}], "usage": {}}

        judge = LLMJudge(post=post, answer="score", max_workers=1)
        r = judge.score_many(["a"], CRIT)[0]
        self.assertAlmostEqual(r.score, 1.6)
        self.assertAlmostEqual(r.probabilities[2], 0.6)

    def test_score_many(self) -> None:
        judge = LLMJudge(post=fake_post, max_workers=2)
        results = judge.score_many(["a pun", "a story", "nothing", "a pun"], CRIT)
        self.assertEqual([r.score for r in results], [2.0, 1.0, None, 2.0])
        self.assertEqual(results[0].probabilities, (0.0, 0.0, 1.0))
        self.assertFalse(results[2].valid)
        self.assertEqual(judge.calls, 3)
        self.assertEqual(judge.usage()["input_tokens"], 300)


if __name__ == "__main__":
    unittest.main()
