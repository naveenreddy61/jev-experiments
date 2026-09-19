"""The GEPA adapter: the "program" is three level descriptions sent to Jev.

`evaluate` scores every labeled punchline of every premise in the batch with
Jev, then gives GEPA one score per premise. `make_reflective_dataset` gives
the reflection LM the jokes, the person's labels, Jev's probabilities, and a
plain diagnosis, so it can rewrite one level description at a time.

See https://github.com/gepa-ai/gepa, `gepa.core.adapter.GEPAAdapter`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from gepa.core.adapter import EvaluationBatch

from joke_pref.criteria import COMPONENTS, Criteria
from joke_pref.data import LABELS, LabeledPremise, Punchline, joke_text
from joke_pref.metric import W_RANK, PremiseEval, premise_score
from joke_pref.scorer import JevScorer, ScoreResult


@dataclass
class Scored:
    punchline: Punchline
    label: int
    result: ScoreResult


@dataclass
class Trajectory:
    premise: LabeledPremise
    scored: list[Scored]
    eval: PremiseEval


class JokeCriteriaAdapter:
    def __init__(
        self,
        scorer: JevScorer,
        instructions: str,
        *,
        w_rank: float = W_RANK,
        reflection_lm: Any = None,
    ):
        self.scorer = scorer
        self.instructions = instructions
        self.w_rank = w_rank
        # When a reflection LM is given, the adapter owns the proposal step:
        # one call rewrites all three levels together (see `_propose_joint`).
        # Otherwise GEPA's default per-component proposer runs with
        # `reflection_prompt_templates()`. GEPA reads this attribute on every
        # reflection step, so it must exist even when it is None.
        self.reflection_lm = reflection_lm
        self.propose_new_texts = self._propose_joint if reflection_lm is not None else None
        self.proposal_failures = 0

    # --- GEPAAdapter.evaluate ---------------------------------------------

    def evaluate(
        self,
        batch: list[LabeledPremise],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, dict[str, Any]]:
        criteria = Criteria.from_candidate(candidate, self.instructions)
        flat: list[tuple[int, Punchline, int]] = []
        for bi, lp in enumerate(batch):
            for punchline, label in lp.items():
                flat.append((bi, punchline, label))
        jokes = [joke_text(batch[bi].premise, p) for bi, p, _ in flat]
        results = self.scorer.score_many(jokes, criteria)

        per_premise: list[list[Scored]] = [[] for _ in batch]
        for (bi, punchline, label), result in zip(flat, results):
            per_premise[bi].append(Scored(punchline, label, result))

        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[Trajectory] = []
        for lp, scored in zip(batch, per_premise):
            ev = premise_score(
                [s.result for s in scored],
                [s.label for s in scored],
                [s.punchline.id for s in scored],
                w_rank=self.w_rank,
            )
            scores.append(ev.score)
            outputs.append({s.punchline.id: s.result.as_dict() for s in scored})
            if capture_traces:
                trajectories.append(Trajectory(lp, scored, ev))
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    # --- GEPAAdapter.make_reflective_dataset ------------------------------

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, dict[str, Any]],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        assert eval_batch.trajectories is not None
        out: dict[str, list[dict[str, Any]]] = {}
        for component in components_to_update:
            level = COMPONENTS.index(component)
            out[component] = [
                self._record(traj, level) for traj in eval_batch.trajectories
            ]
        return out

    def _record(self, traj: Trajectory, level: int) -> dict[str, Any]:
        level_name = LABELS[level]
        inputs = {
            "setup": traj.premise.premise.premise,
            "punchlines": [
                {"punchline": s.punchline.text, "reader_label": LABELS[s.label]}
                for s in traj.scored
            ],
        }
        generated = {}
        for s in traj.scored:
            r = s.result
            if r.probabilities is None:
                generated[s.punchline.text] = {"error": r.error}
            else:
                generated[s.punchline.text] = {
                    "expected_level": round(r.score or 0.0, 2),
                    **{f"p_{LABELS[i]}": round(p, 2) for i, p in enumerate(r.probabilities)},
                }
        return {
            "Inputs": inputs,
            "Generated Outputs": generated,
            "Feedback": self._feedback(traj, level, level_name),
        }

    def _feedback(self, traj: Trajectory, level: int, level_name: str) -> str:
        lines: list[str] = []
        for s in traj.scored:
            r = s.result
            if r.probabilities is None:
                lines.append(f'"{s.punchline.text}": model call failed ({r.error}).')
                continue
            p_level = r.probabilities[level]
            truth = LABELS[s.label]
            if s.label == level and p_level < 0.5:
                lines.append(
                    f'"{s.punchline.text}": the reader rated this {truth}, but the model put '
                    f"only {p_level:.2f} on {level_name}. The {level_name} description must "
                    f"cover a joke like this."
                )
            elif s.label != level and p_level >= 0.35:
                lines.append(
                    f'"{s.punchline.text}": the reader rated this {truth}, not {level_name}, '
                    f"but the model put {p_level:.2f} on {level_name}. The {level_name} "
                    f"description must exclude a joke like this."
                )
            else:
                lines.append(
                    f'"{s.punchline.text}": reader {truth}, model p_{level_name}={p_level:.2f}. Fine.'
                )
        by_id = {s.punchline.id: s for s in traj.scored}
        for pair in traj.eval.inverted:
            hi, lo = by_id[pair.hi_id], by_id[pair.lo_id]
            lines.append(
                f'Order error: the reader preferred "{hi.punchline.text}" ({LABELS[hi.label]}) '
                f'over "{lo.punchline.text}" ({LABELS[lo.label]}), but the model scored them '
                f"{pair.hi_pred:.2f} vs {pair.lo_pred:.2f}."
            )
        conc = traj.eval.concordance
        summary = f"Premise score {traj.eval.score:.2f}"
        if conc is not None:
            summary += f", pair order correct {conc:.0%}"
        summary += f", mean probability on the reader's level {traj.eval.agreement:.2f}."
        lines.append(summary)
        return "\n".join(lines)

    # --- joint proposal: all three levels in one reflection call -----------

    def _propose_joint(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Rewrite every level in one call so the levels stay mutually exclusive.

        The reflective records are the same for every component (one per
        premise), so the prompt uses the records of the first component and
        shows the whole current rubric. On a parse failure the candidate is
        returned unchanged and GEPA skips it as "not better".
        """
        records = reflective_dataset[components_to_update[0]]
        prompt = joint_reflection_prompt(candidate, records)
        text = self.reflection_lm(prompt)
        proposal = parse_joint_proposal(text)
        if proposal is None:
            self.proposal_failures += 1
            return dict(candidate)
        out = dict(candidate)
        for component in components_to_update:
            if component in proposal:
                out[component] = proposal[component]
        return out


def joint_reflection_prompt(candidate: dict[str, str], records: Sequence[Mapping[str, Any]]) -> str:
    rubric = "\n\n".join(f"[{name}]\n{candidate[comp]}" for comp, name in zip(COMPONENTS, LABELS))
    blocks = []
    for i, rec in enumerate(records, 1):
        inputs = rec["Inputs"]
        lines = [f"Premise {i}: {inputs['setup']}"]
        for pl in inputs["punchlines"]:
            gen = rec["Generated Outputs"].get(pl["punchline"], {})
            probs = ", ".join(f"{k[2:]} {v:.2f}" for k, v in gen.items() if k.startswith("p_"))
            lines.append(f'  - "{pl["punchline"]}"  reader: {pl["reader_label"]}  model: {probs or gen}')
        lines.append("  feedback:")
        lines.extend("    " + ln for ln in str(rec["Feedback"]).splitlines())
        blocks.append("\n".join(lines))
    side_info = "\n\n".join(blocks)
    n = len(records)
    return f"""A fast scoring model rates jokes for one specific reader. It receives one full joke (premise and punchline) and a rubric with three levels: bad, good, great. Each level is a short description of a situation. The model judges each level on its own against the joke. It never sees the level's number or the other levels, so words such as "better than", "moderately", or numbers mean nothing to it. Describe situations and concrete features, not degrees.

The current rubric is:
```
{rubric}
```

Below are {n} premises with their punchlines, the reader's own label for each punchline, the model's probabilities under the current rubric, and feedback on each:
```
{side_info}
```

Rewrite all three level descriptions together so that, for each punchline, the model's probability rises on the level the reader chose and falls on the other two. Work from the whole set of {n} premises, not from one or two jokes: name a feature only when it separates the reader's labels across several premises. Look for what the reader's bad, good and great jokes have in common: style (pun, absurdist, dark, deadpan, anti-joke, misdirection, wordplay, hyperbole, callback, meta, observational, self-deprecating), mechanism, subject, tone, length, predictability. The three descriptions must not overlap: a feature that marks one level must not appear in another. Keep what already works in the current rubric. Give the great level a real, specific description, since the reader uses it rarely and the model tends to miss it.

Each description may be a plain paragraph, or an object with two fields: "what" (the description) and "examples" (a list of 2 to 4 short traits or example lines). Keep each under 150 words. Do not name the reader. Do not refer to other levels by number.

Answer with one JSON object and nothing else, inside a ``` block:
{{"level_bad": <string or object>, "level_good": <string or object>, "level_great": <string or object>}}"""


def parse_joint_proposal(text: str) -> dict[str, str] | None:
    """Extract the JSON object from the reflection answer. Values become
    candidate strings: a string as is, an object as indented JSON."""
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    raw = m.group(1) if m else text.strip()
    if not m:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < 0:
            return None
        raw = raw[start : end + 1]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    out: dict[str, str] = {}
    for comp in COMPONENTS:
        val = obj.get(comp)
        if isinstance(val, str) and val.strip():
            out[comp] = val.strip()
        elif isinstance(val, dict) and val:
            out[comp] = json.dumps(val, ensure_ascii=False, indent=2)
    return out or None


def reflection_prompt_templates() -> dict[str, str]:
    """One prompt per component. gepa requires `<curr_param>` and `<side_info>`."""
    templates: dict[str, str] = {}
    for component, name in zip(COMPONENTS, LABELS):
        others = ", ".join(n for n in LABELS if n != name)
        templates[component] = f"""A fast scoring model rates jokes for one specific reader. It receives one full joke and a rubric with three levels: bad, good, great. Each level is a short description of a situation. The model judges each level on its own against the joke. It never sees the level's number or the other levels, so words such as "better than", "moderately", or numbers mean nothing to it. Describe situations and concrete features, not degrees.

You are rewriting the description of the level "{name}" (the other levels are {others}). The current description is:
```
<curr_param>
```

Below are jokes with the reader's own label, the model's probabilities under the current rubric, and feedback on each:
```
<side_info>
```

Write a new description for the "{name}" level so that the model's probability for "{name}" rises on jokes the reader labeled {name} and falls on the others. Look for what the reader's {name} jokes have in common: style (pun, absurdist, dark, deadpan, anti-joke, misdirection, wordplay, hyperbole, callback), mechanism, subject, tone, length, predictability. Name those features. Keep what already works in the current description.

You may write a plain paragraph, or a JSON object with two fields: "what" (the description) and "examples" (a list of 2 to 4 short traits or example lines). Keep it under 120 words. Do not name the reader. Do not refer to other levels by number.

Provide the new description within ``` blocks."""
    return templates
