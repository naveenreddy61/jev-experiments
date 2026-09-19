"""Evaluate a criteria on a split, and run the GEPA optimization."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from joke_pref.adapter import JokeCriteriaAdapter, reflection_prompt_templates
from joke_pref.criteria import Criteria, save_criteria
from joke_pref.data import LabeledPremise, joke_text
from joke_pref.metric import W_RANK, PremiseEval, Row, premise_score, summarize
from joke_pref.scorer import JevScorer


def evaluate(
    scorer: JevScorer,
    items: list[LabeledPremise],
    criteria: Criteria,
    *,
    w_rank: float = W_RANK,
) -> tuple[list[Row], list[PremiseEval], dict[str, Any]]:
    flat = [(lp, p, lab) for lp in items for (p, lab) in lp.items()]
    results = scorer.score_many([joke_text(lp.premise, p) for lp, p, _ in flat], criteria)
    rows = [
        Row(premise_id=lp.id, punchline_id=p.id, label=lab, result=r)
        for (lp, p, lab), r in zip(flat, results)
    ]
    evals: list[PremiseEval] = []
    for lp in items:
        mine = [r for r in rows if r.premise_id == lp.id]
        evals.append(
            premise_score(
                [r.result for r in mine],
                [r.label for r in mine],
                [r.punchline_id for r in mine],
                w_rank=w_rank,
            )
        )
    return rows, evals, summarize(rows, evals)


def write_evaluation(
    out_dir: str | Path,
    *,
    rows: list[Row],
    summary: dict[str, Any],
    criteria: Criteria,
    meta: dict[str, Any],
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rows.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(
                json.dumps(
                    {
                        "premise_id": r.premise_id,
                        "punchline_id": r.punchline_id,
                        "label": r.label,
                        **r.result.as_dict(),
                    }
                )
                + "\n"
            )
    save_criteria(criteria, out / "criteria.json")
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump({**meta, **summary}, fh, indent=2)
        fh.write("\n")
    return out


def optimize(
    scorer: JevScorer,
    *,
    train: list[LabeledPremise],
    val: list[LabeledPremise],
    seed_criteria: Criteria,
    reflection_lm: Callable[[Any], str],
    max_metric_calls: int,
    run_dir: str | Path,
    seed: int = 0,
    reflection_minibatch_size: int = 12,
    w_rank: float = W_RANK,
    joint: bool = True,
    log: Callable[[str], None] = print,
) -> tuple[Criteria, Any]:
    """Run gepa.optimize over the three level descriptions.

    Returns the best criteria by mean validation score, and the raw result.
    """
    import gepa

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter = JokeCriteriaAdapter(
        scorer,
        seed_criteria.instructions,
        w_rank=w_rank,
        reflection_lm=reflection_lm if joint else None,
    )
    t0 = time.perf_counter()
    result = gepa.optimize(
        seed_candidate=seed_criteria.to_candidate(),
        trainset=train,
        valset=val,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_prompt_template=None if joint else reflection_prompt_templates(),
        reflection_minibatch_size=reflection_minibatch_size,
        candidate_selection_strategy="pareto",
        module_selector="all" if joint else "round_robin",
        max_metric_calls=max_metric_calls,
        run_dir=str(run_dir),
        seed=seed,
        track_best_outputs=False,
        display_progress_bar=False,
        raise_on_exception=True,
    )
    elapsed = time.perf_counter() - t0
    best = Criteria.from_candidate(result.best_candidate, seed_criteria.instructions)
    save_criteria(best, run_dir / "best_criteria.json")
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(elapsed, 1),
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "num_full_val_evals": result.num_full_val_evals,
        "best_idx": result.best_idx,
        "val_aggregate_scores": list(getattr(result, "val_aggregate_scores", []) or []),
        "joint_reflection": joint,
        "reflection_minibatch_size": reflection_minibatch_size,
        "w_rank": w_rank,
        "proposal_parse_failures": adapter.proposal_failures,
        "jev_calls": scorer.calls,
        "reflection_calls": getattr(reflection_lm, "calls", None),
        "reflection_input_tokens": getattr(reflection_lm, "input_tokens", None),
        "reflection_output_tokens": getattr(reflection_lm, "output_tokens", None),
    }
    with open(run_dir / "optimize_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    log(f"best candidate {result.best_idx} of {result.num_candidates}; {result.total_metric_calls} metric calls; {elapsed:.0f}s")
    if result.num_candidates <= 1:
        log(
            "WARNING: GEPA produced no new candidate. Check run_log.txt in the run dir for "
            "reflection errors; the result equals the seed."
        )
    return best, result
