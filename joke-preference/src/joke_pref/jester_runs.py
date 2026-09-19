"""Phase 2 of the Jester plan: Jev + GEPA for one user, then the controls.

`optimize_user` runs GEPA on one user's 60 train jokes (12 groups of 5; the
last 3 groups are the validation set GEPA needs, the 40 test jokes stay
untouched), scores the best rubric on all 100 jokes, and evaluates it for
every user. The row for the user is the personal result. The rows for the
other selected users are the cross-user control: the same rubric read for a
person it was not written for.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from joke_pref.criteria import Criteria, save_criteria
from joke_pref.data import LabeledPremise, shuffle_labels
from joke_pref.jester import Joke, Ratings, user_items
from joke_pref.jester_eval import (
    K_NEIGHBOURS,
    SEED,
    RubricEval,
    evaluate_rubric,
    item_split,
    knn_predict,
    loo_item_mean,
    user_mask,
    write_user_table,
)
from joke_pref.runs import optimize
from joke_pref.scorer import JevScorer

N_VAL_GROUPS = 3

COLUMNS = (
    "user_id",
    "personal",
    "generic",
    "cross_user_median",
    "cross_user_partner",
    "partner_id",
    "crowd",
    "knn",
    "personal_agreement",
    "num_candidates",
    "metric_calls",
    "reflection_calls",
    "jev_calls",
    "elapsed_s",
    "null_run",
)


@dataclass
class References:
    """Rating-based predictions on the test jokes, computed once per process."""

    crowd_pred: np.ndarray
    knn_pred: np.ndarray

    @classmethod
    def build(cls, ratings: Ratings, *, seed: int = SEED, k: int = K_NEIGHBOURS) -> "References":
        train_ids, test_ids = item_split(seed)
        train = ratings.columns(train_ids)
        test = ratings.columns(test_ids)
        return cls(
            crowd_pred=loo_item_mean(ratings.matrix)[:, test],
            knn_pred=knn_predict(ratings.matrix, train, test, k),
        )


def split_groups(items: list[LabeledPremise], n_val: int = N_VAL_GROUPS) -> tuple[list[LabeledPremise], list[LabeledPremise]]:
    """The groups are already shuffled per user by `user_items`, so the last
    `n_val` groups are a fixed random validation set."""
    if len(items) <= n_val + 1:
        raise ValueError(f"need more than {n_val + 1} groups, got {len(items)}")
    return items[:-n_val], items[-n_val:]


def load_generic(baseline_dir: str | Path) -> dict[int, float]:
    """Per-user concordance of the generic rubric from a `jester-evaluate` run."""
    with open(Path(baseline_dir) / "users.csv", newline="") as fh:
        return {int(r["user_id"]): float(r["rubric_concordance"]) for r in csv.DictReader(fh)}


def partner_of(user_id: int, selected: Sequence[int]) -> int:
    """A fixed random pairing of the selected users: the next user in the
    seeded permutation of the sorted list."""
    order = list(np.array(sorted(selected))[np.random.default_rng(SEED).permutation(len(selected))])
    i = order.index(user_id)
    return int(order[(i + 1) % len(order)])


def optimize_user(
    scorer: JevScorer,
    *,
    ratings: Ratings,
    jokes: Sequence[Joke],
    user_id: int,
    seed_criteria: Criteria,
    reflection_lm: Callable[[Any], str],
    run_dir: str | Path,
    references: References,
    selected: Sequence[int],
    generic: dict[int, float],
    max_metric_calls: int = 600,
    reflection_minibatch_size: int = 6,
    seed: int = SEED,
    null_run: bool = False,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, Any], RubricEval]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    calls0 = scorer.calls
    items = user_items(ratings, jokes, user_id, "train", seed=seed)
    train, val = split_groups(items)
    if null_run:
        train = shuffle_labels(train, seed=seed)
        val = shuffle_labels(val, seed=seed + 1)
    log(f"user {user_id}: {len(train)} train groups, {len(val)} val groups, null_run={null_run}")
    best, result = optimize(
        scorer,
        train=train,
        val=val,
        seed_criteria=seed_criteria,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        run_dir=run_dir / "gepa",
        seed=seed,
        reflection_minibatch_size=reflection_minibatch_size,
        joint=True,
        log=log,
    )
    save_criteria(best, run_dir / "best_criteria.json")
    results = scorer.score_many([j.text for j in jokes], best)
    ev = evaluate_rubric(
        results, jokes, ratings, seed=seed, crowd_pred=references.crowd_pred, knn_pred=references.knn_pred
    )
    with open(run_dir / "jokes.jsonl", "w", encoding="utf-8") as fh:
        for j, r in zip(jokes, results):
            fh.write(json.dumps({"joke_id": j.id, **r.as_dict()}, ensure_ascii=False) + "\n")
    write_user_table(run_dir / "users.csv", ev)

    me = int(np.flatnonzero(ratings.user_ids == user_id)[0])
    others = user_mask(ratings, [u for u in selected if u != user_id])
    partner = partner_of(user_id, selected)
    partner_row = int(np.flatnonzero(ratings.user_ids == partner)[0])
    row: dict[str, Any] = {
        "user_id": user_id,
        "personal": float(ev.rubric_conc[me]),
        "generic": generic.get(user_id, float("nan")),
        "cross_user_median": float(np.nanmedian(ev.rubric_conc[others])) if others.any() else float("nan"),
        "cross_user_partner": float(ev.rubric_conc[partner_row]),
        "partner_id": partner,
        "crowd": float(ev.crowd_conc[me]),
        "knn": float(ev.knn_conc[me]),
        "personal_agreement": float(ev.rubric_agreement[me]),
        "num_candidates": result.num_candidates,
        "metric_calls": result.total_metric_calls,
        "reflection_calls": getattr(reflection_lm, "calls", None),
        "jev_calls": scorer.calls - calls0,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "null_run": null_run,
    }
    (run_dir / "row.json").write_text(json.dumps(row, indent=2) + "\n")
    log(
        f"user {user_id}: personal {row['personal']:.3f}  generic {row['generic']:.3f}  "
        f"cross-user median {row['cross_user_median']:.3f}  crowd {row['crowd']:.3f}  kNN {row['knn']:.3f}  "
        f"({row['num_candidates']} candidates, {row['jev_calls']} Jev calls, {row['elapsed_s']}s)"
    )
    return row, ev


def append_row(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in COLUMNS})


def done_users(path: str | Path) -> set[int]:
    path = Path(path)
    if not path.exists():
        return set()
    with open(path, newline="") as fh:
        return {int(r["user_id"]) for r in csv.DictReader(fh)}
