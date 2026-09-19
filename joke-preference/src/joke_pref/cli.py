"""Command line: label, stats, split, probe, evaluate, optimize, jester-evaluate.

Paths are relative to the project directory. Run from `joke-preference/`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from joke_pref.criteria import Criteria, load_criteria
from joke_pref.data import (
    LABELS,
    LabeledPremise,
    labeled_premises,
    list_users,
    load_labels,
    load_premises,
    make_splits,
    select,
    shuffle_labels,
)

DEFAULT_PREMISES = "data/premises.jsonl"
DEFAULT_LABELS_DIR = "data/labels"
DEFAULT_RESULTS = "results"
DEFAULT_SEED_CRITERIA = "criteria/seed.json"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_user_items(args) -> tuple[list[LabeledPremise], dict[str, str]]:
    premises = load_premises(args.premises)
    labels = load_labels(args.labels_dir, args.user)
    if not labels:
        sys.exit(f"no labels for user {args.user!r} in {args.labels_dir}. Run `joke-pref label` first.")
    return labeled_premises(premises, labels), labels


def _splits(items: list[LabeledPremise], seed: int):
    return make_splits([it.id for it in items], seed=seed)


def _scorer(args, *, cache_name: str = "jev-cache.sqlite"):
    from joke_pref.scorer import JevScorer, ScoreCache

    cache = None if getattr(args, "no_cache", False) else ScoreCache(Path(args.results) / cache_name)
    return JevScorer(
        timeout=args.timeout,
        max_workers=args.workers,
        repeats=getattr(args, "repeats", 1),
        cache=cache,
    )


# --- commands ---------------------------------------------------------------


def cmd_label(args) -> int:
    from joke_pref.label_server import serve

    server = serve(args.premises, args.labels_dir, host=args.host, port=args.port)
    print(f"Labeling page: http://{args.host}:{server.server_port}/   (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_stats(args) -> int:
    premises = load_premises(args.premises)
    total = sum(len(p.punchlines) for p in premises)
    users = [args.user] if args.user else list_users(args.labels_dir)
    print(f"{len(premises)} premises, {total} punchlines")
    if not users:
        print("no label files yet")
        return 0
    for user in users:
        labels = load_labels(args.labels_dir, user)
        counts = {name: sum(1 for v in labels.values() if v == name) for name in LABELS}
        items = labeled_premises(premises, labels)
        mixed = sum(1 for it in items if len({lab for _, lab in it.items()}) > 1)
        print(f"{user}: {len(labels)}/{total} labeled; {counts}; {len(items)} premises touched, {mixed} with mixed labels")
    return 0


def cmd_split(args) -> int:
    items, _ = _load_user_items(args)
    splits = _splits(items, args.seed)
    out = Path(args.results) / args.user / "splits.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits.as_dict(), indent=2) + "\n")
    print(f"{len(splits.train)} train / {len(splits.val)} val / {len(splits.test)} test premises (seed {args.seed}) -> {out}")
    return 0


def cmd_probe(args) -> int:
    """Does the criteria text move Jev at all? Score n jokes under each
    criteria file in a directory and report the spread per joke."""
    from joke_pref.data import joke_text

    premises = load_premises(args.premises)
    files = sorted(Path(args.criteria_dir).glob("*.json"))
    if len(files) < 2:
        sys.exit(f"need at least two criteria files in {args.criteria_dir}")
    criteria = {f.stem: load_criteria(f) for f in files}
    jokes = [(p, pl) for p in premises for pl in p.punchlines][:: max(1, (sum(len(p.punchlines) for p in premises) // args.n))][: args.n]
    scorer = _scorer(args)
    table: dict[str, list] = {}
    for name, crit in criteria.items():
        table[name] = scorer.score_many([joke_text(p, pl) for p, pl in jokes], crit)
    out_dir = Path(args.results) / "probe" / _stamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    spreads = []
    lines = []
    for j, (p, pl) in enumerate(jokes):
        scores = {name: table[name][j].score for name in criteria}
        valid = [s for s in scores.values() if s is not None]
        spread = (max(valid) - min(valid)) if valid else None
        spreads.append(spread)
        lines.append({"punchline_id": pl.id, "joke": joke_text(p, pl), "scores": scores, "spread": spread})
    with open(out_dir / "probe.jsonl", "w") as fh:
        for row in lines:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    valid_spreads = [s for s in spreads if s is not None]
    per_criteria = {
        name: {
            "mean_score": statistics.fmean([r.score for r in rs if r.score is not None]),
            "mean_confidence": statistics.fmean([r.confidence for r in rs if r.confidence is not None]),
            "invalid": sum(1 for r in rs if not r.valid),
        }
        for name, rs in table.items()
    }
    summary = {
        "n_jokes": len(jokes),
        "criteria": list(criteria),
        "mean_spread": statistics.fmean(valid_spreads) if valid_spreads else None,
        "median_spread": statistics.median(valid_spreads) if valid_spreads else None,
        "max_spread": max(valid_spreads) if valid_spreads else None,
        "per_criteria": per_criteria,
        "jev_calls": scorer.calls,
        "mean_latency_ms": statistics.fmean([r.latency_ms for rs in table.values() for r in rs]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"-> {out_dir}")
    return 0


def cmd_evaluate(args) -> int:
    from joke_pref.runs import evaluate, write_evaluation

    items, _ = _load_user_items(args)
    splits = _splits(items, args.seed)
    chosen = select(items, splits.ids(args.split))
    criteria = load_criteria(args.criteria)
    scorer = _scorer(args)
    rows, evals, summary = evaluate(scorer, chosen, criteria, w_rank=args.w_rank)
    name = args.name or f"eval-{Path(args.criteria).stem}-{args.split}-{_stamp()}"
    out = write_evaluation(
        Path(args.results) / args.user / name,
        rows=rows,
        summary=summary,
        criteria=criteria,
        meta={
            "user": args.user,
            "split": args.split,
            "seed": args.seed,
            "criteria_file": str(args.criteria),
            "repeats": args.repeats,
            "model": scorer.model,
            "jev_calls": scorer.calls,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    print(json.dumps(summary, indent=2))
    print(f"-> {out}")
    return 0


def cmd_optimize(args) -> int:
    from joke_pref.reflection import DeepSeekReflectionLM
    from joke_pref.runs import evaluate, optimize, write_evaluation

    items, _ = _load_user_items(args)
    splits = _splits(items, args.seed)
    train, val, test = (select(items, splits.ids(n)) for n in ("train", "val", "test"))
    if args.shuffle_labels:
        # Null check: break the punchline-label link in train and val only.
        train = shuffle_labels(train, seed=args.seed)
        val = shuffle_labels(val, seed=args.seed + 1)
        print("NULL RUN: labels shuffled within each train/val premise; test keeps real labels")
    if len(train) < 5 or len(val) < 3:
        sys.exit(f"too few labeled premises: {len(train)} train / {len(val)} val")
    seed_criteria = load_criteria(args.criteria)
    run_dir = Path(args.results) / args.user / (args.name or f"gepa-{_stamp()}")
    scorer = _scorer(args)
    reflection = DeepSeekReflectionLM(log_path=run_dir / "reflection.jsonl")
    print(f"{len(train)} train / {len(val)} val / {len(test)} test premises; budget {args.max_metric_calls} metric calls; run dir {run_dir}")
    best, _ = optimize(
        scorer,
        train=train,
        val=val,
        seed_criteria=seed_criteria,
        reflection_lm=reflection,
        max_metric_calls=args.max_metric_calls,
        run_dir=run_dir,
        seed=args.seed,
        reflection_minibatch_size=args.minibatch,
        w_rank=args.w_rank,
        joint=not args.per_level,
    )
    print("best criteria:")
    print(json.dumps(best.as_dict(), indent=2, ensure_ascii=False))
    for split_name, chosen, crit, tag in (
        ("test", test, seed_criteria, "seed"),
        ("test", test, best, "best"),
    ):
        rows, evals, summary = evaluate(scorer, chosen, crit, w_rank=args.w_rank)
        write_evaluation(
            run_dir / f"test-{tag}",
            rows=rows,
            summary=summary,
            criteria=crit,
            meta={"user": args.user, "split": split_name, "seed": args.seed, "criteria": tag, "model": scorer.model},
        )
        print(f"test {tag}: " + json.dumps({k: summary.get(k) for k in ("premise_score", "concordance", "agreement", "exact_accuracy", "spearman", "top1_hit_rate")}))
    print(f"-> {run_dir}")
    return 0


def cmd_jester_evaluate(args) -> int:
    """Score the 100 Jester jokes with one rubric and report per-user
    concordance against tercile labels, next to the crowd and kNN ceilings."""
    import numpy as np

    from joke_pref.criteria import save_criteria
    from joke_pref.jester import complete_users, load_jokes, load_ratings
    from joke_pref.jester_eval import evaluate_rubric, selected_users, user_mask, write_user_table

    jokes = load_jokes()
    ratings = complete_users(load_ratings())
    criteria = load_criteria(args.criteria)
    scorer = _scorer(args)
    t0 = datetime.now(timezone.utc)
    results = scorer.score_many([j.text for j in jokes], criteria)
    ev = evaluate_rubric(results, jokes, ratings, seed=args.seed, k=args.k)
    chosen = selected_users(args.user_stats, seed=args.seed)
    mask = user_mask(ratings, chosen)
    name = args.name or f"{Path(args.criteria).stem}-{_stamp()}"
    out = Path(args.results) / "jester" / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "jokes.jsonl", "w", encoding="utf-8") as fh:
        for j, r in zip(jokes, results):
            fh.write(json.dumps({"joke_id": j.id, **r.as_dict()}, ensure_ascii=False) + "\n")
    write_user_table(out / "users.csv", ev)
    save_criteria(criteria, out / "criteria.json")
    summary = {
        "criteria_file": str(args.criteria),
        "seed": args.seed,
        "k": args.k,
        "n_jokes": len(jokes),
        "test_joke_ids": ev.test_ids,
        "invalid_results": ev.n_invalid,
        "model": scorer.model,
        "jev_calls": scorer.calls,
        "mean_latency_ms": float(np.mean([r.latency_ms for r in results])),
        "created_at": t0.isoformat(timespec="seconds"),
        "all_users": ev.table(),
        "selected_users": ev.table(mask),
        "selected_user_ids": chosen,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for group in ("all_users", "selected_users"):
        t = summary[group]
        print(
            f"{group:<15s} n={t['n_users']:5d}  rubric {t['rubric_concordance_median']:.3f}  "
            f"crowd {t['crowd_concordance_median']:.3f}  kNN {t['knn_concordance_median']:.3f}  "
            f"agreement {t['rubric_agreement_mean']:.3f}  exact {t['rubric_exact_accuracy_mean']:.3f}  "
            f"beats crowd {t['rubric_beats_crowd_share']:.2f}"
        )
    print(f"-> {out}")
    return 0


# --- parser -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="joke-pref", description=__doc__)
    p.add_argument("--premises", default=DEFAULT_PREMISES)
    p.add_argument("--labels-dir", default=DEFAULT_LABELS_DIR)
    p.add_argument("--results", default=DEFAULT_RESULTS)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("label", help="start the labeling web page")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(fn=cmd_label)

    s = sub.add_parser("stats", help="label counts per user")
    s.add_argument("--user")
    s.set_defaults(fn=cmd_stats)

    def jev_args(sp):
        sp.add_argument("--timeout", type=float, default=30.0)
        sp.add_argument("--workers", type=int, default=8)
        sp.add_argument("--no-cache", action="store_true")

    def user_args(sp):
        sp.add_argument("--user", required=True)
        sp.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("split", help="show and save the premise split for a user")
    user_args(s)
    s.set_defaults(fn=cmd_split)

    s = sub.add_parser("probe", help="check that criteria text moves Jev at all")
    s.add_argument("--criteria-dir", default="criteria/probe")
    s.add_argument("--n", type=int, default=20)
    jev_args(s)
    s.set_defaults(fn=cmd_probe)

    s = sub.add_parser("evaluate", help="score one criteria file on a split")
    user_args(s)
    s.add_argument("--criteria", default=DEFAULT_SEED_CRITERIA)
    s.add_argument("--split", default="test", choices=("train", "val", "test", "all"))
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--w-rank", type=float, default=0.6)
    s.add_argument("--name")
    jev_args(s)
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("optimize", help="run GEPA over the three level descriptions")
    user_args(s)
    s.add_argument("--criteria", default=DEFAULT_SEED_CRITERIA)
    s.add_argument("--max-metric-calls", type=int, default=400)
    s.add_argument("--minibatch", type=int, default=12, help="premises per reflection step")
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--w-rank", type=float, default=0.6)
    s.add_argument("--per-level", action="store_true", help="GEPA's default proposer: one level per step (default: all three levels in one call)")
    s.add_argument("--shuffle-labels", action="store_true", help="null check: shuffle labels within each train/val premise")
    s.add_argument("--name")
    jev_args(s)
    s.set_defaults(fn=cmd_optimize)

    s = sub.add_parser("jester-evaluate", help="score the Jester jokes with one rubric, per-user concordance")
    s.add_argument("--criteria", default="criteria/probe/a-generic.json")
    s.add_argument("--seed", type=int, default=20260919, help="item split seed")
    s.add_argument("--k", type=int, default=80, help="kNN neighbours for the ceiling")
    s.add_argument("--user-stats", default="data/jester/user_stats.csv")
    s.add_argument("--name")
    jev_args(s)
    s.set_defaults(fn=cmd_jester_evaluate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
