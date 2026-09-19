#!/usr/bin/env python3
"""Relate a reasoning model's token spend to item difficulty and correctness.

Reads prediction files that carry `reasoning_tokens` (see `_telemetry` in
`jev_eval.cli`) and joins them to the source items for difficulty, fault tags
and artifact size.

    python scripts/analyze_reasoning.py \
        --pred results/ds-tok-seed results/ds-tok-hard \
        --data data/seed data/hard
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _load_items(dirs: list[str]) -> dict[str, dict]:
    items: dict[str, dict] = {}
    for d in dirs:
        for path in sorted(Path(d).glob("*.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    items[row["id"]] = row
    return items


def _load_preds(targets: list[str]) -> list[dict]:
    rows: list[dict] = []
    for t in targets:
        p = Path(t)
        if p.is_dir():
            p = p / "predictions.jsonl"
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return statistics.correlation(xs, ys)


def _summary(vals: list[float]) -> str:
    if not vals:
        return "n=0"
    return (
        f"n={len(vals):<4} mean={statistics.mean(vals):>7.1f} "
        f"median={statistics.median(vals):>7.1f} "
        f"p90={sorted(vals)[int(len(vals) * 0.9) - 1]:>7.1f} max={max(vals):>6.0f}"
    )


def _table(title: str, groups: dict[str, list[float]], order: list[str] | None = None) -> None:
    print(f"\n{title}")
    keys = order or sorted(groups, key=lambda k: -statistics.mean(groups[k]) if groups[k] else 0)
    for k in keys:
        if groups.get(k):
            print(f"  {k:<26}{_summary(groups[k])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    args = ap.parse_args()

    items = _load_items(args.data)
    rows = [r for r in _load_preds(args.pred) if r.get("reasoning_tokens") is not None]
    if not rows:
        raise SystemExit("no rows carry reasoning_tokens; re-run with the current runner")

    for r in rows:
        it = items[r["id"]]
        r["_difficulty"] = it["difficulty"]
        r["_tags"] = it["fault_tags"]
        r["_lines"] = len(it["artifact"].splitlines())
        r["_chars"] = len(it["artifact"])
        r["_correct"] = r["answer"] == r["gold"]

    tok = [float(r["reasoning_tokens"]) for r in rows]
    print(f"rows with telemetry: {len(rows)}")
    print(f"reasoning tokens overall: {_summary(tok)}")
    print(f"total reasoning tokens spent: {sum(tok):,.0f}")

    _table(
        "reasoning tokens by difficulty",
        {d: [float(r["reasoning_tokens"]) for r in rows if r["_difficulty"] == d]
         for d in ("easy", "medium", "hard")},
        order=["easy", "medium", "hard"],
    )
    _table(
        "reasoning tokens by task",
        {t: [float(r["reasoning_tokens"]) for r in rows if r["task"] == t]
         for t in sorted({r["task"] for r in rows})},
    )
    _table(
        "reasoning tokens by outcome",
        {
            "correct": [float(r["reasoning_tokens"]) for r in rows if r["_correct"]],
            "wrong": [float(r["reasoning_tokens"]) for r in rows if not r["_correct"]],
        },
        order=["correct", "wrong"],
    )
    _table(
        "reasoning tokens by gold label",
        {
            "gold yes (valid)": [float(r["reasoning_tokens"]) for r in rows if r["gold"]],
            "gold no (broken)": [float(r["reasoning_tokens"]) for r in rows if not r["gold"]],
        },
        order=["gold yes (valid)", "gold no (broken)"],
    )

    print("\ncorrelations with reasoning tokens (Pearson r)")
    pairs = [
        ("artifact lines", [float(r["_lines"]) for r in rows]),
        ("artifact chars", [float(r["_chars"]) for r in rows]),
        ("prompt tokens", [float(r["prompt_tokens"]) for r in rows]),
        ("latency ms", [float(r["latency_ms"]) for r in rows]),
        ("difficulty rank", [float({"easy": 0, "medium": 1, "hard": 2}[r["_difficulty"]]) for r in rows]),
        ("is broken (gold no)", [0.0 if r["gold"] else 1.0 for r in rows]),
        ("is wrong", [0.0 if r["_correct"] else 1.0 for r in rows]),
        ("confidence", [float(r["confidence"]) for r in rows]),
    ]
    for name, ys in pairs:
        r = _corr(tok, ys)
        print(f"  {name:<22}{'n/a' if r is None else f'{r:+.3f}'}")

    # Does spending more tokens predict getting it wrong?
    print("\nerror rate by reasoning-token quartile")
    ranked = sorted(rows, key=lambda r: r["reasoning_tokens"])
    q = max(1, len(ranked) // 4)
    for i in range(4):
        chunk = ranked[i * q : (i + 1) * q] if i < 3 else ranked[3 * q :]
        if not chunk:
            continue
        toks = [r["reasoning_tokens"] for r in chunk]
        err = sum(1 for r in chunk if not r["_correct"]) / len(chunk)
        print(
            f"  Q{i + 1} tokens {min(toks):>5}-{max(toks):<5} "
            f"n={len(chunk):<4} error rate={err:.3f}"
        )

    # Which fault types cost the most thought?
    by_tag: dict[str, list[float]] = defaultdict(list)
    miss = Counter()
    seen = Counter()
    for r in rows:
        for tag in r["_tags"]:
            by_tag[tag].append(float(r["reasoning_tokens"]))
            seen[tag] += 1
            if not r["_correct"]:
                miss[tag] += 1
    print("\ntop 12 fault types by mean reasoning tokens (n >= 2)")
    ranked_tags = sorted(
        (t for t in by_tag if len(by_tag[t]) >= 2),
        key=lambda t: -statistics.mean(by_tag[t]),
    )[:12]
    print(f"  {'tag':<28}{'n':>3}{'mean tok':>10}{'missed':>8}")
    for t in ranked_tags:
        print(f"  {t:<28}{seen[t]:>3}{statistics.mean(by_tag[t]):>10.0f}{miss[t]:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
