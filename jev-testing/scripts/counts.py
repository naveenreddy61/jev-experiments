#!/usr/bin/env python3
"""Raw counts only: correct / wrong per cell, for two models side by side.

Cells are task x difficulty x the oracle's answer. No calibration, no derived
scores. Every number here is a count of items.

    python scripts/counts.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# One run per model per set. Named so the table is reproducible.
JEV = ["results/jev-full-200", "results/jev-hard-new"]
DS = ["results/ds-tok-seed", "results/ds-tok-hard"]
DATA = ["data/seed", "data/hard"]

TASKS = ["bash.parse", "c.compile", "latex.parse", "python.parse"]
DIFFS = ["easy", "medium", "hard (v1)", "hard (v2)"]


def load_items(dirs):
    out = {}
    for d in dirs:
        p = Path(d)
        for f in ([p] if p.is_file() else sorted(p.glob("*.jsonl"))):
            for line in f.read_text().splitlines():
                if line.strip():
                    o = json.loads(line)
                    # v2 hard items are far longer than v1 hard; keep them apart.
                    o["_diff"] = ("hard (v2)" if ".h2." in o["id"]
                                  else ("hard (v1)" if o["difficulty"] == "hard"
                                        else o["difficulty"]))
                    out[o["id"]] = o
    return out


def load_preds(dirs):
    out = {}
    for d in dirs:
        p = Path(d)
        f = p if p.is_file() else p / "predictions.jsonl"
        for line in f.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                out[o["id"]] = o
    return out


def tally(ids, items, preds, key):
    """key(item) -> cell. Returns cell -> [n, correct, wrong, said_yes, invalid]."""
    t = defaultdict(lambda: [0, 0, 0, 0, 0])
    for i in ids:
        c = t[key(items[i])]
        a = preds[i].get("answer")
        c[0] += 1
        if a is None:
            c[4] += 1
        else:
            c[1 if a == items[i]["answer"] else 2] += 1
            if a:
                c[3] += 1
    return t


def block(title, ids, items, jev, ds, key, order):
    print(f"\n{title}")
    print(f"  {'cell':28s} {'items':>5s} {'yes':>4s} {'no':>4s} "
          f"| {'JEV ok':>6s} {'wrong':>5s} {'said yes':>8s} "
          f"| {'DS ok':>6s} {'wrong':>5s} {'said yes':>8s}")
    tj = tally(ids, items, jev, key)
    td = tally(ids, items, ds, key)
    tg = tally(ids, items, jev, key)  # for gold split
    gold_yes = defaultdict(int)
    for i in ids:
        if items[i]["answer"]:
            gold_yes[key(items[i])] += 1
    keys = [k for k in order if k in tj] + [k for k in tj if k not in order]
    for k in keys:
        n = tj[k][0]
        gy = gold_yes[k]
        print(f"  {str(k):28s} {n:5d} {gy:4d} {n-gy:4d} "
              f"| {tj[k][1]:6d} {tj[k][2]:5d} {tj[k][3]:8d} "
              f"| {td[k][1]:6d} {td[k][2]:5d} {td[k][3]:8d}")
    n = sum(v[0] for v in tj.values())
    gy = sum(gold_yes.values())
    print(f"  {'TOTAL':28s} {n:5d} {gy:4d} {n-gy:4d} "
          f"| {sum(v[1] for v in tj.values()):6d} {sum(v[2] for v in tj.values()):5d} "
          f"{sum(v[3] for v in tj.values()):8d} "
          f"| {sum(v[1] for v in td.values()):6d} {sum(v[2] for v in td.values()):5d} "
          f"{sum(v[3] for v in td.values()):8d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jev", nargs="+", default=JEV)
    ap.add_argument("--ds", nargs="+", default=DS)
    ap.add_argument("--data", nargs="+", default=DATA)
    a = ap.parse_args()

    items = load_items(a.data)
    jev, ds = load_preds(a.jev), load_preds(a.ds)
    ids = sorted(set(items) & set(jev) & set(ds))
    print(f"items scored by both models: {len(ids)}  (of {len(items)} in the data)")
    print(f"  JEV runs: {', '.join(a.jev)}")
    print(f"  DS  runs: {', '.join(a.ds)}")
    print("\n'said yes' = the model answered yes (valid / parses).")
    print("Compare it to the 'yes' column, which is how many are really yes.")

    block("== by task x difficulty x gold answer ==", ids, items, jev, ds,
          lambda o: (o["task"], o["_diff"], "yes" if o["answer"] else "no"),
          [(t, d, g) for t in TASKS for d in DIFFS for g in ("yes", "no")])

    block("== by difficulty x gold answer ==", ids, items, jev, ds,
          lambda o: (o["_diff"], "yes" if o["answer"] else "no"),
          [(d, g) for d in DIFFS for g in ("yes", "no")])

    block("== by task x gold answer ==", ids, items, jev, ds,
          lambda o: (o["task"], "yes" if o["answer"] else "no"),
          [(t, g) for t in TASKS for g in ("yes", "no")])

    block("== by difficulty ==", ids, items, jev, ds,
          lambda o: o["_diff"], DIFFS)

    block("== by task ==", ids, items, jev, ds, lambda o: o["task"], TASKS)

    block("== by gold answer ==", ids, items, jev, ds,
          lambda o: "yes" if o["answer"] else "no", ["yes", "no"])


if __name__ == "__main__":
    main()
