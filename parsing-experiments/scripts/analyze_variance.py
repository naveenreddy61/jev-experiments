#!/usr/bin/env python3
"""Compare repeat runs of one model on one item set.

The suite ran one call per item until now, so it had no variance estimate.
This reads two or more prediction directories for the SAME items and reports
how much the answer, the confidence and the reasoning cost move between runs.

    python scripts/analyze_variance.py \
        --pred results/ds-def-r1 results/ds-def-r2 results/ds-def-r3 \
        --data data/hard
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path


def load_items(dirs: list[str]) -> dict[str, dict]:
    items: dict[str, dict] = {}
    for d in dirs:
        p = Path(d)
        files = [p] if p.is_file() else sorted(p.glob("*.jsonl"))
        for f in files:
            for line in f.read_text().splitlines():
                if line.strip():
                    o = json.loads(line)
                    items[o["id"]] = o
    return items


def load_pred(d: str) -> dict[str, dict]:
    p = Path(d)
    f = p if p.is_file() else p / "predictions.jsonl"
    rows = {}
    for line in f.read_text().splitlines():
        if line.strip():
            o = json.loads(line)
            rows[o["id"]] = o
    return rows


def mean(xs):
    return st.mean(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--show-flips", type=int, default=25)
    args = ap.parse_args()

    items = load_items(args.data)
    runs = [load_pred(d) for d in args.pred]
    names = [Path(d).name for d in args.pred]
    k = len(runs)

    ids = sorted(set(runs[0]) & set(items))
    for r in runs[1:]:
        ids = [i for i in ids if i in r]
    print(f"runs={k}  items common to all runs={len(ids)}\n")

    # ---- per-run accuracy -------------------------------------------------
    print("== accuracy per run ==")
    accs = []
    for name, r in zip(names, runs):
        ok = [1.0 if r[i].get("answer") == items[i]["answer"] else 0.0 for i in ids]
        inval = sum(1 for i in ids if r[i].get("answer") is None)
        accs.append(mean(ok))
        print(f"  {name:16s} acc={mean(ok):.4f}  invalid={inval}")
    if k > 1:
        print(f"  spread: min={min(accs):.4f} max={max(accs):.4f} "
              f"range={max(accs)-min(accs):.4f} sd={st.pstdev(accs):.4f}")

    # ---- agreement between runs ------------------------------------------
    print("\n== per-item answer agreement ==")
    unstable = []
    for i in ids:
        ans = [r[i].get("answer") for r in runs]
        if len(set(ans)) > 1:
            unstable.append(i)
    print(f"  unanimous : {len(ids)-len(unstable)} / {len(ids)} "
          f"({100*(len(ids)-len(unstable))/len(ids):.1f}%)")
    print(f"  disagreed : {len(unstable)}")

    # Majority vote needs an odd k; with an even k a tie has no winner and the
    # number would only report how Counter breaks the tie.
    if k % 2 == 1:
        maj_ok = 0
        for i in ids:
            c = Counter(r[i].get("answer") for r in runs
                        if r[i].get("answer") is not None)
            if c and c.most_common(1)[0][0] == items[i]["answer"]:
                maj_ok += 1
        print(f"  majority-of-{k} accuracy = {maj_ok/len(ids):.4f} "
              f"(best single {max(accs):.4f}, mean single {mean(accs):.4f})")
    else:
        print(f"  majority vote skipped: {k} runs is even, so ties have no winner")

    # ---- what the unstable items look like -------------------------------
    if unstable:
        print("\n== items that changed answer between runs ==")
        tags = Counter()
        for i in unstable:
            tags.update(items[i].get("fault_tags") or ["(valid source)"])
        print("  fault tags:")
        for t, n in tags.most_common(12):
            print(f"    {t:32s} {n}")

        st_rt, un_rt = [], []
        for i in ids:
            rts = [r[i].get("reasoning_tokens") for r in runs]
            rts = [x for x in rts if x is not None]
            if not rts:
                continue
            (un_rt if i in unstable else st_rt).append(mean(rts))
        if st_rt and un_rt:
            print(f"\n  mean reasoning tokens, stable items   = {mean(st_rt):8.0f} "
                  f"(median {st.median(st_rt):.0f}, n={len(st_rt)})")
            print(f"  mean reasoning tokens, unstable items = {mean(un_rt):8.0f} "
                  f"(median {st.median(un_rt):.0f}, n={len(un_rt)})")

        print(f"\n  first {min(args.show_flips,len(unstable))} of {len(unstable)}:")
        for i in unstable[: args.show_flips]:
            ans = "".join("Y" if r[i].get("answer") else
                          ("-" if r[i].get("answer") is None else "n") for r in runs)
            ps = ",".join(f"{r[i].get('p_yes'):.2f}"
                          if r[i].get("p_yes") is not None else "--" for r in runs)
            rts = ",".join(str(r[i].get("reasoning_tokens") or "-") for r in runs)
            gold = "valid" if items[i]["answer"] else "broken"
            tg = ",".join(items[i].get("fault_tags") or [])
            print(f"    {i:24s} gold={gold:6s} ans={ans} p_yes=[{ps}] rt=[{rts}] {tg}")

    # ---- reasoning-token stability ---------------------------------------
    print("\n== reasoning-token stability, per item ==")
    cvs, means_ = [], []
    for i in ids:
        rts = [r[i].get("reasoning_tokens") for r in runs]
        rts = [x for x in rts if x is not None]
        if len(rts) < 2 or mean(rts) == 0:
            continue
        cvs.append(st.pstdev(rts) / mean(rts))
        means_.append(mean(rts))
    if cvs:
        print(f"  n={len(cvs)}  median CV={st.median(cvs):.3f}  mean CV={mean(cvs):.3f}")
        print(f"  CV quartiles: " + ", ".join(
            f"{q:.3f}" for q in st.quantiles(cvs, n=4)))
        tot = [sum(r[i].get("reasoning_tokens") or 0 for i in ids) for r in runs]
        print("  total reasoning tokens per run: " +
              ", ".join(f"{t:,}" for t in tot))
        if len(set(tot)) > 1:
            print(f"    spread = {100*(max(tot)-min(tot))/mean(tot):.1f}% of the mean")

    # ---- does the group ranking survive? ---------------------------------
    print("\n== median reasoning tokens by gold label, per run ==")
    for name, r in zip(names, runs):
        for lab, g in (("valid ", True), ("broken", False)):
            v = [r[i].get("reasoning_tokens") for i in ids
                 if items[i]["answer"] is g and r[i].get("reasoning_tokens")]
            if v:
                print(f"  {name:16s} {lab} median={st.median(v):7.0f} "
                      f"mean={mean(v):8.0f} max={max(v):6d}")


if __name__ == "__main__":
    main()
