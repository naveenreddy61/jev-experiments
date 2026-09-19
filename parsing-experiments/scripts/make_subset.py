#!/usr/bin/env python3
"""Build a label-balanced subset of the seed set for a bounded run.

`run --limit N` takes the first N items of the joined list. The seed files are
not shuffled, so the head of each file is strongly yes-skewed and a limited run
measures almost nothing. This script samples a fixed number of yes and no items
per task instead.

    python scripts/make_subset.py --per-label 5 --out /tmp/subset
    python -m jev_eval run --provider jev --data /tmp/subset --out results/jev

The seed is fixed, so the same subset comes back on every run.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from jev_eval.tasks import TASK_IDS

DEFAULT_SEED = 20260918


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/seed", help="source seed directory")
    ap.add_argument("--per-label", type=int, default=5, help="yes items and no items per task")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    src = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    total = 0
    # Sorted, not registry order: the sample must not change if TASKS is
    # reordered, because the RNG stream is shared across tasks.
    for task in sorted(TASK_IDS):
        path = src / f"{task}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        yes = [r for r in rows if r["answer"] is True]
        no = [r for r in rows if r["answer"] is False]
        if len(yes) < args.per_label or len(no) < args.per_label:
            raise SystemExit(
                f"{task}: need {args.per_label} of each label, "
                f"have {len(yes)} yes and {len(no)} no"
            )
        pick = rng.sample(yes, args.per_label) + rng.sample(no, args.per_label)
        rng.shuffle(pick)
        (out / f"{task}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in pick)
        )
        total += len(pick)
        print(f"{task}: {args.per_label} yes + {args.per_label} no")

    print(f"wrote {total} items to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
