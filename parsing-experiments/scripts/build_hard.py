#!/usr/bin/env python3
"""Turn candidate artifacts into schema-valid items with oracle labels.

Candidates are authored source fragments. They carry an `intended_yes` flag,
but that flag NEVER becomes the label. The oracle decides every label, which
is the suite's rule (see README: "Answers are oracle labels only").

`intended_yes` is used only to catch authoring mistakes. When the oracle
disagrees with the author, the candidate is dropped and reported:

- intended broken, but the oracle accepts it  -> the "fault" does not break
  parsing, so the item would be a mislabelled clean item.
- intended clean, but the oracle rejects it   -> the "clean" artifact is
  actually broken, so it has no fault tags to record.

Input  (JSONL): {"task", "artifact", "intended_yes", "fault_tags", "note"}
Output (JSONL): the item schema from `jev_eval.schema`, difficulty "hard".

    python scripts/build_hard.py --cand <dir> --out data/hard
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from jev_eval.oracles import OracleToolMissing, check
from jev_eval.schema import dumps_item, item_from_dict, load_jsonl
from jev_eval.tasks import TASK_IDS, get_task

ID_INFIX = "h2"


def _seed_artifacts(seed_dir: Path) -> set[str]:
    """Artifacts already in the seed set, so the extension adds new work."""
    seen: set[str] = set()
    if not seed_dir.is_dir():
        return seen
    for path in sorted(seed_dir.glob("*.jsonl")):
        for item in load_jsonl(path):
            seen.add(item.artifact)
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cand", required=True, help="directory of candidate jsonl files")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seed", default="data/seed", help="seed dir, for de-duplication")
    args = ap.parse_args()

    cand_dir = Path(args.cand)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    already = _seed_artifacts(Path(args.seed))

    grand = Counter()
    for task in TASK_IDS:
        path = cand_dir / f"{task}.jsonl"
        if not path.exists():
            print(f"{task}: no candidate file, skipped")
            continue

        spec = get_task(task)
        kept: list[str] = []
        stats = Counter()
        seen_here: set[str] = set()

        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  {task}:{lineno}: bad JSON, dropped ({exc})")
                stats["bad_json"] += 1
                continue

            artifact = cand.get("artifact")
            if not isinstance(artifact, str) or not artifact.strip():
                stats["empty"] += 1
                continue
            if artifact in already or artifact in seen_here:
                stats["duplicate"] += 1
                continue
            seen_here.add(artifact)

            try:
                res = check(task, artifact)
            except OracleToolMissing as exc:
                print(f"  {task}: oracle unavailable, stopping ({exc})")
                stats["oracle_missing"] += 1
                break

            intended = bool(cand.get("intended_yes"))
            if res.ok != intended:
                # The author and the oracle disagree. The oracle is right, but
                # the candidate is then not the item the author meant to write.
                stats["intent_mismatch_yes" if res.ok else "intent_mismatch_no"] += 1
                continue

            tags = cand.get("fault_tags") or []
            if res.ok and tags:
                stats["clean_with_tags"] += 1
                continue
            if not res.ok and not tags:
                stats["broken_without_tags"] += 1
                continue

            row = {
                "id": f"{task}.{ID_INFIX}.{len(kept) + 1:05d}",
                "task": task,
                "difficulty": "hard",
                "artifact": artifact,
                "question": spec.question,
                "answer": res.ok,
                "fault_tags": list(tags),
                "oracle_meta": res.meta(),
            }
            # Re-validate through the real schema so nothing invalid is written.
            kept.append(dumps_item(item_from_dict(row)))
            stats["kept_yes" if res.ok else "kept_no"] += 1

        (out_dir / f"{task}.jsonl").write_text("".join(s + "\n" for s in kept))
        grand.update(stats)
        detail = ", ".join(f"{k}={v}" for k, v in sorted(stats.items()) if not k.startswith("kept"))
        print(
            f"{task}: kept {stats['kept_yes'] + stats['kept_no']} "
            f"(yes={stats['kept_yes']} no={stats['kept_no']})"
            + (f"  dropped: {detail}" if detail else "")
        )

    total = grand["kept_yes"] + grand["kept_no"]
    print(f"\ntotal kept {total} (yes={grand['kept_yes']} no={grand['kept_no']}) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
