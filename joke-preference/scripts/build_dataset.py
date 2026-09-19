"""Merge data/raw/part-*.jsonl into data/premises.jsonl.

Validates every record, rejects duplicate ids and duplicate premise text,
and interleaves the parts so consecutive premises come from different
domains. The output order is the labeling order.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from joke_pref.data import parse_premise  # noqa: E402

STYLES = {
    "pun", "wordplay", "absurdist", "dark", "anti-joke", "meta", "observational",
    "deadpan", "misdirection", "hyperbole", "self-deprecating", "nerdy", "dad-joke", "callback",
}
FORMATS = {"question", "difference", "narrative", "one-liner", "dialogue", "observational"}


def main() -> int:
    raw_dir = ROOT / "data" / "raw"
    parts = sorted(raw_dir.glob("part-*.jsonl"))
    if not parts:
        sys.exit(f"no part files in {raw_dir}")
    per_part: list[list[dict]] = []
    problems: list[str] = []
    for path in parts:
        rows = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                parse_premise(obj)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{path.name}:{n}: {exc}")
                continue
            if obj.get("format") not in FORMATS:
                problems.append(f"{path.name}:{n}: bad format {obj.get('format')!r}")
            styles = [p.get("style") for p in obj["punchlines"]]
            bad = [s for s in styles if s not in STYLES]
            if bad:
                problems.append(f"{path.name}:{n}: unknown styles {bad}")
            if len(set(styles)) != len(styles):
                problems.append(f"{path.name}:{n}: repeated style inside a premise")
            rows.append(obj)
        per_part.append(rows)

    merged: list[dict] = []
    longest = max(len(r) for r in per_part)
    for i in range(longest):
        for rows in per_part:
            if i < len(rows):
                merged.append(rows[i])

    ids = Counter(r["id"] for r in merged)
    for k, c in ids.items():
        if c > 1:
            problems.append(f"duplicate id {k}")
    texts = Counter(r["premise"].strip().lower() for r in merged)
    for t, c in texts.items():
        if c > 1:
            problems.append(f"duplicate premise text: {t[:60]}")
    all_punch = Counter(p["text"].strip().lower() for r in merged for p in r["punchlines"])
    for t, c in all_punch.items():
        if c > 1:
            problems.append(f"duplicate punchline text: {t[:60]}")

    if problems:
        print("\n".join(problems))
        return 1

    out = ROOT / "data" / "premises.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for r in merged:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_punch = sum(len(r["punchlines"]) for r in merged)
    print(f"{len(merged)} premises, {n_punch} punchlines -> {out}")
    print("formats:", dict(Counter(r["format"] for r in merged)))
    print("domains:", dict(Counter(r["domain"] for r in merged)))
    print("styles:", dict(Counter(p["style"] for r in merged for p in r["punchlines"])))
    print("punchlines per premise:", dict(sorted(Counter(len(r["punchlines"]) for r in merged).items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
