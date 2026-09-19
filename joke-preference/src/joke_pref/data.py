"""Premises, punchlines, personal labels, and splits.

The dataset file `data/premises.jsonl` holds one premise per line with its
candidate punchlines. It is shared and committed.

A label file `data/labels/<user>.jsonl` holds one row per click in the
labeling tool. It is personal and gitignored. The latest row for a punchline
wins, so a person can change their mind. A `skip` row removes a label.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

LABELS: tuple[str, ...] = ("bad", "good", "great")
LABEL_INDEX: dict[str, int] = {name: i for i, name in enumerate(LABELS)}
SKIP = "skip"

_USER_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


@dataclass(frozen=True)
class Punchline:
    id: str
    premise_id: str
    index: int
    text: str
    style: str


@dataclass(frozen=True)
class Premise:
    id: str
    premise: str
    format: str
    domain: str
    punchlines: tuple[Punchline, ...]

    def punchline(self, punchline_id: str) -> Punchline:
        for p in self.punchlines:
            if p.id == punchline_id:
                return p
        raise KeyError(punchline_id)


@dataclass
class LabeledPremise:
    """A premise with the labels one person gave to its punchlines.

    Only labeled punchlines are kept. This is the unit GEPA sees.
    """

    premise: Premise
    labels: dict[str, str] = field(default_factory=dict)  # punchline_id -> label

    @property
    def id(self) -> str:
        return self.premise.id

    def items(self) -> list[tuple[Punchline, int]]:
        """Labeled punchlines with the label index, in dataset order."""
        return [
            (p, LABEL_INDEX[self.labels[p.id]])
            for p in self.premise.punchlines
            if p.id in self.labels
        ]


def joke_text(premise: Premise, punchline: Punchline) -> str:
    """The full joke as Jev sees it: setup, blank line, punchline.

    A premise with an empty setup (a Jester group) gives the punchline alone.
    """
    setup = premise.premise.strip()
    if not setup:
        return punchline.text.strip()
    return f"{setup}\n\n{punchline.text.strip()}"


def punchline_id(premise_id: str, index: int) -> str:
    return f"{premise_id}.{index}"


# --- premises ---------------------------------------------------------------


def parse_premise(obj: dict) -> Premise:
    pid = str(obj["id"])
    punchlines = tuple(
        Punchline(
            id=punchline_id(pid, i),
            premise_id=pid,
            index=i,
            text=str(p["text"]).strip(),
            style=str(p.get("style", "")),
        )
        for i, p in enumerate(obj["punchlines"])
    )
    if not 2 <= len(punchlines) <= 5:
        raise ValueError(f"{pid}: expected 2-5 punchlines, got {len(punchlines)}")
    return Premise(
        id=pid,
        premise=str(obj["premise"]).strip(),
        format=str(obj.get("format", "")),
        domain=str(obj.get("domain", "")),
        punchlines=punchlines,
    )


def load_premises(path: str | Path) -> list[Premise]:
    out: list[Premise] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                premise = parse_premise(json.loads(line))
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if premise.id in seen:
                raise ValueError(f"{path}:{line_no}: duplicate id {premise.id}")
            seen.add(premise.id)
            out.append(premise)
    return out


# --- labels -----------------------------------------------------------------


def validate_user(user: str) -> str:
    user = user.strip().lower()
    if not _USER_RE.match(user):
        raise ValueError(
            "user name must be 1-32 characters of a-z, 0-9, '-' or '_', "
            "and start with a letter or digit"
        )
    return user


def label_path(labels_dir: str | Path, user: str) -> Path:
    return Path(labels_dir) / f"{validate_user(user)}.jsonl"


def append_label(labels_dir: str | Path, user: str, pl_id: str, label: str) -> dict:
    if label not in LABELS and label != SKIP:
        raise ValueError(f"unknown label {label!r}")
    path = label_path(labels_dir, user)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "user": validate_user(user),
        "punchline_id": pl_id,
        "label": label,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def iter_label_rows(labels_dir: str | Path, user: str) -> Iterator[dict]:
    path = label_path(labels_dir, user)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_labels(labels_dir: str | Path, user: str) -> dict[str, str]:
    """Latest label per punchline. A `skip` row removes the label."""
    latest: dict[str, str] = {}
    for row in iter_label_rows(labels_dir, user):
        pl_id = row["punchline_id"]
        if row["label"] == SKIP:
            latest.pop(pl_id, None)
        else:
            latest[pl_id] = row["label"]
    return latest


def list_users(labels_dir: str | Path) -> list[str]:
    d = Path(labels_dir)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.jsonl"))


def labeled_premises(
    premises: Iterable[Premise], labels: dict[str, str], *, min_labeled: int = 1
) -> list[LabeledPremise]:
    out: list[LabeledPremise] = []
    for premise in premises:
        mine = {p.id: labels[p.id] for p in premise.punchlines if p.id in labels}
        if len(mine) >= min_labeled:
            out.append(LabeledPremise(premise=premise, labels=mine))
    return out


# --- splits -----------------------------------------------------------------


@dataclass(frozen=True)
class Splits:
    seed: int
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
        }

    def ids(self, name: str) -> tuple[str, ...]:
        if name == "all":
            return self.train + self.val + self.test
        return getattr(self, name)


def make_splits(
    premise_ids: Iterable[str],
    *,
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.58, 0.17, 0.25),
) -> Splits:
    """Split by premise so a test punchline never shares a premise with train.

    Sorted then shuffled with a fixed seed, so the same labeled set and seed
    give the same split on any machine.
    """
    ids = sorted(set(premise_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = round(n * fractions[0])
    n_val = round(n * fractions[1])
    train = tuple(ids[:n_train])
    val = tuple(ids[n_train : n_train + n_val])
    test = tuple(ids[n_train + n_val :])
    return Splits(seed=seed, train=train, val=val, test=test)


def select(items: list[LabeledPremise], ids: Iterable[str]) -> list[LabeledPremise]:
    wanted = set(ids)
    return [it for it in items if it.id in wanted]


def shuffle_labels(items: list[LabeledPremise], seed: int = 0) -> list[LabeledPremise]:
    """Permute the labels inside each premise. The label counts per premise
    stay the same, but the link between punchline and label is broken. An
    optimizer trained on this must show no gain on real labels; if it does,
    the pipeline leaks."""
    rng = random.Random(seed)
    out: list[LabeledPremise] = []
    for item in items:
        ids = list(item.labels)
        values = [item.labels[i] for i in ids]
        rng.shuffle(values)
        out.append(LabeledPremise(item.premise, dict(zip(ids, values))))
    return out
