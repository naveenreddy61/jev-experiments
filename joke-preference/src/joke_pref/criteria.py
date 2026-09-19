"""Criteria files and the GEPA candidate <-> Jev criteria mapping.

A criteria file is JSON:

    {
      "instructions": "How much will this reader enjoy this joke?",
      "levels": ["...bad...", "...good...", "...great..."]
    }

Each level is a string, or an object such as {"what": "...", "examples": [...]}.
The three levels are the only text GEPA changes. `instructions` stays fixed.

GEPA works on a `dict[str, str]` candidate. The three components are
`level_bad`, `level_good`, `level_great`. A component text that parses as a
JSON object is sent to Jev as an object; anything else is sent as a string.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from joke_pref.data import LABELS

Level = str | dict[str, Any]

COMPONENTS: tuple[str, ...] = tuple(f"level_{name}" for name in LABELS)

DEFAULT_INSTRUCTIONS = "How much will this reader enjoy this joke?"


@dataclass(frozen=True)
class Criteria:
    instructions: str
    levels: tuple[Level, ...]

    def __post_init__(self) -> None:
        if len(self.levels) != len(LABELS):
            raise ValueError(f"criteria needs {len(LABELS)} levels, got {len(self.levels)}")
        for lv in self.levels:
            if isinstance(lv, str):
                if not lv.strip():
                    raise ValueError("a level description is empty")
            elif not isinstance(lv, dict) or not lv:
                raise ValueError("a level must be a non-empty string or object")

    def as_dict(self) -> dict:
        return {"instructions": self.instructions, "levels": list(self.levels)}

    def to_candidate(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, lv in zip(COMPONENTS, self.levels):
            out[name] = lv if isinstance(lv, str) else json.dumps(lv, ensure_ascii=False, indent=2)
        return out

    @classmethod
    def from_candidate(cls, candidate: dict[str, str], instructions: str) -> "Criteria":
        return cls(
            instructions=instructions,
            levels=tuple(parse_level(candidate[name]) for name in COMPONENTS),
        )

    def key(self) -> str:
        """A stable string for caching."""
        return json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False)


def parse_level(text: str) -> Level:
    """A component text is an object when it parses as a JSON object."""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        if isinstance(obj, dict) and obj:
            return obj
    return stripped


def load_criteria(path: str | Path) -> Criteria:
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    return Criteria(
        instructions=str(obj.get("instructions") or DEFAULT_INSTRUCTIONS),
        levels=tuple(obj["levels"]),
    )


def save_criteria(criteria: Criteria, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(criteria.as_dict(), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
