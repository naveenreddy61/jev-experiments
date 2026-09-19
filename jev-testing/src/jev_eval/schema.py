"""JSONL item schema and model-output contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from jev_eval.tasks import TASKS, get_task

DIFFICULTIES = frozenset({"easy", "medium", "hard"})
ITEM_KEYS = (
    "id",
    "task",
    "difficulty",
    "artifact",
    "question",
    "answer",
    "fault_tags",
    "oracle_meta",
)


@dataclass(frozen=True)
class Item:
    id: str
    task: str
    difficulty: str
    artifact: str
    question: str
    answer: bool
    fault_tags: tuple[str, ...]
    oracle_meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "difficulty": self.difficulty,
            "artifact": self.artifact,
            "question": self.question,
            "answer": self.answer,
            "fault_tags": list(self.fault_tags),
            "oracle_meta": dict(self.oracle_meta),
        }


@dataclass(frozen=True)
class ModelOutput:
    answer: bool
    p_yes: float

    def to_dict(self) -> dict[str, Any]:
        return {"answer": self.answer, "p_yes": self.p_yes}


@dataclass(frozen=True)
class InvalidOutput:
    reason: str
    raw: str


class SchemaError(ValueError):
    pass


def item_from_dict(raw: Any) -> Item:
    if not isinstance(raw, dict):
        raise SchemaError("item must be a JSON object")
    missing = [k for k in ITEM_KEYS if k not in raw]
    if missing:
        raise SchemaError(f"missing fields: {missing}")

    task_id = raw["task"]
    if not isinstance(task_id, str) or task_id not in TASKS:
        raise SchemaError(f"unknown task: {task_id!r}")
    spec = get_task(task_id)

    item_id = raw["id"]
    if not isinstance(item_id, str) or not item_id.startswith(f"{task_id}."):
        raise SchemaError(f"id {item_id!r} must start with {task_id}.")

    difficulty = raw["difficulty"]
    if difficulty not in DIFFICULTIES:
        raise SchemaError(f"difficulty must be easy|medium|hard, got {difficulty!r}")

    artifact = raw["artifact"]
    if not isinstance(artifact, str) or not artifact:
        raise SchemaError("artifact must be a non-empty string")

    question = raw["question"]
    if question != spec.question:
        raise SchemaError(f"question must be the exact locked rubric: {spec.question!r}")

    answer = raw["answer"]
    if type(answer) is not bool:
        raise SchemaError("answer must be a JSON boolean from the oracle")

    tags = raw["fault_tags"]
    if not isinstance(tags, list) or any(not isinstance(t, str) or not t for t in tags):
        raise SchemaError("fault_tags must be a list of non-empty strings")
    if answer and tags:
        raise SchemaError("fault_tags must be empty when answer is true")
    if (not answer) and not tags:
        raise SchemaError("fault_tags must name mutation ids when answer is false")

    meta = raw["oracle_meta"]
    if not isinstance(meta, dict):
        raise SchemaError("oracle_meta must be an object")
    for key in ("tool", "version_note", "exit_code"):
        if key not in meta:
            raise SchemaError(f"oracle_meta missing {key!r}")
    if not isinstance(meta["tool"], str) or not meta["tool"]:
        raise SchemaError("oracle_meta.tool must be a non-empty string")
    if not isinstance(meta["version_note"], str):
        raise SchemaError("oracle_meta.version_note must be a string")
    if type(meta["exit_code"]) is not int:
        raise SchemaError("oracle_meta.exit_code must be an int")

    return Item(
        id=item_id,
        task=task_id,
        difficulty=difficulty,
        artifact=artifact,
        question=question,
        answer=answer,
        fault_tags=tuple(tags),
        oracle_meta=dict(meta),
    )


def dumps_item(item: Item) -> str:
    return json.dumps(item.to_dict(), ensure_ascii=False, separators=(",", ":"))


def load_jsonl(path) -> list[Item]:
    items: list[Item] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                items.append(item_from_dict(json.loads(text)))
            except (json.JSONDecodeError, SchemaError) as exc:
                raise SchemaError(f"{path}:{lineno}: {exc}") from exc
    return items


def load_items(paths: Iterable) -> list[Item]:
    items: list[Item] = []
    seen: set[str] = set()
    for path in paths:
        for item in load_jsonl(path):
            if item.id in seen:
                raise SchemaError(f"duplicate id: {item.id}")
            seen.add(item.id)
            items.append(item)
    return items


def _extract_fenced_json(text: str) -> str | None:
    marker = "```"
    start = text.find(marker)
    if start < 0:
        return None
    after = text[start + len(marker) :]
    if after.startswith("json"):
        after = after[4:]
    after = after.lstrip("\r\n")
    end = after.find(marker)
    if end < 0:
        return None
    return after[:end].strip()


def _extract_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_json_candidate(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_model_output(text: str | None) -> ModelOutput | InvalidOutput:
    """Enforce the {answer: bool, p_yes: float} contract.

    Missing or garbage output is invalid. Types are not imputed.
    """
    raw = "" if text is None else str(text)
    stripped = raw.strip()
    if not stripped:
        return InvalidOutput(reason="empty output", raw=raw)

    obj = _parse_json_candidate(stripped)
    if obj is None:
        fenced = _extract_fenced_json(stripped)
        if fenced is not None:
            obj = _parse_json_candidate(fenced)
    if obj is None:
        balanced = _extract_balanced_object(stripped)
        if balanced is not None:
            obj = _parse_json_candidate(balanced)
    if obj is None:
        return InvalidOutput(reason="output is not JSON", raw=raw)
    if not isinstance(obj, dict):
        return InvalidOutput(reason="JSON root must be an object", raw=raw)
    if "answer" not in obj or "p_yes" not in obj:
        return InvalidOutput(reason="missing answer and/or p_yes", raw=raw)
    if type(obj["answer"]) is not bool:
        return InvalidOutput(reason="answer must be a JSON boolean", raw=raw)
    p_yes = obj["p_yes"]
    if isinstance(p_yes, bool) or not isinstance(p_yes, (int, float)):
        return InvalidOutput(reason="p_yes must be a number in [0, 1]", raw=raw)
    p_yes_f = float(p_yes)
    if not 0.0 <= p_yes_f <= 1.0:
        return InvalidOutput(reason="p_yes must be in [0, 1]", raw=raw)
    return ModelOutput(answer=obj["answer"], p_yes=p_yes_f)


def confidence(p_yes: float) -> float:
    return max(p_yes, 1.0 - p_yes)
