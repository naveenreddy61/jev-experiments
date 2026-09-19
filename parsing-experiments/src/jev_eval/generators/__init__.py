from __future__ import annotations

import random
from pathlib import Path

from jev_eval.generators import bash, c, latex, python
from jev_eval.schema import Item, dumps_item
from jev_eval.tasks import TASK_IDS

_GENERATORS = {
    "latex.parse": latex.generate,
    "bash.parse": bash.generate,
    "python.parse": python.generate,
    "c.compile": c.generate,
}


def generate(task_id: str, n: int, seed: int) -> list[Item]:
    if task_id not in _GENERATORS:
        known = ", ".join(TASK_IDS)
        raise KeyError(f"no generator for {task_id!r}; v1 tasks: {known}")
    rng = random.Random(f"{seed}:{task_id}")
    return _GENERATORS[task_id](n, rng)


def write_jsonl(items: list[Item], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(dumps_item(item))
            fh.write("\n")
