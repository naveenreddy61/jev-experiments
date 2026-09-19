"""Locked v1 task registry.

Atomic only: never mix parse and run. Future ids (python.exec, will-run)
are reserved here as comments / constants and are not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

# Reserved for a later suite revision. Do not implement in v1.
FUTURE_TASK_IDS: tuple[str, ...] = ("python.exec", "will-run")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    question: str
    oracle: str
    yes_means: str


TASKS: dict[str, TaskSpec] = {
    "latex.parse": TaskSpec(
        task_id="latex.parse",
        question="Does this parse as LaTeX?",
        oracle="pdflatex -interaction=nonstopmode -halt-on-error -draftmode (fragment wrapped in article)",
        yes_means="exit 0",
    ),
    "bash.parse": TaskSpec(
        task_id="bash.parse",
        question="Does this parse as bash?",
        oracle="bash -n",
        yes_means="exit 0",
    ),
    "python.parse": TaskSpec(
        task_id="python.parse",
        question="Does this parse as Python 3.12?",
        oracle="ast.parse(..., feature_version=(3, 12)) only — never exec",
        yes_means="no SyntaxError",
    ),
    "c.compile": TaskSpec(
        task_id="c.compile",
        question="Does this compile as C11 (no link)?",
        oracle="gcc -std=c11 -fsyntax-only",
        yes_means="exit 0",
    ),
}

TASK_IDS: tuple[str, ...] = tuple(TASKS.keys())


def get_task(task_id: str) -> TaskSpec:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        known = ", ".join(TASK_IDS)
        raise KeyError(f"unknown task {task_id!r}; v1 tasks: {known}") from exc
