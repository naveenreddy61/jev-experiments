"""Task oracles. Labels come from these tools only — never hand-label."""

from __future__ import annotations

from jev_eval.oracles.result import OracleResult, OracleToolMissing
from jev_eval.oracles import bash as bash_oracle
from jev_eval.oracles import c as c_oracle
from jev_eval.oracles import latex as latex_oracle
from jev_eval.oracles import python as python_oracle
from jev_eval.tasks import TASK_IDS

__all__ = ["OracleResult", "OracleToolMissing", "check"]

_DISPATCH = {
    "latex.parse": latex_oracle.check,
    "bash.parse": bash_oracle.check,
    "python.parse": python_oracle.check,
    "c.compile": c_oracle.check,
}


def check(task_id: str, artifact: str) -> OracleResult:
    if task_id not in _DISPATCH:
        known = ", ".join(TASK_IDS)
        raise KeyError(f"no oracle for {task_id!r}; v1 tasks: {known}")
    return _DISPATCH[task_id](artifact)
