from __future__ import annotations

import shutil
import subprocess

from jev_eval.oracles.result import OracleToolMissing


def require_tool(name: str, install_hint: str) -> str:
    path = shutil.which(name)
    if not path:
        raise OracleToolMissing(
            f"required tool {name!r} not found on PATH. {install_hint}"
        )
    return path


def run(
    argv: list[str],
    *,
    timeout: float,
    cwd: str | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            input=input_text,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv, 124, "", "timeout")


def first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""
