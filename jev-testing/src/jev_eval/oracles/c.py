"""c.compile oracle: gcc -std=c11 -fsyntax-only (no link)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from jev_eval.oracles.result import OracleResult
from jev_eval.oracles._proc import first_line, require_tool, run

INSTALL_HINT = "Install gcc. The oracle is `gcc -std=c11 -fsyntax-only`."
GCC_ARGS = ["-std=c11", "-fsyntax-only"]


def _version_note(gcc: str) -> str:
    proc = run([gcc, "--version"], timeout=5)
    line = first_line(proc.stdout) or "gcc"
    return f"{line}; gcc {' '.join(GCC_ARGS)}"


def check(artifact: str) -> OracleResult:
    gcc = require_tool("gcc", INSTALL_HINT)
    note = _version_note(gcc)
    with tempfile.TemporaryDirectory(prefix="jev-c-") as tmp:
        path = Path(tmp) / "unit.c"
        path.write_text(artifact, encoding="utf-8")
        proc = run([gcc, *GCC_ARGS, str(path)], timeout=10)
    return OracleResult(
        ok=proc.returncode == 0,
        exit_code=int(proc.returncode),
        tool="gcc -std=c11 -fsyntax-only",
        version_note=note,
        detail=(proc.stderr or proc.stdout).strip(),
    )
