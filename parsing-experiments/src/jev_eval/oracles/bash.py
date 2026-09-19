"""bash.parse oracle: `bash -n` (parse only, no execute)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from jev_eval.oracles.result import OracleResult
from jev_eval.oracles._proc import first_line, require_tool, run

INSTALL_HINT = "Install bash (GNU bash). The oracle is `bash -n`."


def _version_note(bash: str) -> str:
    proc = run([bash, "--version"], timeout=5)
    return first_line(proc.stdout) or "bash -n"


def check(artifact: str) -> OracleResult:
    bash = require_tool("bash", INSTALL_HINT)
    note = _version_note(bash)
    with tempfile.TemporaryDirectory(prefix="jev-bash-") as tmp:
        path = Path(tmp) / "snippet.sh"
        path.write_text(artifact, encoding="utf-8")
        proc = run([bash, "-n", str(path)], timeout=8)
    return OracleResult(
        ok=proc.returncode == 0,
        exit_code=int(proc.returncode),
        tool="bash -n",
        version_note=note,
        detail=(proc.stderr or proc.stdout).strip(),
    )
