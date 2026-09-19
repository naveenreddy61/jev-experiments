"""latex.parse oracle: pdflatex dry-run of a fragment wrapped in article.

Headless command (exit 0 means yes):

    pdflatex -interaction=nonstopmode -halt-on-error -draftmode \\
        -output-directory <tmp> fragment.tex

Install (Debian/Ubuntu):

    sudo apt-get install -y --no-install-recommends texlive-latex-base

tectonic is a fine alternative engine for other machines, but this suite
pins pdflatex so seed labels stay reproducible.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from jev_eval.oracles.result import OracleResult
from jev_eval.oracles._proc import first_line, require_tool, run

INSTALL_HINT = (
    "Install TeX Live (pdflatex). On Debian/Ubuntu: "
    "sudo apt-get install -y --no-install-recommends texlive-latex-base"
)
PDFLATEX_ARGS = [
    "-interaction=nonstopmode",
    "-halt-on-error",
    "-draftmode",
]


def wrap_fragment(artifact: str) -> str:
    return (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        f"{artifact}\n"
        "\\end{document}\n"
    )


def _version_note(pdflatex: str) -> str:
    proc = run([pdflatex, "--version"], timeout=5)
    line = first_line(proc.stdout) or "pdflatex"
    return f"{line}; {' '.join(PDFLATEX_ARGS)} on article-wrapped fragment"


def check(artifact: str) -> OracleResult:
    pdflatex = require_tool("pdflatex", INSTALL_HINT)
    note = _version_note(pdflatex)
    source = wrap_fragment(artifact)
    with tempfile.TemporaryDirectory(prefix="jev-latex-") as tmp:
        path = Path(tmp) / "fragment.tex"
        path.write_text(source, encoding="utf-8")
        proc = run(
            [pdflatex, *PDFLATEX_ARGS, f"-output-directory={tmp}", str(path)],
            timeout=20,
            cwd=tmp,
        )
    return OracleResult(
        ok=proc.returncode == 0,
        exit_code=int(proc.returncode),
        tool="pdflatex",
        version_note=note,
        detail=(proc.stderr or proc.stdout).strip()[-2000:],
    )
