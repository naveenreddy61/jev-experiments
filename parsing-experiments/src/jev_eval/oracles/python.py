"""python.parse oracle: ast.parse only (never exec). Pins Python 3.12 grammar."""

from __future__ import annotations

import ast
import sys

from jev_eval.oracles.result import OracleResult

FEATURE_VERSION = (3, 12)
VERSION_NOTE = (
    f"ast.parse feature_version={FEATURE_VERSION}; "
    f"runtime {sys.version.split()[0]}"
)


def check(artifact: str) -> OracleResult:
    try:
        ast.parse(artifact, filename="<artifact>", feature_version=FEATURE_VERSION)
    except SyntaxError as exc:
        detail = f"{exc.msg} (line {exc.lineno})"
        return OracleResult(
            ok=False,
            exit_code=1,
            tool="ast.parse",
            version_note=VERSION_NOTE,
            detail=detail,
        )
    except ValueError as exc:
        # null bytes etc.
        return OracleResult(
            ok=False,
            exit_code=1,
            tool="ast.parse",
            version_note=VERSION_NOTE,
            detail=str(exc),
        )
    return OracleResult(
        ok=True,
        exit_code=0,
        tool="ast.parse",
        version_note=VERSION_NOTE,
        detail="",
    )
