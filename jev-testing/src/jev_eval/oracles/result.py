from __future__ import annotations

from dataclasses import dataclass


class OracleToolMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class OracleResult:
    ok: bool
    exit_code: int
    tool: str
    version_note: str
    detail: str = ""

    def meta(self) -> dict:
        return {
            "tool": self.tool,
            "version_note": self.version_note,
            "exit_code": self.exit_code,
        }
