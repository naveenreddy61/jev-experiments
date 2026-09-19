from __future__ import annotations

import unittest
from pathlib import Path

from jev_eval.oracles import check

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class OracleFixtureTests(unittest.TestCase):
    def test_python_ok(self) -> None:
        src = (FIXTURES / "python_ok.py").read_text(encoding="utf-8")
        result = check("python.parse", src)
        self.assertTrue(result.ok, result.detail)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.tool, "ast.parse")

    def test_python_bad(self) -> None:
        src = (FIXTURES / "python_bad.py").read_text(encoding="utf-8")
        result = check("python.parse", src)
        self.assertFalse(result.ok)
        self.assertNotEqual(result.exit_code, 0)

    def test_python_never_execs(self) -> None:
        # A parse-valid module that would be catastrophic if exec'd.
        src = "raise SystemExit('oracle must not exec')\n"
        result = check("python.parse", src)
        self.assertTrue(result.ok, result.detail)

    def test_bash_ok(self) -> None:
        src = (FIXTURES / "bash_ok.sh").read_text(encoding="utf-8")
        result = check("bash.parse", src)
        self.assertTrue(result.ok, result.detail)
        self.assertEqual(result.exit_code, 0)

    def test_bash_bad(self) -> None:
        src = (FIXTURES / "bash_bad.sh").read_text(encoding="utf-8")
        result = check("bash.parse", src)
        self.assertFalse(result.ok)

    def test_c_ok(self) -> None:
        src = (FIXTURES / "c_ok.c").read_text(encoding="utf-8")
        result = check("c.compile", src)
        self.assertTrue(result.ok, result.detail)
        self.assertEqual(result.exit_code, 0)

    def test_c_bad(self) -> None:
        src = (FIXTURES / "c_bad.c").read_text(encoding="utf-8")
        result = check("c.compile", src)
        self.assertFalse(result.ok)

    def test_latex_ok(self) -> None:
        src = (FIXTURES / "latex_ok.tex").read_text(encoding="utf-8")
        result = check("latex.parse", src)
        self.assertTrue(result.ok, result.detail)
        self.assertEqual(result.exit_code, 0)

    def test_latex_bad(self) -> None:
        src = (FIXTURES / "latex_bad.tex").read_text(encoding="utf-8")
        result = check("latex.parse", src)
        self.assertFalse(result.ok)


if __name__ == "__main__":
    unittest.main()
