"""latex.parse generator: valid fragments + tagged structural faults."""

from __future__ import annotations

import random

from jev_eval.generators.base import Fault, Template, generate_task
from jev_eval.schema import Item
from jev_eval.tasks import get_task

TASK_ID = "latex.parse"


def _t(name: str, difficulty: str, body: str) -> Template:
    def render(noise: str) -> str:
        return body.replace("{{NOISE}}", noise)

    return Template(name=name, difficulty=difficulty, render=render)


TEMPLATES = [
    _t("hello", "easy", "% noise: {{NOISE}}\nHello world.\n"),
    _t(
        "textbf",
        "easy",
        "% noise: {{NOISE}}\n\\textbf{bold} and \\emph{emph}.\n",
    ),
    _t(
        "inline_math",
        "easy",
        "% noise: {{NOISE}}\nThe sum is $x + y = z$.\n",
    ),
    _t(
        "section",
        "easy",
        "% noise: {{NOISE}}\n\\section{Notes}\nA short paragraph.\n",
    ),
    _t(
        "frac_math",
        "medium",
        "% noise: {{NOISE}}\nA fraction: $\\frac{1}{2}$.\n",
    ),
    _t(
        "itemize",
        "medium",
        "% noise: {{NOISE}}\n\\begin{itemize}\n\\item one\n\\item two\n\\end{itemize}\n",
    ),
    _t(
        "equation",
        "medium",
        "% noise: {{NOISE}}\n\\begin{equation}\na^{2} + b^{2} = c^{2}\n\\end{equation}\n",
    ),
    _t(
        "enumerate_math",
        "medium",
        "% noise: {{NOISE}}\n\\begin{enumerate}\n\\item $a_i$\n\\item $b_j$\n\\end{enumerate}\n",
    ),
    _t(
        "tabular",
        "hard",
        "% noise: {{NOISE}}\n\\begin{tabular}{cc}\n1 & 2 \\\\\n3 & 4\n\\end{tabular}\n",
    ),
    _t(
        "nested",
        "hard",
        "% noise: {{NOISE}}\n\\section{Nested}\n\\begin{enumerate}\n\\item $f(x) = \\frac{x^{2}}{2}$\n\\item \\textbf{text} with $a_{i}$\n\\end{enumerate}\n",
    ),
    _t(
        "display_and_text",
        "hard",
        "% noise: {{NOISE}}\n\\[ \\frac{a+b}{c} \\]\nand some \\textit{prose}.\n",
    ),
    _t(
        "composed",
        "hard",
        "% noise: {{NOISE}}\n\\section{Mix}\n\\begin{itemize}\n\\item $\\alpha + \\beta$\n\\item \\textbf{done}\n\\end{itemize}\n\\begin{equation}\nx_{1} + x_{2}\n\\end{equation}\n",
    ),
]


def _unmatched_brace(src: str, _rng: random.Random) -> str | None:
    if "}" not in src:
        return src + "\\textbf{open\n"
    return src.replace("}", "", 1)


def _unmatched_dollar(src: str, _rng: random.Random) -> str | None:
    if src.count("$") >= 2:
        return src.replace("$", "", 1)
    return src + "Still open $x + y\n"


def _bad_begin_end(src: str, _rng: random.Random) -> str | None:
    if "\\end{itemize}" in src:
        return src.replace("\\end{itemize}", "\\end{enumerate}", 1)
    if "\\end{enumerate}" in src:
        return src.replace("\\end{enumerate}", "\\end{itemize}", 1)
    if "\\end{equation}" in src:
        return src.replace("\\end{equation}", "\\end{align}", 1)
    if "\\end{tabular}" in src:
        return src.replace("\\end{tabular}", "\\end{table}", 1)
    return src + "\\begin{itemize}\n\\item x\n\\end{enumerate}\n"


def _frac_arity(src: str, _rng: random.Random) -> str | None:
    if "\\frac{" in src:
        return src.replace("\\frac{1}{2}", "\\frac{1}", 1).replace(
            "\\frac{x^{2}}{2}", "\\frac{x^{2}}", 1
        ).replace("\\frac{a+b}{c}", "\\frac{a+b}", 1)
    return src + "Use $\\frac{1}$.\n"


def _math_mode_trap(src: str, _rng: random.Random) -> str | None:
    # \frac outside math mode is a real LaTeX error.
    return src + "\\frac{1}{2}\n"


def _unclosed_env(src: str, _rng: random.Random) -> str | None:
    for env in ("itemize", "enumerate", "equation", "tabular"):
        needle = f"\\end{{{env}}}"
        if needle in src:
            return src.replace(needle, "", 1)
    return src + "\\begin{itemize}\n\\item open\n"


def _extra_end(src: str, _rng: random.Random) -> str | None:
    return src + "\\end{itemize}\n"


def _unmatched_group(src: str, _rng: random.Random) -> str | None:
    return src + "\\textbf{still {nested\n"


FAULTS = [
    Fault("unmatched_brace", False, _unmatched_brace),
    Fault("unmatched_dollar", False, _unmatched_dollar),
    Fault("bad_begin_end", False, _bad_begin_end),
    Fault("frac_arity", True, _frac_arity),
    Fault("math_mode_trap", True, _math_mode_trap),
    Fault("unclosed_env", False, _unclosed_env),
    Fault("extra_end", False, _extra_end),
    Fault("unmatched_group", True, _unmatched_group),
]


def generate(n: int, rng: random.Random) -> list[Item]:
    get_task(TASK_ID)
    return generate_task(TASK_ID, n, rng, TEMPLATES, FAULTS)
