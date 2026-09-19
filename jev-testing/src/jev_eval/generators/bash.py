"""bash.parse generator: valid snippets + tagged parse faults."""

from __future__ import annotations

import random

from jev_eval.generators.base import Fault, Template, generate_task
from jev_eval.schema import Item
from jev_eval.tasks import get_task

TASK_ID = "bash.parse"


def _t(name: str, difficulty: str, body: str) -> Template:
    def render(noise: str) -> str:
        return body.replace("{{NOISE}}", noise)

    return Template(name=name, difficulty=difficulty, render=render)


TEMPLATES = [
    _t("echo", "easy", "# noise: {{NOISE}}\necho hello\n"),
    _t(
        "assign",
        "easy",
        '# noise: {{NOISE}}\nname="world"\necho "$name"\n',
    ),
    _t(
        "pipeline",
        "easy",
        "# noise: {{NOISE}}\nprintf '%s\\n' a b c | wc -l\n",
    ),
    _t(
        "simple_if",
        "easy",
        '# noise: {{NOISE}}\nif test -n "$HOME"; then\n  echo home\nfi\n',
    ),
    _t(
        "if_double",
        "medium",
        '# noise: {{NOISE}}\nif [[ -n "$1" ]]; then\n  echo "got $1"\nelse\n  echo none\nfi\n',
    ),
    _t(
        "for_arith",
        "medium",
        "# noise: {{NOISE}}\nsum=0\nfor i in 1 2 3 4; do\n  sum=$((sum + i))\ndone\necho \"$sum\"\n",
    ),
    _t(
        "while_count",
        "medium",
        "# noise: {{NOISE}}\nn=0\nwhile (( n < 3 )); do\n  n=$((n + 1))\ndone\n",
    ),
    _t(
        "function_local",
        "medium",
        "# noise: {{NOISE}}\ngreet() {\n  local who=$1\n  echo \"hi ${who:-stranger}\"\n}\ngreet alice\n",
    ),
    _t(
        "case_esac",
        "medium",
        "# noise: {{NOISE}}\nwho=${1:-guest}\ncase \"$who\" in\n  alice|bob) echo known ;;\n  *) echo other ;;\nesac\n",
    ),
    _t(
        "nested_if_case",
        "hard",
        '# noise: {{NOISE}}\ngreet() {\n  local who=$1\n  case "$who" in\n    alice|bob) echo "hi $who" ;;\n    *) echo hey ;;\n  esac\n}\nif [[ ${#} -gt 0 ]]; then\n  greet "$1"\nfi\n',
    ),
    _t(
        "heredoc",
        "hard",
        "# noise: {{NOISE}}\ncat <<'EOF'\nline one\nline two {{NOISE}}\nEOF\n",
    ),
    _t(
        "composed",
        "hard",
        "# noise: {{NOISE}}\nset -euo pipefail\nlog() { printf '%s\\n' \"$1\"; }\nif [[ -d /tmp ]]; then\n  for name in a b; do\n    log \"$name\"\n  done\nfi\n",
    ),
]


def _quote_imbalance(src: str, _rng: random.Random) -> str | None:
    if '"' not in src:
        return src + '\necho "open\n'
    return src.replace('"', "", 1)


def _if_fi_mismatch(src: str, _rng: random.Random) -> str | None:
    if "\nfi\n" in src or src.endswith("\nfi"):
        return src.replace("fi", "", 1)
    return src + "\nif true; then\n  echo x\n"


def _then_missing(src: str, _rng: random.Random) -> str | None:
    if "then" not in src:
        return src + '\nif [[ -n "$x" ]];\n  echo x\nfi\n'
    return src.replace("then", "", 1)


def _bad_double_bracket(src: str, _rng: random.Random) -> str | None:
    if "]]" in src:
        return src.replace("]]", "]", 1)
    if "[[" in src:
        return src.replace("[[", "[", 1)
    return src + '\nif [[ -n "$x" ]; then\n  echo x\nfi\n'


def _bad_arithmetic(src: str, _rng: random.Random) -> str | None:
    if "((" in src and "))" in src:
        return src.replace("))", ")", 1)
    return src + "\n(( n + 1 )\n"


def _redirect_typo(src: str, _rng: random.Random) -> str | None:
    return src + "\necho hi 2>\n"


def _case_esac_mismatch(src: str, _rng: random.Random) -> str | None:
    if "esac" in src:
        return src.replace("esac", "", 1)
    return src + '\ncase "$x" in\n  a) echo a ;;\n'


def _do_done_mismatch(src: str, _rng: random.Random) -> str | None:
    if "done" in src:
        return src.replace("done", "", 1)
    return src + "\nfor i in 1 2; do\n  echo $i\n"


def _unclosed_heredoc(src: str, _rng: random.Random) -> str | None:
    if "EOF" in src:
        return src.replace("EOF", "END", 1)
    return src + "\ncat <<EOF\nstill open\n"


FAULTS = [
    Fault("quote_imbalance", False, _quote_imbalance),
    Fault("if_fi_mismatch", False, _if_fi_mismatch),
    Fault("then_missing", False, _then_missing),
    Fault("bad_double_bracket", True, _bad_double_bracket),
    Fault("bad_arithmetic", True, _bad_arithmetic),
    Fault("redirect_typo", True, _redirect_typo),
    Fault("case_esac_mismatch", False, _case_esac_mismatch),
    Fault("do_done_mismatch", False, _do_done_mismatch),
    Fault("unclosed_heredoc", True, _unclosed_heredoc),
]


def generate(n: int, rng: random.Random) -> list[Item]:
    get_task(TASK_ID)
    return generate_task(TASK_ID, n, rng, TEMPLATES, FAULTS)
