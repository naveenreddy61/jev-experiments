"""python.parse generator: valid 3.12 templates + tagged syntax faults."""

from __future__ import annotations

import random
import re

from jev_eval.generators.base import Fault, Template, generate_task
from jev_eval.schema import Item
from jev_eval.tasks import get_task

TASK_ID = "python.parse"


def _t(name: str, difficulty: str, body: str) -> Template:
    def render(noise: str) -> str:
        return body.replace("{{NOISE}}", noise)

    return Template(name=name, difficulty=difficulty, render=render)


TEMPLATES = [
    _t(
        "assign",
        "easy",
        "# noise: {{NOISE}}\nx = 1\ny = x + 2\n",
    ),
    _t(
        "function",
        "easy",
        "# noise: {{NOISE}}\ndef add(a, b):\n    return a + b\n",
    ),
    _t(
        "hello",
        "easy",
        "# noise: {{NOISE}}\nname = 'world'\nprint(f'hello {name}')\n",
    ),
    _t(
        "listcomp",
        "easy",
        "# noise: {{NOISE}}\nvalues = [i * i for i in range(8) if i % 2 == 0]\n",
    ),
    _t(
        "nested_fn",
        "medium",
        "# noise: {{NOISE}}\ndef outer(n):\n    def inner(k):\n        return k * n\n    return [inner(i) for i in range(n)]\n",
    ),
    _t(
        "point_class",
        "medium",
        "# noise: {{NOISE}}\nclass Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n\n    def dist2(self):\n        return self.x * self.x + self.y * self.y\n",
    ),
    _t(
        "try_with",
        "medium",
        "# noise: {{NOISE}}\ndef read_pair(path):\n    try:\n        with open(path, encoding='utf-8') as fh:\n            left, right = fh.read().split(',', 1)\n            return left.strip(), right.strip()\n    except OSError:\n        return None\n",
    ),
    _t(
        "walrus_loop",
        "medium",
        "# noise: {{NOISE}}\ndef take_until(xs, limit):\n    acc = []\n    i = 0\n    while i < len(xs) and (total := sum(acc)) < limit:\n        acc.append(xs[i])\n        i += 1\n    return acc, total if acc else 0\n",
    ),
    _t(
        "match_case",
        "medium",
        "# noise: {{NOISE}}\ndef label(value):\n    match value:\n        case 0:\n            return 'zero'\n        case int() as n if n > 0:\n            return 'pos'\n        case _:\n            return 'other'\n",
    ),
    _t(
        "type_params",
        "hard",
        "# noise: {{NOISE}}\ntype Vector[T] = list[T]\n\ndef map_vec[T, U](xs: Vector[T], fn):\n    out: Vector[U] = []\n    for x in xs:\n        if (y := fn(x)) is not None:\n            out.append(y)\n    return out\n",
    ),
    _t(
        "match_generic",
        "hard",
        "# noise: {{NOISE}}\ntype Pair[T] = tuple[T, T]\n\ndef first[T](p: Pair[T]) -> T:\n    match p:\n        case (a, _):\n            return a\n        case _:\n            raise ValueError('bad pair')\n",
    ),
    _t(
        "composed",
        "hard",
        "# noise: {{NOISE}}\nfrom contextlib import contextmanager\n\n@contextmanager\ndef nest(name):\n    yield name\n\ndef pipeline(rows):\n    out = []\n    with nest('stage'):\n        for row in rows:\n            match row:\n                case {'ok': True, 'n': int() as n} if n >= 0:\n                    out.append(f'{n}')\n                case _:\n                    continue\n    return out\n",
    ),
]


def _indent_error(src: str, rng: random.Random) -> str | None:
    lines = src.splitlines()
    indented = [i for i, line in enumerate(lines) if line.startswith("    ")]
    if not indented:
        return None
    i = rng.choice(indented)
    lines[i] = lines[i].lstrip()
    return "\n".join(lines) + "\n"


def _bad_fstring(src: str, rng: random.Random) -> str | None:
    if "f'" in src:
        return src.replace("f'", "f'", 1).replace("'}", "}", 1) if "'}" in src else src + "\nbad = f'{oops'\n"
    if 'f"' in src:
        return re.sub(r'f"([^"]*)"', r'f"\1', src, count=1)
    return src + '\nlabel = f"{value"\n'


def _unmatched_paren(src: str, _rng: random.Random) -> str | None:
    if ")" not in src:
        return src + "\nvalues = (1, 2\n"
    return src.replace(")", "", 1)


def _unmatched_bracket(src: str, _rng: random.Random) -> str | None:
    if "]" not in src:
        return src + "\nvalues = [1, 2, 3\n"
    return src.replace("]", "", 1)


def _invalid_walrus(src: str, _rng: random.Random) -> str | None:
    return src + "\nacc := 0\n"


def _invalid_match(src: str, _rng: random.Random) -> str | None:
    if "case " not in src:
        return src + "\nmatch x:\n    case 1\n        y = 1\n"
    return src.replace("case ", "case ", 1).replace(":\n", "\n", 1)


def _duplicate_arg(src: str, _rng: random.Random) -> str | None:
    match = re.search(r"def (\w+)\(([a-zA-Z_][a-zA-Z0-9_]*)", src)
    if not match:
        return src + "\ndef dup(x, x):\n    return x\n"
    name, arg = match.group(1), match.group(2)
    return src + f"\ndef {name}_dup({arg}, {arg}):\n    return {arg}\n"


def _return_outside(src: str, _rng: random.Random) -> str | None:
    return src + "\nreturn 0\n"


def _unclosed_string(src: str, _rng: random.Random) -> str | None:
    return src + '\nmsg = "still open\n'


FAULTS = [
    Fault("indent_error", False, _indent_error),
    Fault("bad_fstring", True, _bad_fstring),
    Fault("unmatched_paren", False, _unmatched_paren),
    Fault("unmatched_bracket", False, _unmatched_bracket),
    Fault("invalid_walrus", True, _invalid_walrus),
    Fault("invalid_match", True, _invalid_match),
    Fault("duplicate_arg", True, _duplicate_arg),
    Fault("return_outside", False, _return_outside),
    Fault("unclosed_string", False, _unclosed_string),
]


def generate(n: int, rng: random.Random) -> list[Item]:
    get_task(TASK_ID)
    return generate_task(TASK_ID, n, rng, TEMPLATES, FAULTS)
