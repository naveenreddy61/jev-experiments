"""c.compile generator: valid C11 translation units + tagged compile faults."""

from __future__ import annotations

import random
import re

from jev_eval.generators.base import Fault, Template, generate_task
from jev_eval.schema import Item
from jev_eval.tasks import get_task

TASK_ID = "c.compile"


def _t(name: str, difficulty: str, body: str) -> Template:
    def render(noise: str) -> str:
        return body.replace("{{NOISE}}", noise)

    return Template(name=name, difficulty=difficulty, render=render)


TEMPLATES = [
    _t(
        "add",
        "easy",
        "/* noise: {{NOISE}} */\nint add(int a, int b) {\n    return a + b;\n}\n",
    ),
    _t(
        "max",
        "easy",
        "/* noise: {{NOISE}} */\nstatic int max2(int a, int b) {\n    return a > b ? a : b;\n}\n",
    ),
    _t(
        "fact",
        "easy",
        "/* noise: {{NOISE}} */\nunsigned fact(unsigned n) {\n    unsigned acc = 1;\n    for (unsigned i = 2; i <= n; i++) {\n        acc *= i;\n    }\n    return acc;\n}\n",
    ),
    _t(
        "point",
        "medium",
        "/* noise: {{NOISE}} */\n#include <stddef.h>\nstruct Point {\n    int x;\n    int y;\n};\nint dist2(struct Point p) {\n    return p.x * p.x + p.y * p.y;\n}\n",
    ),
    _t(
        "enum_switch",
        "medium",
        "/* noise: {{NOISE}} */\nenum Color { RED, GREEN, BLUE };\nint pick(enum Color c) {\n    switch (c) {\n        case RED: return 1;\n        case GREEN: return 2;\n        case BLUE: return 3;\n        default: return 0;\n    }\n}\n",
    ),
    _t(
        "pointer_walk",
        "medium",
        "/* noise: {{NOISE}} */\nint sum(const int *xs, int n) {\n    int acc = 0;\n    for (int i = 0; i < n; i++) {\n        acc += xs[i];\n    }\n    return acc;\n}\n",
    ),
    _t(
        "macro_guard",
        "medium",
        "/* noise: {{NOISE}} */\n#define SQR(x) ((x) * (x))\nint energy(int v) {\n    return SQR(v);\n}\n#undef SQR\n",
    ),
    _t(
        "static_assert",
        "hard",
        "/* noise: {{NOISE}} */\n#include <stddef.h>\n_Static_assert(sizeof(int) >= 2, \"int width\");\nenum Color { RED, GREEN, BLUE };\nint pick(enum Color c) {\n    return (int)c;\n}\n",
    ),
    _t(
        "c11_for",
        "hard",
        "/* noise: {{NOISE}} */\n#include <stddef.h>\nint fold(int n, int xs[static 1]) {\n    int acc = 0;\n    for (int i = 0; i < n; i++) {\n        acc += xs[i];\n    }\n    return acc;\n}\n",
    ),
    _t(
        "compound",
        "hard",
        "/* noise: {{NOISE}} */\nstruct Pair { int a; int b; };\nint sum_pair(void) {\n    struct Pair p = { .a = 1, .b = 2 };\n    return p.a + p.b;\n}\n",
    ),
    _t(
        "nested_blocks",
        "hard",
        "/* noise: {{NOISE}} */\nint classify(int x) {\n    if (x < 0) {\n        return -1;\n    } else {\n        {\n            int y = x * 2;\n            return y > 10 ? 2 : 1;\n        }\n    }\n}\n",
    ),
    _t(
        "alignof",
        "hard",
        "/* noise: {{NOISE}} */\n#include <stddef.h>\nstruct Align {\n    _Alignas(8) int v;\n};\nsize_t width(void) {\n    return sizeof(struct Align);\n}\n",
    ),
]


def _missing_semicolon(src: str, rng: random.Random) -> str | None:
    lines = src.splitlines()
    candidates = [i for i, line in enumerate(lines) if line.rstrip().endswith(";")]
    if not candidates:
        return None
    i = rng.choice(candidates)
    lines[i] = lines[i].rstrip()[:-1]
    return "\n".join(lines) + "\n"


def _undeclared_ident(src: str, _rng: random.Random) -> str | None:
    return src.replace("return ", "return undeclared_", 1)


def _type_mismatch(src: str, _rng: random.Random) -> str | None:
    # Array assignment is a constraint violation (gcc error), unlike many
    # pointer/int mixes that are warnings-only without extra flags.
    extra = "\nint jev_type_trap(void) { int xs[2]; xs = 1; return 0; }\n"
    return src + extra


def _bad_preprocessor(src: str, _rng: random.Random) -> str | None:
    return "#if 1\n" + src


def _brace_mismatch(src: str, _rng: random.Random) -> str | None:
    if "}" not in src:
        return src + "\nint f(void) {\n    return 0;\n"
    return src.replace("}", "", 1)


def _extra_int(src: str, _rng: random.Random) -> str | None:
    return re.sub(r"\bint\b", "int int", src, count=1)


def _unclosed_macro(src: str, _rng: random.Random) -> str | None:
    return src + "\n#define JEV_OPEN(x\n"


def _missing_paren(src: str, _rng: random.Random) -> str | None:
    if ")" not in src:
        return None
    return src.replace(")", "", 1)


FAULTS = [
    Fault("missing_semicolon", False, _missing_semicolon),
    Fault("undeclared_ident", False, _undeclared_ident),
    Fault("type_mismatch", True, _type_mismatch),
    Fault("bad_preprocessor", True, _bad_preprocessor),
    Fault("brace_mismatch", False, _brace_mismatch),
    Fault("extra_int", False, _extra_int),
    Fault("unclosed_macro", True, _unclosed_macro),
    Fault("missing_paren", False, _missing_paren),
]


def generate(n: int, rng: random.Random) -> list[Item]:
    get_task(TASK_ID)
    return generate_task(TASK_ID, n, rng, TEMPLATES, FAULTS)
