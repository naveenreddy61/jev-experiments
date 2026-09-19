"""Shared generation loop: templates → optional tagged faults → oracle gate."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from jev_eval.oracles import check
from jev_eval.schema import Item
from jev_eval.tasks import get_task

Difficulty = str
Mutator = Callable[[str, random.Random], str | None]


@dataclass(frozen=True)
class Template:
    name: str
    difficulty: Difficulty
    render: Callable[[str], str]


@dataclass(frozen=True)
class Fault:
    tag: str
    subtle: bool
    apply: Mutator


@dataclass
class Candidate:
    difficulty: str
    artifact: str
    intended_yes: bool
    fault_tags: list[str]
    pair_key: str
    template: str


def noise_token(rng: random.Random) -> str:
    alphabet = "abcdef0123456789"
    return "".join(rng.choice(alphabet) for _ in range(12))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _difficulty_for(base: str, tags: list[str], subtle: bool) -> str:
    if not tags:
        return base
    if subtle:
        return "hard"
    return base


def _select_balanced(candidates: list[Candidate], n: int, rng: random.Random) -> list[Candidate]:
    """Prefer with/without-fault pairs, diversify faults, then fill polarity/difficulty."""
    by_pair: dict[str, dict[bool, list[Candidate]]] = defaultdict(lambda: defaultdict(list))
    for cand in candidates:
        by_pair[cand.pair_key][cand.intended_yes].append(cand)

    pair_keys = [key for key, buckets in by_pair.items() if buckets.get(True) and buckets.get(False)]
    rng.shuffle(pair_keys)

    selected: list[Candidate] = []
    used_hashes: set[str] = set()
    fault_counts: Counter[str] = Counter()
    bucket_counts: Counter[tuple[str, bool]] = Counter()

    def take(cand: Candidate) -> bool:
        h = _sha(cand.artifact)
        if h in used_hashes:
            return False
        used_hashes.add(h)
        selected.append(cand)
        for tag in cand.fault_tags:
            fault_counts[tag] += 1
        bucket_counts[(cand.difficulty, cand.intended_yes)] += 1
        return True

    yes_target = n // 2
    no_target = n - yes_target
    yes_n = 0
    no_n = 0

    def polarity_ok(cand: Candidate) -> bool:
        if cand.intended_yes:
            return yes_n < yes_target
        return no_n < no_target

    def no_sort_key(cand: Candidate) -> tuple:
        tag = cand.fault_tags[0] if cand.fault_tags else ""
        return (fault_counts[tag], bucket_counts[(cand.difficulty, False)], _sha(cand.artifact))

    for key in pair_keys:
        if len(selected) + 2 > n or yes_n >= yes_target or no_n >= no_target:
            break
        yes_c = by_pair[key][True][0]
        no_c = sorted(by_pair[key][False], key=no_sort_key)[0]
        if take(yes_c):
            yes_n += 1
        if take(no_c):
            no_n += 1

    pool = [c for c in candidates if _sha(c.artifact) not in used_hashes]
    rng.shuffle(pool)
    while len(selected) < n and pool:
        pool.sort(
            key=lambda c: (
                0 if polarity_ok(c) else 1,
                fault_counts[c.fault_tags[0]] if c.fault_tags else 0,
                bucket_counts[(c.difficulty, c.intended_yes)],
                _sha(c.artifact),
            )
        )
        cand = pool.pop(0)
        if not polarity_ok(cand):
            continue
        if take(cand):
            if cand.intended_yes:
                yes_n += 1
            else:
                no_n += 1
        pool = [c for c in pool if _sha(c.artifact) not in used_hashes]

    return selected[:n]


def generate_task(
    task_id: str,
    n: int,
    rng: random.Random,
    templates: list[Template],
    faults: list[Fault],
    *,
    rounds: int = 6,
) -> list[Item]:
    spec = get_task(task_id)
    candidates: list[Candidate] = []
    seen: set[str] = set()

    def consider(cand: Candidate) -> None:
        h = _sha(cand.artifact)
        if h in seen:
            return
        result = check(task_id, cand.artifact)
        if result.ok != cand.intended_yes:
            return
        seen.add(h)
        candidates.append(cand)

    for round_i in range(rounds):
        for tmpl in templates:
            token = noise_token(rng)
            base = tmpl.render(token)
            pair_key = f"{tmpl.name}:{round_i}:{token}"
            consider(
                Candidate(
                    difficulty=tmpl.difficulty,
                    artifact=base,
                    intended_yes=True,
                    fault_tags=[],
                    pair_key=pair_key,
                    template=tmpl.name,
                )
            )
            ordered_faults = list(faults)
            rng.shuffle(ordered_faults)
            for fault in ordered_faults:
                mutated = fault.apply(base, rng)
                if mutated is None or mutated == base:
                    continue
                consider(
                    Candidate(
                        difficulty=_difficulty_for(tmpl.difficulty, [fault.tag], fault.subtle),
                        artifact=mutated,
                        intended_yes=False,
                        fault_tags=[fault.tag],
                        pair_key=pair_key,
                        template=tmpl.name,
                    )
                )

    selected = _select_balanced(candidates, n, rng)
    if len(selected) < n:
        # Last-resort: keep generating unpaired items from remaining templates.
        extra_rounds = 0
        while len(selected) < n and extra_rounds < 20:
            extra_rounds += 1
            tmpl = rng.choice(templates)
            token = noise_token(rng)
            base = tmpl.render(token)
            pair_key = f"extra:{tmpl.name}:{extra_rounds}:{token}"
            need_yes = sum(1 for c in selected if c.intended_yes) < n // 2
            if need_yes:
                consider(
                    Candidate(
                        difficulty=tmpl.difficulty,
                        artifact=base,
                        intended_yes=True,
                        fault_tags=[],
                        pair_key=pair_key,
                        template=tmpl.name,
                    )
                )
            else:
                fault = rng.choice(faults)
                mutated = fault.apply(base, rng)
                if mutated:
                    consider(
                        Candidate(
                            difficulty=_difficulty_for(tmpl.difficulty, [fault.tag], fault.subtle),
                            artifact=mutated,
                            intended_yes=False,
                            fault_tags=[fault.tag],
                            pair_key=pair_key,
                            template=tmpl.name,
                        )
                    )
            selected = _select_balanced(candidates, n, rng)

    selected.sort(key=lambda c: (c.difficulty, not c.intended_yes, _sha(c.artifact)))
    items: list[Item] = []
    for i, cand in enumerate(selected, start=1):
        result = check(task_id, cand.artifact)
        items.append(
            Item(
                id=f"{task_id}.{i:05d}",
                task=task_id,
                difficulty=cand.difficulty,
                artifact=cand.artifact,
                question=spec.question,
                answer=result.ok,
                fault_tags=tuple(cand.fault_tags) if not result.ok else tuple(),
                oracle_meta=result.meta(),
            )
        )
    return items
