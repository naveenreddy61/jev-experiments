"""How well Jev's Score tracks one person's labels.

Per premise (the GEPA unit):

- `concordance`: fraction of punchline pairs with different labels that Jev
  orders the same way the person did. Ties count one half. This is the
  recommender goal: pick the punchline the person prefers.
- `agreement`: mean probability Jev puts on the person's level. This keeps
  the three levels meaningful, not only the order.
- `premise_score = w_rank * concordance + (1 - w_rank) * agreement`. When a
  premise has no pair with different labels, the score is `agreement` alone.

Over a whole split, `summarize` also reports exact-level accuracy, mean
absolute error of the expected score, Spearman correlation, top-1 hit rate,
and a confusion table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from joke_pref.data import LABELS
from joke_pref.scorer import ScoreResult

W_RANK = 0.6


@dataclass(frozen=True)
class Pair:
    hi_id: str  # the punchline the person rated higher
    lo_id: str
    hi_pred: float
    lo_pred: float

    @property
    def credit(self) -> float:
        if self.hi_pred > self.lo_pred:
            return 1.0
        if self.hi_pred == self.lo_pred:
            return 0.5
        return 0.0


def pairs(
    preds: Sequence[float], labels: Sequence[int], ids: Sequence[str]
) -> list[Pair]:
    out: list[Pair] = []
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            if labels[i] == labels[j]:
                continue
            hi, lo = (i, j) if labels[i] > labels[j] else (j, i)
            out.append(Pair(ids[hi], ids[lo], preds[hi], preds[lo]))
    return out


def concordance(ps: Iterable[Pair]) -> float | None:
    ps = list(ps)
    if not ps:
        return None
    return sum(p.credit for p in ps) / len(ps)


def agreement(results: Sequence[ScoreResult], labels: Sequence[int]) -> float:
    """Mean probability on the true level. An invalid result counts as 0."""
    if not results:
        return 0.0
    total = 0.0
    for r, lab in zip(results, labels):
        if r.probabilities is not None:
            total += r.probabilities[lab]
    return total / len(results)


@dataclass(frozen=True)
class PremiseEval:
    score: float
    concordance: float | None
    agreement: float
    inverted: tuple[Pair, ...]  # pairs Jev ordered the wrong way


def premise_score(
    results: Sequence[ScoreResult],
    labels: Sequence[int],
    ids: Sequence[str],
    *,
    w_rank: float = W_RANK,
) -> PremiseEval:
    preds = [r.score if r.score is not None else -1.0 for r in results]
    ps = pairs(preds, labels, ids)
    conc = concordance(ps)
    agr = agreement(results, labels)
    score = agr if conc is None else w_rank * conc + (1.0 - w_rank) * agr
    inverted = tuple(p for p in ps if p.credit == 0.0)
    return PremiseEval(score=score, concordance=conc, agreement=agr, inverted=inverted)


# --- split-level summary ----------------------------------------------------


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=xs.__getitem__)
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


@dataclass
class Row:
    premise_id: str
    punchline_id: str
    label: int
    result: ScoreResult


def summarize(rows: Sequence[Row], evals: Sequence[PremiseEval]) -> dict:
    valid = [r for r in rows if r.result.valid]
    n = len(rows)
    out: dict = {
        "n_punchlines": n,
        "n_premises": len(evals),
        "invalid_rate": (n - len(valid)) / n if n else None,
        "premise_score": _mean([e.score for e in evals]),
        "concordance": _mean([e.concordance for e in evals if e.concordance is not None]),
        "agreement": _mean([e.agreement for e in evals]),
    }
    if valid:
        labels = [r.label for r in valid]
        preds = [r.result.score for r in valid]  # type: ignore[misc]
        argmax = [r.result.argmax for r in valid]
        out["exact_accuracy"] = sum(a == lab for a, lab in zip(argmax, labels)) / len(valid)
        out["mae"] = sum(abs(p - lab) for p, lab in zip(preds, labels)) / len(valid)  # type: ignore[operator]
        out["spearman"] = spearman(preds, [float(x) for x in labels])  # type: ignore[arg-type]
        out["mean_confidence"] = _mean([r.result.confidence for r in valid])  # type: ignore[misc]
        out["mean_pred_score"] = _mean(preds)  # type: ignore[arg-type]
        conf = [[0] * len(LABELS) for _ in LABELS]
        for a, lab in zip(argmax, labels):
            conf[lab][a] += 1  # type: ignore[index]
        out["confusion_rows_true_cols_pred"] = conf
        out["label_counts"] = {name: labels.count(i) for i, name in enumerate(LABELS)}
        out["top1_hit_rate"] = top1_hit_rate(rows)
    return out


def top1_hit_rate(rows: Sequence[Row]) -> float | None:
    """Per premise: does the punchline with the highest Jev score carry the
    person's best label? Premises where every label is equal are skipped."""
    by_premise: dict[str, list[Row]] = {}
    for r in rows:
        by_premise.setdefault(r.premise_id, []).append(r)
    hits, total = 0, 0
    for group in by_premise.values():
        valid = [r for r in group if r.result.valid]
        if len(valid) < 2 or len({r.label for r in valid}) < 2:
            continue
        best_label = max(r.label for r in valid)
        top = max(valid, key=lambda r: r.result.score)  # type: ignore[arg-type,return-value]
        hits += int(top.label == best_label)
        total += 1
    return hits / total if total else None


def _mean(xs: Sequence[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None
