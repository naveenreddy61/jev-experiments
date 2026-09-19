"""Accuracy, invalid-rate, latency, and p_yes calibration hooks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from jev_eval.schema import confidence


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    ordered = sorted(xs)
    if len(ordered) == 1:
        return ordered[0]
    idx = q * (len(ordered) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def expected_calibration_error(
    golds: list[bool],
    p_yes: list[float],
    *,
    bins: int = 10,
) -> tuple[float | None, list[dict[str, Any]]]:
    """ECE over P(yes) bins, plus a reliability table.

    Bin i covers [i/bins, (i+1)/bins]; the last bin includes 1.0.
    """
    if not golds or len(golds) != len(p_yes) or bins < 1:
        return None, []
    bucket_p: dict[int, list[float]] = defaultdict(list)
    bucket_y: dict[int, list[float]] = defaultdict(list)
    for y, p in zip(golds, p_yes, strict=True):
        if p >= 1.0:
            idx = bins - 1
        else:
            idx = min(bins - 1, max(0, int(p * bins)))
        bucket_p[idx].append(p)
        bucket_y[idx].append(1.0 if y else 0.0)

    ece = 0.0
    n = len(golds)
    table: list[dict[str, Any]] = []
    for idx in range(bins):
        ps = bucket_p.get(idx, [])
        ys = bucket_y.get(idx, [])
        count = len(ps)
        avg_p = sum(ps) / count if count else None
        emp = sum(ys) / count if count else None
        if count and avg_p is not None and emp is not None:
            ece += (count / n) * abs(avg_p - emp)
        table.append(
            {
                "bin": idx,
                "lo": idx / bins,
                "hi": (idx + 1) / bins,
                "count": count,
                "avg_p_yes": avg_p,
                "empirical_yes_rate": emp,
            }
        )
    return ece, table


def brier_score(golds: list[bool], p_yes: list[float]) -> float | None:
    if not golds or len(golds) != len(p_yes):
        return None
    return sum((p - (1.0 if y else 0.0)) ** 2 for y, p in zip(golds, p_yes, strict=True)) / len(
        golds
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def score_rows(rows: Iterable[dict[str, Any]], *, include_by_task: bool = True) -> dict[str, Any]:
    rows = list(rows)
    n = len(rows)
    invalid = [r for r in rows if r.get("invalid")]
    valid = [r for r in rows if not r.get("invalid")]
    correct = [r for r in valid if r.get("answer") is r.get("gold")]

    latencies = [float(r["latency_ms"]) for r in rows if _is_number(r.get("latency_ms"))]

    cal_golds: list[bool] = []
    cal_p: list[float] = []
    confidences: list[float] = []
    for r in valid:
        p = r.get("p_yes")
        gold = r.get("gold")
        if type(gold) is bool and _is_number(p):
            cal_golds.append(gold)
            cal_p.append(float(p))
            confidences.append(confidence(float(p)))

    ece, reliability = expected_calibration_error(cal_golds, cal_p)
    summary: dict[str, Any] = {
        "n": n,
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "invalid_rate": (len(invalid) / n) if n else 0.0,
        "accuracy": (len(correct) / len(valid)) if valid else None,
        "accuracy_note": "accuracy is computed on valid outputs only; invalids are not imputed",
        "latency_ms": {
            "mean": _mean(latencies),
            "p50": _quantile(latencies, 0.50),
            "p95": _quantile(latencies, 0.95),
        },
        "calibration": {
            "n_with_p_yes": len(cal_p),
            "brier": brier_score(cal_golds, cal_p),
            "ece": ece,
            "mean_confidence": _mean(confidences),
            "confidence_note": "confidence = max(p_yes, 1 - p_yes)",
            "reliability": reliability,
        },
    }
    if include_by_task:
        tasks = sorted({str(r.get("task")) for r in rows if r.get("task")})
        if len(tasks) > 1:
            summary["by_task"] = {
                task: score_rows(
                    [r for r in rows if r.get("task") == task],
                    include_by_task=False,
                )
                for task in tasks
            }
    return summary
