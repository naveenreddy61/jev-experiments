"""Phase 2 report: personal rubric against generic, cross-user and null.

Reads `results/jester/personal/users.csv` (and `results/jester/null/users.csv`
when present) and prints the summary table, a paired sign test with an exact
binomial p-value, a bootstrap interval on the mean paired difference, and
the share of the generic-to-kNN gap that the personal rubric closes.

```bash
.venv/bin/python scripts/jester_phase2_report.py [--personal PATH] [--null PATH]
```
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} has no rows yet")
    cols = {}
    for k in ("user_id", "personal", "generic", "cross_user_median", "cross_user_partner", "crowd", "knn", "personal_agreement", "num_candidates", "jev_calls", "elapsed_s"):
        cols[k] = np.array([float(r[k]) if r[k] not in ("", "None", "nan") else np.nan for r in rows])
    return cols


def sign_test(diff: np.ndarray) -> tuple[int, int, float]:
    """Wins, losses and the exact two-sided binomial p-value, ties dropped."""
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    n = wins + losses
    if n == 0:
        return wins, losses, 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return wins, losses, min(1.0, 2 * tail)


def bootstrap_mean(x: np.ndarray, seed: int = 0, n: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def report(cols: dict[str, np.ndarray], title: str) -> None:
    n = cols["user_id"].size
    print(f"\n== {title}: {n} users ==")
    print(f"{'column':<22s}{'median':>8s}{'mean':>8s}")
    for k in ("crowd", "generic", "personal", "cross_user_median", "cross_user_partner", "knn"):
        print(f"{k:<22s}{np.nanmedian(cols[k]):8.3f}{np.nanmean(cols[k]):8.3f}")
    for a, b in (("personal", "generic"), ("personal", "cross_user_partner"), ("cross_user_partner", "generic"), ("personal", "crowd")):
        d = cols[a] - cols[b]
        d = d[~np.isnan(d)]
        wins, losses, p = sign_test(d)
        lo, hi = bootstrap_mean(d)
        print(f"{a} - {b}: mean {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], median {np.median(d):+.3f}, wins {wins} / losses {losses}, sign test p = {p:.3g}")
    gap = cols["knn"] - cols["generic"]
    closed = (cols["personal"] - cols["generic"]) / gap
    ok = gap > 0.02
    print(f"share of the generic->kNN gap closed (users with gap > 0.02, n={int(ok.sum())}): median {np.nanmedian(closed[ok]):+.2f}, mean {np.nanmean(closed[ok]):+.2f}")
    print(f"users where personal >= kNN: {int((cols['personal'] >= cols['knn']).sum())}")
    print(f"GEPA: median candidates {np.nanmedian(cols['num_candidates']):.0f}, median Jev calls per user {np.nanmedian(cols['jev_calls']):.0f}, median elapsed {np.nanmedian(cols['elapsed_s'])/60:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--personal", default=ROOT / "results/jester/personal/users.csv", type=Path)
    ap.add_argument("--null", default=ROOT / "results/jester/null/users.csv", type=Path)
    args = ap.parse_args()
    personal = load(args.personal)
    report(personal, "personal rubric")
    top = personal["user_id"].size
    if args.null.exists():
        null = load(args.null)
        report(null, "null run (shuffled train labels)")
        both = np.intersect1d(personal["user_id"], null["user_id"])
        if both.size:
            p = {int(u): v for u, v in zip(personal["user_id"], personal["personal"])}
            q = {int(u): v for u, v in zip(null["user_id"], null["personal"])}
            g = {int(u): v for u, v in zip(personal["user_id"], personal["generic"])}
            d_real = np.array([p[int(u)] - g[int(u)] for u in both])
            d_null = np.array([q[int(u)] - g[int(u)] for u in both])
            print(f"\nsame {both.size} users: personal - generic real {d_real.mean():+.3f}, null {d_null.mean():+.3f}, real - null {(d_real - d_null).mean():+.3f}")


if __name__ == "__main__":
    main()
