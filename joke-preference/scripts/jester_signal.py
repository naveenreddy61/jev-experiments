#!/usr/bin/env python
"""Signal diagnostics on Jester. No model calls.

How much of joke taste is shared, and how much of the personal part is
learnable? The script answers with the ratings alone.

Run: `.venv/bin/python scripts/jester_signal.py`

It prints every table that `docs/jester-signal.md` reports, and it writes
`data/jester/user_stats.csv`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from joke_pref.jester import (  # noqa: E402
    GAUGE_JOKES,
    complete_users,
    item_split,
    load_jokes,
    load_ratings,
)
from joke_pref.metric import concordance as ref_concordance  # noqa: E402
from joke_pref.metric import pairs as ref_pairs  # noqa: E402

SEED = 20260919
K_NEIGHBOURS = 80
SCALE = 20.0  # the rating range, -10..+10, for NMAE
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


# The metric and the predictors live in `joke_pref.jester_eval` now.
from joke_pref.jester_eval import (  # noqa: E402
    bin_by_own_quantiles,
    concordance_rows,
    eigentaste_predict,
    knn_predict,
    loo_item_mean,
)


def check_metric_against_reference(preds: np.ndarray, truth: np.ndarray) -> None:
    """Confirm the vectorized metric equals the one in `joke_pref.metric`."""
    fast = concordance_rows(preds, truth)
    ids = [str(i) for i in range(truth.shape[1])]
    for row in (0, 1, 7, 100, 1000):
        slow = ref_concordance(ref_pairs(preds[row].tolist(), truth[row].tolist(), ids))
        assert slow is not None
        assert abs(slow - fast[row]) < 1e-9, (row, slow, fast[row])
    print("check   vectorized concordance matches joke_pref.metric on 5 users")


def quantiles(x: np.ndarray) -> dict[str, float]:
    x = x[~np.isnan(x)]
    out = {f"p{int(q * 100):02d}": float(np.quantile(x, q)) for q in QUANTILES}
    out["mean"] = float(x.mean())
    return out


def show(name: str, x: np.ndarray) -> None:
    q = quantiles(x)
    cells = "  ".join(f"{k}={v:+.3f}" for k, v in q.items())
    print(f"{name:<34s} {cells}")


def pair_supply(truth: np.ndarray) -> float:
    """Mean share of item pairs that the truth does not tie."""
    n = truth.shape[1]
    total = n * (n - 1) / 2
    d = truth[:, :, None] - truth[:, None, :]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(((d != 0) & upper).sum(axis=(1, 2)).mean() / total)


# --- main -------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--k", type=int, default=K_NEIGHBOURS)
    args = ap.parse_args()

    jokes = load_jokes()
    full = load_ratings()
    data = complete_users(full)
    x = data.matrix
    print(f"data    {full.n_users} users, {data.n_users} rated all 100 jokes")

    train_ids, test_ids = item_split(args.seed)
    train = data.columns(train_ids)
    test = data.columns(test_ids)
    print(f"split   seed {args.seed}: {len(train_ids)} train, {len(test_ids)} test")
    print(f"        gauge always in train: {list(GAUGE_JOKES)}")
    print(f"        test jokes: {test_ids}")

    truth = x[:, test]

    # (a) crowd agreement
    print("\n== (a) crowd agreement, all 100 jokes ==")
    loo = loo_item_mean(x)
    a_c = x - x.mean(axis=1, keepdims=True)
    b_c = loo - loo.mean(axis=1, keepdims=True)
    denom = np.sqrt((a_c**2).sum(axis=1) * (b_c**2).sum(axis=1))
    denom[denom == 0] = np.nan
    crowd_r = (a_c * b_c).sum(axis=1) / denom
    show("pearson r vs LOO item mean", crowd_r)
    show("r^2 (variance explained)", crowd_r**2)
    print(f"mean r^2 = {float(np.nanmean(crowd_r**2)):.3f}; "
          f"share of users with r < 0.2: {float(np.nanmean(crowd_r < 0.2)):.3f}")

    # (b) crowd concordance on the test jokes
    print("\n== (b) pairwise concordance, crowd predictor, 40 test jokes ==")
    crowd_pred = loo[:, test]
    check_metric_against_reference(crowd_pred[:2000], truth[:2000])
    crowd_conc = concordance_rows(crowd_pred, truth)
    show("crowd concordance", crowd_conc)
    crowd_nmae = np.abs(crowd_pred - truth).mean(axis=1) / SCALE
    show("crowd NMAE", crowd_nmae)

    # (c) collaborative filtering
    print("\n== (c) collaborative filtering ceiling ==")
    knn_pred = knn_predict(x, train, test, args.k)
    knn_conc = concordance_rows(knn_pred, truth)
    knn_nmae = np.abs(knn_pred - truth).mean(axis=1) / SCALE
    show(f"kNN k={args.k} concordance", knn_conc)
    show(f"kNN k={args.k} NMAE", knn_nmae)

    eig_pred = eigentaste_predict(x, train, test)
    eig_conc = concordance_rows(eig_pred, truth)
    eig_nmae = np.abs(eig_pred - truth).mean(axis=1) / SCALE
    show("eigentaste concordance", eig_conc)
    show("eigentaste NMAE", eig_nmae)

    own = np.tile(x[:, train].mean(axis=1, keepdims=True), (1, len(test)))
    show("user-mean NMAE", np.abs(own - truth).mean(axis=1) / SCALE)
    print("published (different split): POP NMAE 0.203, Eigentaste NMAE 0.187, "
          "80-NN NMAE 0.187")

    # (d) where the personal gap is
    print("\n== (d) personal gap ==")
    gap = knn_conc - crowd_conc
    show("kNN minus crowd concordance", gap)
    for t in (0.20, 0.10, 0.05, 0.0):
        print(f"users with gap >= {t:.2f}: {int((gap >= t).sum())} "
              f"({float((gap >= t).mean()):.3f})")
    lo = crowd_r < np.nanquantile(crowd_r, 0.10)
    print(f"bottom-decile crowd r (r < {float(np.nanquantile(crowd_r, 0.10)):.3f}): "
          f"n={int(lo.sum())}, median crowd concordance "
          f"{float(np.nanmedian(crowd_conc[lo])):.3f}, median kNN concordance "
          f"{float(np.nanmedian(knn_conc[lo])):.3f}, median gap "
          f"{float(np.nanmedian(gap[lo])):.3f}")
    # is a large gap a property of the user, or noise in 40 items?
    half = np.random.default_rng(args.seed + 1).permutation(len(test))
    a, b = np.sort(half[:20]), np.sort(half[20:])
    gaps = []
    for part in (a, b):
        gaps.append(
            concordance_rows(knn_pred[:, part], truth[:, part])
            - concordance_rows(crowd_pred[:, part], truth[:, part])
        )
    ga, gb = gaps
    ok = ~(np.isnan(ga) | np.isnan(gb))
    r_half = float(np.corrcoef(ga[ok], gb[ok])[0, 1])
    pick = np.argsort(-ga)[:137]
    print(f"split-half check: correlation of the gap between two 20-joke halves "
          f"r={r_half:.3f}")
    print(f"the top 137 users by half A have median gap {float(np.nanmedian(ga[pick])):+.3f} "
          f"on half A and {float(np.nanmedian(gb[pick])):+.3f} on half B")

    # an honest selection statistic: the gap inside the 60 train jokes only
    rng = np.random.default_rng(args.seed + 2)
    perm = rng.permutation(train.size)
    half_a, half_b = np.sort(train[perm[:30]]), np.sort(train[perm[30:]])
    parts = []
    for fit, held in ((half_a, half_b), (half_b, half_a)):
        p_knn = knn_predict(x, fit, held, args.k)
        t = x[:, held]
        parts.append(
            concordance_rows(p_knn, t) - concordance_rows(loo[:, held], t)
        )
    gap_train = np.nanmean(np.vstack(parts), axis=0)
    ok2 = ~(np.isnan(gap_train) | np.isnan(gap))
    print(f"train-only gap: median {float(np.nanmedian(gap_train)):+.3f}; "
          f"correlation with the test gap r="
          f"{float(np.corrcoef(gap_train[ok2], gap[ok2])[0, 1]):.3f}")
    sel = np.argsort(-gap_train)[:200]
    print(f"top 200 users by the train-only gap: median test gap "
          f"{float(np.nanmedian(gap[sel])):+.3f}, median crowd concordance "
          f"{float(np.nanmedian(crowd_conc[sel])):.3f}, median kNN concordance "
          f"{float(np.nanmedian(knn_conc[sel])):.3f}")

    med = float(np.nanmedian(crowd_conc))
    print(f"typical user: crowd {med:.3f} -> a rubric needs {med + 0.20:.3f}")
    atyp = float(np.nanmedian(crowd_conc[lo]))
    print(f"atypical user: crowd {atyp:.3f} -> a rubric needs {atyp + 0.20:.3f}")

    # (e) item facts
    print("\n== (e) item facts ==")
    item_mean = x.mean(axis=0)
    item_var = x.var(axis=0)
    words = np.array([j.n_words for j in jokes])
    print(f"item mean: min {item_mean.min():+.2f} (joke {int(item_mean.argmin()) + 1}), "
          f"max {item_mean.max():+.2f} (joke {int(item_mean.argmax()) + 1})")
    print(f"item variance: min {item_var.min():.2f}, max {item_var.max():.2f}")
    print(f"words: min {words.min()}, median {int(np.median(words))}, max {words.max()}")
    print("ten highest-variance jokes (id, var, mean, words, in test):")
    for j in np.argsort(-item_var)[:10]:
        jid = int(j) + 1
        print(f"  {jid:3d}  var {item_var[j]:6.2f}  mean {item_mean[j]:+5.2f}  "
              f"{int(words[j]):3d} words  test={jid in test_ids}  "
              f"| {jokes[j].text.replace(chr(10), ' ')[:60]}")

    # (f) three-level and five-level labels
    print("\n== (f) binned labels ==")
    for name, n_bins in (("tercile", 3), ("quintile", 5)):
        binned = bin_by_own_quantiles(x, n_bins)[:, test]
        counts = np.bincount(binned.ravel(), minlength=n_bins) / binned.size
        conc = concordance_rows(crowd_pred, binned.astype(float))
        knn_b = concordance_rows(knn_pred, binned.astype(float))
        print(f"{name}: class balance on the test jokes "
              f"{[round(float(c), 3) for c in counts]}")
        print(f"{name}: usable pair share {pair_supply(binned.astype(float)):.3f}")
        show(f"{name} crowd concordance", conc)
        show(f"{name} kNN concordance", knn_b)
        print(f"{name}: median kNN minus crowd "
              f"{float(np.nanmedian(knn_b - conc)):+.3f}")

    # user stats file
    out = ROOT / "data" / "jester" / "user_stats.csv"
    order = np.argsort(-gap)
    with out.open("w") as f:
        f.write(
            "user_id,crowd_r,crowd_concordance,knn_concordance,eigentaste_concordance,"
            "gap_knn_minus_crowd,gap_train_only,crowd_nmae,knn_nmae,train_mean,train_std\n"
        )
        tr_mean = x[:, train].mean(axis=1)
        tr_std = x[:, train].std(axis=1)
        for i in order:
            f.write(
                f"{int(data.user_ids[i])},{crowd_r[i]:.4f},{crowd_conc[i]:.4f},"
                f"{knn_conc[i]:.4f},{eig_conc[i]:.4f},{gap[i]:.4f},{gap_train[i]:.4f},"
                f"{crowd_nmae[i]:.4f},{knn_nmae[i]:.4f},"
                f"{tr_mean[i]:.3f},{tr_std[i]:.3f}\n"
            )
    print(f"\nwrote   {out} ({data.n_users} rows, sorted by gap)")


if __name__ == "__main__":
    main()
