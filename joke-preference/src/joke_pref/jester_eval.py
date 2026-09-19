"""Score the Jester jokes with one rubric and measure per-user concordance.

Jev scores a joke, not a user. One rubric therefore costs 100 Jev calls for
all users at once. The per-user work is numpy only:

- `tercile_labels` bins each user's ratings into bad / good / great with cut
  points from that user's **train** jokes only.
- `concordance_rows` is the vectorized twin of `joke_pref.metric`: pairwise
  concordance per user over the pairs with different labels, ties one half.
- `loo_item_mean`, `knn_predict`, `eigentaste_predict` are the rating-based
  reference predictors. They read the ratings of other users on the test
  jokes, which a rubric never does, so they are ceilings, not competitors.
- `evaluate_rubric` puts it together for one rubric and one item split.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from joke_pref.criteria import Criteria
from joke_pref.jester import Joke, Ratings, item_split
from joke_pref.scorer import ScoreResult

SEED = 20260919
K_NEIGHBOURS = 80
SCALE = 20.0  # rating range -10..+10, for NMAE
N_LEVELS = 3


# --- metric -----------------------------------------------------------------


def concordance_rows(preds: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Pairwise concordance for each row.

    The definition mirrors `joke_pref.metric.pairs` and `concordance`: a pair
    with equal true values is skipped, and a tie in the prediction counts one
    half. A row with no usable pair gives `nan`.
    """
    d_true = truth[:, :, None] - truth[:, None, :]
    d_pred = preds[:, :, None] - preds[:, None, :]
    n = truth.shape[1]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    usable = (d_true != 0) & upper
    credit = np.where(d_pred == 0, 0.5, (np.sign(d_pred) == np.sign(d_true)).astype(float))
    total = usable.sum(axis=(1, 2)).astype(float)
    got = np.where(usable, credit, 0.0).sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, got / np.where(total == 0, 1, total), np.nan)


def tercile_labels(train: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Level index 0/1/2 for each `target` rating, with cut points at the
    33rd and 67th percentile of the same user's `train` ratings.

    A rating equal to a cut point goes to the level above it. Both arrays
    have one row per user.
    """
    edges = np.quantile(train, [1 / 3, 2 / 3], axis=1).T  # (n_users, 2)
    out = np.empty(target.shape, dtype=np.int64)
    for i in range(target.shape[0]):
        out[i] = np.searchsorted(edges[i], target[i], side="right")
    return out


def bin_by_own_quantiles(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin each user's ratings by that user's own quantiles of the same array."""
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1], axis=1).T
    out = np.empty_like(x, dtype=np.int64)
    for i in range(x.shape[0]):
        out[i] = np.searchsorted(edges[i], x[i], side="right")
    return out


# --- rating-based predictors (ceilings) ------------------------------------


def loo_item_mean(x: np.ndarray) -> np.ndarray:
    """Leave-one-out item mean. `out[u, j]` excludes user u."""
    n = x.shape[0]
    return (x.sum(axis=0)[None, :] - x) / (n - 1)


def knn_predict(
    x: np.ndarray, train: np.ndarray, test: np.ndarray, k: int = K_NEIGHBOURS, *, chunk: int = 512
) -> np.ndarray:
    """User-based kNN. Pearson similarity on the train jokes only.

    The similarity block is built one chunk of users at a time, so the peak
    memory stays near `chunk * n_users` floats.
    """
    tr = x[:, train]
    mu = tr.mean(axis=1, keepdims=True)
    sd = tr.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    z = ((tr - mu) / sd).astype(np.float32)
    te_c = (x[:, test] - mu).astype(np.float32)
    k = min(k, x.shape[0] - 1)

    out = np.empty((x.shape[0], test.size), dtype=np.float64)
    for start in range(0, x.shape[0], chunk):
        stop = min(start + chunk, x.shape[0])
        sim = (z[start:stop] @ z.T) / tr.shape[1]
        sim[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        w = np.take_along_axis(sim, top, axis=1).astype(np.float64)
        neigh = te_c[top].astype(np.float64)  # (chunk, k, n_test)
        denom = np.abs(w).sum(axis=1)
        denom[denom == 0] = 1.0
        out[start:stop] = mu[start:stop] + (w[:, :, None] * neigh).sum(axis=1) / denom[:, None]
        del sim, top, w, neigh
    return out


def eigentaste_predict(
    x: np.ndarray, train: np.ndarray, test: np.ndarray, *, bins: int = 8, floor: int = 20
) -> np.ndarray:
    """Eigentaste style: two principal components of the train block, a grid
    of cells, then the cell mean of each test joke without the user itself."""
    tr = x[:, train]
    tr_c = tr - tr.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(tr_c, full_matrices=False)
    proj = tr_c @ vt[:2].T  # (n, 2)

    cell = np.zeros(x.shape[0], dtype=np.int64)
    for d in range(2):
        edges = np.quantile(proj[:, d], np.linspace(0, 1, bins + 1)[1:-1])
        cell = cell * bins + np.searchsorted(edges, proj[:, d])

    te = x[:, test]
    fallback = loo_item_mean(te)
    out = fallback.copy()
    for c in np.unique(cell):
        rows = np.flatnonzero(cell == c)
        if rows.size <= floor:
            continue
        block = te[rows]
        out[rows] = (block.sum(axis=0)[None, :] - block) / (rows.size - 1)
    return out


# --- user selection -----------------------------------------------------------


def selected_users(
    stats_path: str | Path, *, n_top: int = 40, n_random: int = 20, seed: int = SEED
) -> list[int]:
    """The 60 study users: the top `n_top` by `gap_train_only` in
    `user_stats.csv` (atypical users, chosen on train jokes only) plus
    `n_random` others drawn with a fixed seed. Sorted by user id."""
    with open(stats_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: -float(r["gap_train_only"]))
    top = [int(r["user_id"]) for r in rows[:n_top]]
    rest = [int(r["user_id"]) for r in rows[n_top:]]
    rng = np.random.default_rng(seed)
    extra = [rest[i] for i in rng.choice(len(rest), size=n_random, replace=False)]
    return sorted(top + extra)


# --- rubric evaluation -------------------------------------------------------


@dataclass
class RubricEval:
    user_ids: np.ndarray  # (n_users,)
    test_ids: list[int]  # joke ids in test order
    truth: np.ndarray  # (n_users, n_test) tercile labels 0/1/2
    rubric_conc: np.ndarray  # (n_users,)
    rubric_agreement: np.ndarray  # mean probability on the true level
    rubric_exact: np.ndarray  # exact-level accuracy
    crowd_conc: np.ndarray
    knn_conc: np.ndarray
    n_invalid: int

    def table(self, mask: np.ndarray | None = None) -> dict[str, float]:
        m = np.ones(self.user_ids.shape, dtype=bool) if mask is None else mask
        return {
            "n_users": int(m.sum()),
            "rubric_concordance_median": float(np.nanmedian(self.rubric_conc[m])),
            "rubric_concordance_mean": float(np.nanmean(self.rubric_conc[m])),
            "rubric_agreement_mean": float(np.nanmean(self.rubric_agreement[m])),
            "rubric_exact_accuracy_mean": float(np.nanmean(self.rubric_exact[m])),
            "crowd_concordance_median": float(np.nanmedian(self.crowd_conc[m])),
            "knn_concordance_median": float(np.nanmedian(self.knn_conc[m])),
            "rubric_beats_crowd_share": float(np.nanmean(self.rubric_conc[m] > self.crowd_conc[m])),
        }


def user_mask(ratings: Ratings, user_ids: Sequence[int]) -> np.ndarray:
    wanted = set(int(u) for u in user_ids)
    return np.array([int(u) in wanted for u in ratings.user_ids], dtype=bool)


def evaluate_rubric(
    results: Sequence[ScoreResult],
    jokes: Sequence[Joke],
    ratings: Ratings,
    *,
    seed: int = SEED,
    k: int = K_NEIGHBOURS,
    crowd_pred: np.ndarray | None = None,
    knn_pred: np.ndarray | None = None,
) -> RubricEval:
    """Per-user concordance of one rubric against tercile labels on the test
    jokes, next to the crowd and kNN predictors on the same labels.

    `results[i]` is the Jev result for `jokes[i]`. Pass `crowd_pred` and
    `knn_pred` (shape `(n_users, n_test)`) to reuse them across rubrics.
    """
    if len(results) != len(jokes):
        raise ValueError("one result per joke is needed")
    x = ratings.matrix
    train_ids, test_ids = item_split(seed)
    train = ratings.columns(train_ids)
    test = ratings.columns(test_ids)
    truth = tercile_labels(x[:, train], x[:, test])

    by_id = {j.id: r for j, r in zip(jokes, results)}
    ordered = [by_id[int(jid)] for jid in test_ids]
    n_invalid = sum(1 for r in ordered if not r.valid)
    pred = np.array([r.score if r.score is not None else -1.0 for r in ordered], dtype=float)
    probs = np.array(
        [r.probabilities if r.probabilities is not None else (0.0,) * N_LEVELS for r in ordered],
        dtype=float,
    )  # (n_test, 3)
    argmax = probs.argmax(axis=1)

    rubric_pred = np.tile(pred[None, :], (x.shape[0], 1))
    rubric_conc = concordance_rows(rubric_pred, truth.astype(float))
    agreement = probs[np.arange(len(test_ids))[None, :], truth].mean(axis=1)
    exact = (argmax[None, :] == truth).mean(axis=1)

    if crowd_pred is None:
        crowd_pred = loo_item_mean(x)[:, test]
    if knn_pred is None:
        knn_pred = knn_predict(x, train, test, k)
    return RubricEval(
        user_ids=ratings.user_ids,
        test_ids=list(test_ids),
        truth=truth,
        rubric_conc=rubric_conc,
        rubric_agreement=agreement,
        rubric_exact=exact,
        crowd_conc=concordance_rows(crowd_pred, truth.astype(float)),
        knn_conc=concordance_rows(knn_pred, truth.astype(float)),
        n_invalid=n_invalid,
    )


def write_user_table(path: str | Path, ev: RubricEval) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["user_id", "rubric_concordance", "rubric_agreement", "rubric_exact", "crowd_concordance", "knn_concordance"])
        for i, u in enumerate(ev.user_ids):
            w.writerow(
                [int(u)]
                + [f"{v:.4f}" for v in (ev.rubric_conc[i], ev.rubric_agreement[i], ev.rubric_exact[i], ev.crowd_conc[i], ev.knn_conc[i])]
            )
