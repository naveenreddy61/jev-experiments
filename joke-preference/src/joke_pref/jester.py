"""Loaders for the Jester data in `data/jester/`.

Build the files first:

```bash
.venv/bin/python scripts/fetch_jester.py
```

- `load_jokes()` gives the 100 joke texts.
- `load_ratings()` gives a dense `float` matrix with `nan` for a missing
  rating, plus the user ids for the rows and the joke ids for the columns.
- `complete_users()` gives the row mask for users who rated all 100 jokes.

Nothing here downloads anything.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

N_JOKES = 100
GAUGE_JOKES = (5, 7, 8, 13, 15, 16, 17, 18, 19, 20)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "jester"


@dataclass(frozen=True)
class Joke:
    id: int
    text: str

    @property
    def n_words(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class Ratings:
    """`matrix[i, j]` is the rating of `joke_ids[j]` by `user_ids[i]`."""

    matrix: np.ndarray  # shape (n_users, n_jokes), nan where not rated
    user_ids: np.ndarray  # shape (n_users,)
    joke_ids: np.ndarray  # shape (n_jokes,)

    @property
    def n_users(self) -> int:
        return int(self.matrix.shape[0])

    def column(self, joke_id: int) -> int:
        """Column index of a joke id."""
        hit = np.flatnonzero(self.joke_ids == joke_id)
        if hit.size != 1:
            raise KeyError(f"joke {joke_id} is not in this matrix")
        return int(hit[0])

    def columns(self, joke_ids: Iterable[int]) -> np.ndarray:
        return np.array([self.column(j) for j in joke_ids], dtype=int)


def load_jokes(path: Path | None = None) -> list[Joke]:
    path = path or DATA_DIR / "jokes.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run scripts/fetch_jester.py")
    jokes = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                jokes.append(Joke(id=int(rec["id"]), text=rec["text"]))
    jokes.sort(key=lambda j: j.id)
    return jokes


def load_ratings(path: Path | None = None, *, n_jokes: int = N_JOKES) -> Ratings:
    path = path or DATA_DIR / "ratings.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run scripts/fetch_jester.py")
    opener = gzip.open if path.suffix == ".gz" else open
    users: list[int] = []
    index: dict[int, int] = {}
    rows: list[list[float]] = []
    with opener(path, "rt") as f:  # type: ignore[operator]
        header = f.readline().strip().split(",")
        if header != ["user_id", "joke_id", "rating"]:
            raise ValueError(f"unexpected header {header}")
        for line in f:
            if not line.strip():
                continue
            u_s, j_s, r_s = line.split(",")
            u = int(u_s)
            row = index.get(u)
            if row is None:
                row = len(users)
                index[u] = row
                users.append(u)
                rows.append([float("nan")] * n_jokes)
            rows[row][int(j_s) - 1] = float(r_s)
    matrix = np.array(rows, dtype=float)
    return Ratings(
        matrix=matrix,
        user_ids=np.array(users, dtype=int),
        joke_ids=np.arange(1, n_jokes + 1, dtype=int),
    )


def complete_users(ratings: Ratings) -> Ratings:
    """Keep only the users who rated every joke."""
    keep = ~np.isnan(ratings.matrix).any(axis=1)
    return Ratings(
        matrix=ratings.matrix[keep],
        user_ids=ratings.user_ids[keep],
        joke_ids=ratings.joke_ids,
    )


def item_split(
    seed: int, *, n_train: int = 50, gauge: Iterable[int] = GAUGE_JOKES
) -> tuple[list[int], list[int]]:
    """The fixed item split. The gauge jokes are always in train.

    Returns `(train_joke_ids, test_joke_ids)`, both sorted.
    """
    gauge = tuple(gauge)
    rest = np.array([j for j in range(1, N_JOKES + 1) if j not in gauge])
    order = np.random.default_rng(seed).permutation(rest.size)
    shuffled = rest[order]
    train = sorted(list(gauge) + shuffled[:n_train].tolist())
    test = sorted(shuffled[n_train:].tolist())
    return train, test
