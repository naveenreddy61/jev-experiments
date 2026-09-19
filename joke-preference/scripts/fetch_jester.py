#!/usr/bin/env python
"""Download and parse the Jester dataset.

Sources:

- ratings: https://goldberg.berkeley.edu/jester-data/jester-data-1.zip
- joke text: https://raw.githubusercontent.com/BerkeleyAutomation/jester/
  master/jokes/jesterjoketext.zip

If a host refuses, the script tries the Internet Archive copy and prints a
note.

Raw downloads go to `data/jester/raw/` (gitignored). The parsed outputs are
`data/jester/jokes.jsonl` and `data/jester/ratings.csv.gz`.

Run: `.venv/bin/python scripts/fetch_jester.py`
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import re
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "jester"
RAW = OUT / "raw"

RATINGS_URL = "https://goldberg.berkeley.edu/jester-data/jester-data-1.zip"
TEXT_URL = (
    "https://raw.githubusercontent.com/BerkeleyAutomation/jester/"
    "master/jokes/jesterjoketext.zip"
)
ARCHIVE = "https://web.archive.org/web/2023/"

NOT_RATED = 99.0
N_JOKES = 100
N_USERS = 24983
GAUGE = {5, 7, 8, 13, 15, 16, 17, 18, 19, 20}

JOKE_RE = re.compile(
    r"<!--\s*begin of joke\s*-->(.*?)<!--\s*end of joke\s*-->",
    re.S | re.I,
)
TAG_RE = re.compile(r"<[^>]+>")


# --- download ---------------------------------------------------------------


def fetch(url: str, dest: Path) -> str:
    """Download `url` to `dest`. Return the URL that worked."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"keep    {dest.name} ({dest.stat().st_size} bytes)")
        return "cached"
    for candidate in (url, ARCHIVE + url):
        try:
            req = urllib.request.Request(
                candidate, headers={"User-Agent": "joke-pref/0.1 (research)"}
            )
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"failed  {candidate}: {exc}", file=sys.stderr)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        if candidate != url:
            print(f"NOTE    the first host refused. Used {candidate}")
        print(f"got     {dest.name} ({len(data)} bytes)")
        return candidate
    raise SystemExit(f"cannot download {url}")


# --- joke text --------------------------------------------------------------


def clean(raw: str) -> str:
    text = raw.replace("<P>", "\n").replace("<p>", "\n")
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    lines = [" ".join(ln.split()) for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln).strip()


def parse_jokes(zip_path: Path) -> list[dict]:
    """Map `init<N>.html` to joke id N. Skip the macOS resource forks."""
    jokes: dict[int, str] = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            base = Path(name).name
            if not base.startswith("init") or not base.endswith(".html"):
                continue
            if "__MACOSX" in name or base.startswith("._"):
                continue
            num = int(base[len("init") : -len(".html")])
            body = JOKE_RE.search(z.read(name).decode("latin-1"))
            if body is None:
                raise SystemExit(f"no joke markers in {name}")
            jokes[num] = clean(body.group(1))
    if sorted(jokes) != list(range(1, N_JOKES + 1)):
        raise SystemExit(f"expected jokes 1..100, got {len(jokes)}")
    return [{"id": i, "text": jokes[i]} for i in range(1, N_JOKES + 1)]


# --- ratings ----------------------------------------------------------------


def read_matrix(zip_path: Path) -> list[list[float]]:
    """Read the rating matrix from the zipped workbook.

    The page says the file is Excel. It is a true OLE2 / BIFF8 workbook, not
    a CSV file with an .xls name, so `xlrd` reads it.
    """
    import xlrd

    with zipfile.ZipFile(zip_path) as z:
        inner = [n for n in z.namelist() if n.lower().endswith(".xls")]
        if not inner:
            raise SystemExit("no .xls member in the ratings zip")
        blob = z.read(inner[0])
    head = blob[:8]
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        print("format  OLE2 workbook (BIFF), not CSV. Parsed with xlrd.")
    else:
        print(f"format  unexpected header {head!r}; trying xlrd anyway")
    book = xlrd.open_workbook(file_contents=blob)
    sheet = book.sheet_by_index(0)
    return [sheet.row_values(i) for i in range(sheet.nrows)]


def check_matrix(rows: list[list[float]]) -> dict:
    """Validate shape, the count column, and the gauge-set fingerprint."""
    if len(rows) != N_USERS:
        raise SystemExit(f"expected {N_USERS} rows, got {len(rows)}")
    if len(rows[0]) != N_JOKES + 1:
        raise SystemExit(f"expected 101 columns, got {len(rows[0])}")
    if not all(isinstance(v, float) for v in rows[0]):
        raise SystemExit("row 0 is not numeric; a header row is present")

    count_ok = 0
    per_joke = [0] * N_JOKES
    for row in rows:
        rated = 0
        for j in range(N_JOKES):
            if row[j + 1] != NOT_RATED:
                rated += 1
                per_joke[j] += 1
        count_ok += int(int(row[0]) == rated)
    order = sorted(range(N_JOKES), key=lambda j: -per_joke[j])
    dense = {j + 1 for j in order[:10]}
    print(f"check   column 1 equals the rated count for {count_ok}/{len(rows)} users")
    print(f"check   ten densest jokes {sorted(dense)}")
    print(f"check   rank 10 has {per_joke[order[9]]} ratings, rank 11 has {per_joke[order[10]]}")
    if dense != GAUGE:
        raise SystemExit(f"gauge fingerprint mismatch: {sorted(dense)}")
    print("check   the dense set equals the published gauge set: column N = joke N")
    return {"count_ok": count_ok, "per_joke": per_joke}


def write_ratings(rows: list[list[float]], dest: Path) -> int:
    n = 0
    with gzip.open(dest, "wt", newline="") as f:
        f.write("user_id,joke_id,rating\n")
        for u, row in enumerate(rows, start=1):
            for j in range(N_JOKES):
                v = row[j + 1]
                if v == NOT_RATED:
                    continue
                f.write(f"{u},{j + 1},{v:.2f}\n")
                n += 1
    return n


# --- main -------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="parse again")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    ratings_zip = RAW / "jester-data-1.zip"
    text_zip = RAW / "jesterjoketext.zip"
    fetch(RATINGS_URL, ratings_zip)
    fetch(TEXT_URL, text_zip)

    jokes_out = OUT / "jokes.jsonl"
    if args.force or not jokes_out.exists():
        jokes = parse_jokes(text_zip)
        with jokes_out.open("w") as f:
            for rec in jokes:
                f.write(json.dumps(rec) + "\n")
        print(f"wrote   {jokes_out.name}: {len(jokes)} jokes")
        for i in (1, 50, 100):
            text = jokes[i - 1]["text"].replace("\n", " ")
            print(f"  joke {i:3d}: {text[:90]}...")
    else:
        print(f"keep    {jokes_out.name}")

    ratings_out = OUT / "ratings.csv.gz"
    if args.force or not ratings_out.exists():
        rows = read_matrix(ratings_zip)
        check_matrix(rows)
        n = write_ratings(rows, ratings_out)
        print(f"wrote   {ratings_out.name}: {n} ratings")
    else:
        print(f"keep    {ratings_out.name}")


if __name__ == "__main__":
    main()
