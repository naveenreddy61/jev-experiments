# Jester data

Ratings and joke text for the signal study. Build the files with:

```bash
.venv/bin/python scripts/fetch_jester.py
```

## Files

| File | Content |
| --- | --- |
| `raw/jester-data-1.zip` | the downloaded ratings archive (gitignored) |
| `raw/jesterjoketext.zip` | the downloaded joke text archive (gitignored) |
| `jokes.jsonl` | one record for each joke: `{"id": 1..100, "text": ...}` |
| `ratings.csv.gz` | `user_id,joke_id,rating`. Only real ratings. No 99 values |
| `user_stats.csv` | per-user diagnostics, written by `scripts/jester_signal.py` |

`user_id` is the 1-based row index in the source matrix. `joke_id` is the
1-based rating column, which is source column index + 1.

## Sources

- Ratings: <https://goldberg.berkeley.edu/jester-data/jester-data-1.zip>.
  Both downloads came direct from the hosts on 2026-09-19. The Internet
  Archive fallback was not necessary.
- Joke text:
  <https://raw.githubusercontent.com/BerkeleyAutomation/jester/master/jokes/jesterjoketext.zip>

## File format correction

The Berkeley page says the unzipped file is "in Excel (.xls) format", and our
task note said the file is CSV-like with an `.xls` name. That note is wrong.
The file is a true OLE2 compound document with a BIFF8 workbook inside (magic
bytes `d0 cf 11 e0`, 15.3 MB). `scripts/fetch_jester.py` checks the magic
bytes and reads the workbook with `xlrd` 2.x, which supports `.xls` only.

## License, verbatim

From <https://goldberg.berkeley.edu/jester-data/>:

> Freely available for research use when acknowledged with the following
> reference:
>
> Eigentaste: A Constant Time Collaborative Filtering Algorithm. Ken Goldberg,
> Theresa Roeder, Dhruv Gupta, and Chris Perkins. Information Retrieval, 4(2),
> 133-151. July 2001.

> As a courtesy, if you use the data, I would appreciate knowing your name,
> what research group you are in, and the publications that may result.

This is not an open-source license. Research use only. Do not redistribute
the raw files. `raw/` is gitignored for that reason.

The joke text repository `BerkeleyAutomation/jester` has no `LICENSE` file
(HTTP 404 on 2026-09-19). Treat the joke text under the same Berkeley terms.
A clean redistributable copy of the text exists in the R package
`recommenderlab` (GPL-2).

## Required citation

Ken Goldberg, Theresa Roeder, Dhruv Gupta, and Chris Perkins. "Eigentaste: A
Constant Time Collaborative Filtering Algorithm." *Information Retrieval*,
4(2), 133-151. July 2001.

## How the `init<N>.html` to column N mapping was checked

Three checks. Each one tells a different part of the story.

1. **Offset, from the gauge set.** The Eigentaste paper names
   {5, 7, 8, 13, 15, 16, 17, 18, 19, 20} as the dense "universal query" set.
   We counted non-99 values in each rating column. The ten densest columns
   are exactly those ten joke ids. Rank 10 (joke 16) has 24,975 ratings and
   rank 11 (joke 50) has 24,972, so the set is tight but the ranking is
   unambiguous. An off-by-one would give {4, 6, 7, 12, 14...19} or
   {6, 8, 9, 14, 16...21}. This pins the column offset.
2. **Hand check on three jokes, from the file's own title.** Many text files
   state their own number in the `<TITLE>` tag.
   - `init1.html`: `<TITLE>Joke 1 of 25</TITLE>`
   - `init5.html`: `<TITLE>Joke 5</TITLE>`
   - `init13.html`: `<TITLE>Joke 13 of 20</TITLE>`
   In all three the stated number equals the file name number. Across all
   100 files, 40 carry a title of this form and all 40 agree.
3. **Structure.** The matrix has exactly 24,983 rows and 101 columns. Row 0
   holds numbers, not a header. For all 24,983 users, column 1 equals the
   count of non-99 values in the remaining 100 columns.

Limit: checks 1 and 3 fix the offset and the parse. Check 2 ties the text to
the index for 40 jokes. For the other 60 jokes, the mapping still rests on
the file name convention plus the gauge-set fingerprint.
