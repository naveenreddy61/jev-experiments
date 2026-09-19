# Evaluation — Jev vs DeepSeek Flash, full seed set

`jev-1.13.0` and `deepseek-flash` on all 200 v1 seed items, one run each.

| | `jev-1.13.0` | `deepseek-flash` |
|---|---|---|
| accuracy | 0.785 | **0.970** |
| invalid rate | 0.0 | 0.0 |
| Brier | 0.151 | **0.034** |
| ECE | 0.146 | **0.036** |
| mean confidence | 0.824 | 0.986 |
| latency mean | **375 ms** | 2662 ms |
| latency p50 | **342 ms** | 1635 ms |
| latency p95 | **462 ms** | 6361 ms |

Both models answered the same 200 items in the same order. Neither produced an
invalid output. Jev is about 7 times faster on mean latency and about 14 times
faster at p95.

| task | `jev` | `deepseek-flash` |
|---|---|---|
| `python.parse` | 0.90 | 1.00 |
| `c.compile` | 0.84 | 1.00 |
| `bash.parse` | 0.72 | 0.94 |
| `latex.parse` | 0.68 | 0.94 |

- Date: 2026-09-18
- Threshold: `YES_THRESHOLD = 0.5` for Jev; DeepSeek reports its own boolean
- Raw runs: `results/jev-full-200/`, `results/deepseek-full-200/` (gitignored)

## How to repeat it

```bash
pip install -e '.[jev]'
export JEV_API_KEY=... DEEPSEEK_API_KEY=...
python -m jev_eval run --provider jev --data data/seed --out results/jev-full-200
python -m jev_eval run --provider deepseek-flash --data data/seed \
    --out results/deepseek-full-200
```

## Correction to the earlier 40-item check

An earlier bounded run used 40 items (`scripts/make_subset.py --per-label 5`).
DeepSeek scored 40 of 40 there, and that run concluded the seed set could not
rank strong models. **The full set disproves that.** DeepSeek scores 0.970 on
200 items and makes 6 errors, all of them informative. The 40-item sample
simply missed the hard items that discriminate.

Jev moved very little between the two runs: 0.800 on 40, 0.785 on 200. The
subset was representative for Jev and misleading for DeepSeek, because a
perfect score has no resolution below it.

## Both models fail the same way: they say yes too often

| | Jev | DeepSeek |
|---|---|---|
| gold-yes recall | 0.960 | 1.000 |
| gold-no recall | 0.610 | 0.940 |
| errors | 43 | 6 |
| false positives | 39 of 43 (91%) | 6 of 6 (100%) |

Almost every error is a false positive: the model reported that a broken
artifact parses. DeepSeek made no false negatives at all. Jev made 4, and all
of them are easy `latex.parse` items, so the exception is narrow and does not
change the picture.

**DeepSeek's errors are a strict subset of Jev's.** All 6 items DeepSeek missed
were also missed by Jev, and DeepSeek made no error that Jev got right.

## The shared frontier

DeepSeek's 6 failures are all `hard` items, and they concentrate on three fault
types:

| fault type | items | Jev missed | DeepSeek missed |
|---|---|---|---|
| `bad_double_bracket` | 3 | 3 | **3** |
| `math_mode_trap` | 3 | 3 | 2 |
| `frac_arity` | 3 | 3 | 1 |

`bad_double_bracket` defeats both models completely, on all 3 of 3 items. That
is a frontier identified, not sized: three items cannot measure how wide it is.
These are the items the suite exists to find, and the set needs more of them.

Accuracy by difficulty:

| difficulty | items | Jev | DeepSeek |
|---|---|---|---|
| easy | 55 | 0.82 | 1.00 |
| medium | 71 | 0.80 | 1.00 |
| hard | 74 | 0.74 | 0.92 |

DeepSeek is perfect on easy and medium items. All of its resolution comes from
the hard set, which is the part of the suite worth growing.

## Jev's blind spot is non-local structure

Jev's miss rate by fault type separates cleanly. Of 31 fault types, 7 are
missed on every item and 12 are caught on every item. A sample of each:

| Jev misses every time | Jev always catches |
|---|---|
| `then_missing` (4 of 4) | `missing_semicolon` (0 of 3) |
| `do_done_mismatch` (3 of 3) | `undeclared_ident` (0 of 3) |
| `unclosed_env` (3 of 3) | `unclosed_macro` (0 of 3) |
| `bad_double_bracket` (3 of 3) | `bad_fstring` (0 of 3) |
| `bad_preprocessor` (3 of 3) | `unmatched_paren` (0 of 4) |
| `math_mode_trap` (3 of 3) | `unmatched_bracket` (0 of 4) |
| `frac_arity` (3 of 3) | `redirect_typo` (0 of 3) |

Jev catches **local, token-level** faults: a missing semicolon, an unclosed
string, an unbalanced bracket. It misses **non-local, block-structure** faults:
a missing `then`, a `do` without `done`, an unclosed environment, a macro with
the wrong arity.

On the missed group it is not merely unsure. Mean `p_yes` per fault type runs
0.67 to 0.92, so it is confidently wrong. The caught group runs 0.05 to 0.40.
The two groups do not overlap.

This is a capability statement, not a bug. Deciding whether a block closes
needs a count held across distance, which is what a single fast judgment is
least suited to.

## Threshold

Jev's boolean uses `YES_THRESHOLD = 0.5`. Re-thresholding the same 200
predictions costs no new calls:

| threshold | accuracy |
|---|---|
| 0.5 | 0.785 |
| 0.6 | 0.820 |
| 0.7 | 0.835 |
| **0.75** | **0.860** |
| 0.8 | 0.855 |
| 0.9 | 0.730 |

0.75 gives +0.075 over the default, and the earlier 40-item run peaked nearby
at 0.7. The gain is real but it is fitted on the same data it is measured on.
Hold out a split before you change the default.

## Limits

- One run per model. There is no repeat measurement and no variance estimate,
  so small per-task gaps are not reliable.
- DeepSeek's 6 errors sit in three fault types with 3 items each. The frontier
  is identified, but its width is not measured. Generate more `hard` items of
  those types before drawing conclusions about them.
- The two models differ in kind. DeepSeek self-reports a boolean and a
  probability from generated text. Jev returns a typed noul thresholded here
  at 0.5.
- The LaTeX labels come from a pinned `pdflatex`. The seed `oracle_meta` shows
  the labels were made on Linux, and the oracle was not re-run here.

## Next steps

1. Grow the `hard` set around `bad_double_bracket`, `math_mode_trap` and
   `frac_arity`. That is where both models fail and where the suite has
   resolution.
2. Add block-structure faults at longer distances, to map Jev's limit rather
   than only detect it.
3. Test whether `criteria` with `true` and `false` descriptions reduce Jev's
   false positives. Keep it as a separate run: the v1 question is locked, and
   changing it changes what is measured.
4. Choose a Jev threshold on a held-out split if 0.5 is not wanted.
