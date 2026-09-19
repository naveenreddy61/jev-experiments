# Extended hard set — results

The v1 hard set did not separate the models enough. This adds 160 items built
to hold the fault far from the construct it breaks, then re-runs both models on
hard items only.

**The extension worked, but not the way it was meant to.** It did not make the
task harder for the strong baseline. It exposed that Jev was near its limit
already. The gap between the models more than doubled.

> **Single-run figures.** Every DeepSeek number on this page comes from one
> call per item. Three repeats of the hard set score 0.9187, 0.9313 and
> 0.9437, so the 0.950 below is the top of the range rather than the centre.
> Read DeepSeek on this set as 0.93 ± 0.01. See
> [run-variance.md](run-variance.md).
>
> **Jev's 0.544 is also a single run and has no error bar.** Jev was not
> repeated, so the two columns below are not equally well measured: the
> DeepSeek figure is a mean of three runs and the Jev figure is one call per
> item. The gap conclusion does not depend on this — 0.386 against the v1
> set's 0.176 is not a close call — but do not read Jev's number as the better
> established of the two. It is the less established one.

| hard set | items | Jev | DeepSeek | gap |
|---|---|---|---|---|
| v1 seed hard | 74 | 0.743 | 0.919 | 0.176 |
| **new** | 160 | **0.544** | **0.950** | **0.406** |
| combined | 234 | 0.607 | 0.940 | 0.333 |

Jev falls to 0.544, which is near the 0.500 chance line for a balanced set.
DeepSeek rises slightly. Neither model produced an invalid output.

The two columns are not scored by the same rule. Jev's boolean comes from
`p_yes >= 0.5`, and DeepSeek reports its own boolean. Scoring DeepSeek by the
same threshold gives **0.944** on the new set instead of 0.950. The comparison
below uses each model's native output, and the difference is small enough that
it changes no conclusion here.

Per task on the combined 234 items:

| task | Jev | DeepSeek |
|---|---|---|
| `c.compile` | 0.662 | 0.985 |
| `bash.parse` | 0.649 | 0.877 |
| `python.parse` | 0.582 | 1.000 |
| `latex.parse` | 0.526 | 0.895 |

Calibration on the combined set: Jev Brier 0.259 and ECE 0.254; DeepSeek Brier
0.061 and ECE 0.055.

## What changed in the data

| | v1 hard | new hard |
|---|---|---|
| items | 74 | 160 |
| median lines per artifact | 8 | 21 |
| median characters | 154 | 566 |
| distinct fault types | 21 | 65 |

Four Opus subagents authored candidates, one per task. They wrote source only.
Every label comes from the oracle through `scripts/build_hard.py`, which is the
suite's rule. Candidates carry an author intent flag, but that flag never
becomes a label — it only drops authoring mistakes, so a "fault" that does not
actually break parsing cannot enter the set. All 160 items pass an independent
`validate` pass against `bash -n`, `gcc -std=c11 -fsyntax-only`, `ast.parse`
and `pdflatex`. That pass was run here, over the committed files, and does not
rely on the authoring agents' own reports of their work.

Half of each task's items are valid source that looks broken. That tests the
opposite bias: PEP 695 generics, C11 flexible array members with anonymous
unions, LaTeX `\verb` forms holding braces.

## Jev does not detect faults in long artifacts

Recall on broken artifacts is the number that moved.

| hard set | Jev | DeepSeek |
|---|---|---|
| v1 seed hard | 0.596 | 0.872 |
| new hard | **0.150** | 0.925 |

On the new items Jev accepts 68 of 80 broken artifacts. It answers yes to 143
of 160 items when only 80 are valid. On `latex.parse` it answered **yes to all
40 of 40**, so its 0.500 there is exactly chance and carries no information.

Its mean probability on broken items is 0.712, so it is confidently wrong, not
uncertain. Recall on valid artifacts stays high at 0.938. Jev has not become
noisy; it has stopped detecting faults.

This is consistent with the v1 finding. Jev catches local, token-level faults
and misses non-local, block-structure faults. The new items are longer and put
more distance between the opener and the fault, and that is the axis Jev fails
along.

## DeepSeek's failures

DeepSeek made 8 errors on the new items. Six are false positives. Every fault
type it missed, with how many items of that type exist and how many Jev also
missed:

| fault type | items | Jev missed | DeepSeek missed |
|---|---|---|---|
| `bad_double_bracket` | 2 | 2 | **2** |
| `do_done_mismatch` | 4 | 3 | 1 |
| `if_fi_mismatch` | 2 | 2 | 1 |
| `missing_package_env` | 1 | 1 | 1 |
| `undefined_control_sequence` | 1 | 1 | 1 |
| `unmatched_dollar` | 1 | 1 | 1 |

`bad_double_bracket` defeats both models completely across both hard sets: 3 of
3 on the v1 set and 2 of 2 here, so 5 of 5 in total for each model. It is the
only fault type with a perfect miss record on both sides, and the only one that
still separates nothing because neither model can do it.

Two are new: DeepSeek produced its **first false negatives**, rejecting valid
source. Its errors are therefore no longer a strict subset of Jev's, which they
were on the v1 seed set.

## DeepSeek contradicts itself; Jev cannot

On 3 of 160 new items DeepSeek returned a boolean that disagrees with its own
probability:

```json
{"answer": false, "p_yes": 0.95}
```

The harness accepts these, because both fields are well typed. The contract
cannot catch a self-contradiction. The same check finds 1 such row in the
earlier 200-item run, so the rate rises with artifact difficulty.

Scoring DeepSeek by `p_yes >= 0.5`, the same rule applied to Jev, gives 0.944
instead of the 0.950 from its self-reported boolean.

A Noul answer cannot fail this way. It is one number, so there is no second
field to disagree with. That is a real benefit of typed output, separate from
accuracy.

## How to repeat it

```bash
python -m jev_eval validate data/hard
python -m jev_eval run --provider jev --data data/hard --out results/jev-hard-new
python -m jev_eval run --provider deepseek-flash --data data/hard \
    --out results/deepseek-hard-new
# v1 hard items only, from the seed set
python -m jev_eval run --provider jev --data data/seed --difficulty hard \
    --out results/jev-hard-v1
```

The v1 hard numbers above were taken from the earlier full 200-item runs by
selecting the 74 hard ids, so they cost no new calls.

The combined 234-item figures (Jev 0.607, DeepSeek 0.940, and the Brier and ECE
values) come from concatenating the two prediction files per model and scoring
the result:

```bash
# per model: the 74 hard rows from the full run, plus the 160 new rows
python - <<'PY'
import json
hard = {json.loads(l)["id"]
        for t in ("bash.parse", "c.compile", "latex.parse", "python.parse")
        for l in open(f"data/seed/{t}.jsonl")
        if json.loads(l)["difficulty"] == "hard"}
rows = [l for l in open("results/jev-full-200/predictions.jsonl")
        if json.loads(l)["id"] in hard]
rows += open("results/jev-hard-new/predictions.jsonl").readlines()
open("results/jev-hard-all.jsonl", "w").writelines(rows)
PY
python -m jev_eval score results/jev-hard-all.jsonl
```

`results/` is gitignored, so these files are local only. Re-run the two runs
above to rebuild them.

## Limits

- One run per model per set. No variance estimate.
- The new set did not raise the ceiling: DeepSeek scores 0.950, above its 0.919
  on v1 hard. The set now separates the two models well, but it still does not
  measure the top of DeepSeek's range.
- Jev at 0.544 on a balanced set is close to chance. Below about 0.55 the score
  stops ranking anything, so further difficulty on this axis will not produce
  more information about Jev.
- The items were authored by one model family. A different author would
  probably find different faults.
- `latex.parse` labels come from TeX Live 2026 on macOS. The local oracles
  reproduce all 200 v1 seed labels with zero mismatches, so the toolchain
  agrees with the one that built the seed.

## Next steps

1. Jev's useful range on this suite is valid-source recognition, not fault
   detection. Measure that separately rather than as one accuracy number.
2. Build a graded distance axis — the same fault at 5, 15 and 40 lines of
   separation — to find where Jev's detection actually breaks, instead of only
   showing that it does.
3. Test whether `criteria` with `true` and `false` descriptions recover any of
   the lost recall. Keep it a separate run: the v1 question is locked.
4. To measure the top of DeepSeek's range, the faults need a different axis
   than length. `bad_double_bracket` is the one that still works.
