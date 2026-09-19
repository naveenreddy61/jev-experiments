# Run-to-run variance, DeepSeek on the hard set

The suite made one call per item until now, so every number it reported was a
single sample with no error bar. Removing the pinned `temperature: 0` made that
gap urgent, because the harness now takes the provider default.

Three passes of `deepseek-flash` over the same 160 hard items, default sampling,
nothing else changed.

**Headline: a single run on this set carries about ±0.025 of accuracy noise, and
all of it comes from 11 items. The previously documented 0.950 was the top of
the range, not the centre.**

## Accuracy moves by 2.5 points across identical runs

| run | accuracy | invalid |
|---|---|---|
| `ds-def-r1` | 0.9437 | 0 |
| `ds-def-r2` | 0.9313 | 0 |
| `ds-def-r3` | 0.9187 | 0 |

Range 0.0250, standard deviation 0.0102. Mean 0.9313.

For comparison, the two earlier runs that sent `temperature: 0` scored 0.9500
and 0.9625. Both sit at or above the top of the default range. That hints the
pinned setting helped, but two runs cannot establish it and the sets of runs
are not otherwise matched. Treat it as a question, not a result.

What it does settle: **0.950 should never have been quoted as DeepSeek's score
on this set.** The honest statement is 0.93 ± 0.01.

## The spread comes from 11 items, and only those

149 of 160 items (93.1%) give the same answer in all three runs. Accuracy on
those 149 is **0.9597 in every run — identical to four decimal places**. Every
point of the spread is carried by the 11 items that changed.

Of the 11: 8 broken, 3 valid. None is answered correctly in all three runs by
definition; 7 are right twice and 4 are right once.

Their fault tags are all different — `quote_imbalance`, `bad_arithmetic`,
`bad_double_bracket`, `bad_preprocessor`, `designated_index_range`,
`math_mode_trap`, `frac_arity`, `newcommand_redefine`, and three valid items.
Instability is not a property of one fault type. It is a property of items the
model finds hard.

## Instability tracks reasoning cost — unlike error

| | mean reasoning tokens | median |
|---|---|---|
| 149 stable items | 1451 | 716 |
| 11 unstable items | 3573 | 3314 |

A 4.6x gap at the median. The two earlier temperature-0 runs show the same
shape on their 4 disagreements: 3249 against 1668.

This sharpens the earlier finding rather than contradicting it. Reasoning
length does **not** predict whether an answer is wrong (r = +0.004). It does
predict whether the answer is *stable*. A long chain means the model is near a
decision boundary, and which side it lands on varies between calls.

That is the more useful signal of the two, because it is available without
knowing the right answer. Reasoning length can flag an item as unreliable at
inference time.

## Per-item token counts are unstable; the aggregate is not

| measure | value |
|---|---|
| median per-item coefficient of variation | 0.370 |
| CV quartiles | 0.256 / 0.370 / 0.510 |
| total reasoning tokens per run | 242,793 / 272,379 / 251,475 |
| spread of the total | 11.6% of the mean |

A single item's reasoning-token count is worth little: the typical item's count
moves by 37% between calls. The distribution over 160 items is stable to about
12%. Read the shape, never one item.

The clearest casualty is the "worst single item" from the earlier write-up,
`python.parse.h2.00036`, reported at 21,386 tokens and correct. Across these
three runs it costs 1329, 3961 and 4682 tokens and **flips its answer** —
`True`, `False`, `True` — on valid source.

## Majority vote helps a little

Majority-of-3 scores **0.9375**: above the mean single run (0.9313), below the
best single run (0.9437). Three calls buy about half a point over one call on
average, at three times the cost. On this set that is a poor trade. It would
matter more if the 11 unstable items were a larger share.

## Corrections to `docs/reasoning-tokens.md`

That document reports one temperature-0 run. Three repeats confirm its main
argument and weaken three specific numbers:

| claim | one run | three repeats |
|---|---|---|
| top 5% share of the budget | 42% | 32%, 35%, 34% |
| valid vs broken median | 540 vs 890 (clear inversion) | 733/728, 653/684, 558/733 — near equal, inconsistent sign |
| valid vs broken mean | 2073 vs 1342 | 1928/1107, 2128/1277, 1756/1388 — inversion holds 3 of 3 |

So the **tail** claim survives: valid source has the heavier tail and the higher
mean in every run, and the extreme maxima are valid (26135, 26349, 16929). The
**median** claim does not. Typical valid and typical broken items cost about the
same; the difference is entirely in the tail.

The difficulty ordering and the length-is-not-the-driver result were not
re-tested here, because these runs cover only the hard set.

## How to repeat it

```bash
for i in 1 2 3; do
  python -m jev_eval run --provider deepseek-flash \
      --data data/hard --out "results/ds-def-r$i"
done
python scripts/analyze_variance.py \
    --pred results/ds-def-r1 results/ds-def-r2 results/ds-def-r3 \
    --data data/hard
```

## Limits

- Three runs. The standard deviation of 0.0102 is itself estimated from three
  numbers and is not precise.
- One model, one item set. Jev was not re-run; its path sends no sampling
  parameter and never did, but that does not make it deterministic.
- The temperature-0 comparison uses two runs against three, and the runs were
  not made under otherwise matched conditions. It is not a controlled test of
  the temperature change.
