# DeepSeek reasoning tokens — what drives the thinking

`deepseek-flash` is a reasoning model. The harness used to discard the API
`usage` object, so this cost was invisible. It is now recorded per item:
`reasoning_tokens`, the prompt and completion counts, and the reasoning text.

This run covers all 360 items (200 seed + 160 new hard), one call each.

> **Repeats exist now.** Three default-temperature repeats of the hard set are
> reported in [run-variance.md](run-variance.md). They confirm the main
> argument of this document and weaken three numbers in it, each flagged below.
> The most important one: a single run on the hard set carries about ±0.025 of
> accuracy noise.
>
> **Provenance.** These runs were made while the harness still sent
> `temperature: 0`. The harness no longer sends any sampling parameter; it
> takes each provider's default, so it measures the model as shipped. The
> numbers below therefore describe a pinned-temperature configuration and are
> not reproducible with the current code. Re-run before quoting them as
> current.

**Headline: reasoning length tracks difficulty, not artifact size. It does not
predict errors.**

- Accuracy: 0.970 on the seed set, 0.963 on the new hard set.
- 348,335 reasoning tokens across 360 items, which is **98% of all output
  tokens**. The visible answer is about 2%.
- Median 380 tokens, mean 968. The top 5% of items consume **42%** of the
  total. Three repeats on the hard set give 32–35%, so read this as "roughly a
  third", not as 42%.

## Reasoning cost rises with difficulty

| group | items | median lines | median tokens |
|---|---|---|---|
| easy | 55 | 4 | 160 |
| medium | 71 | 9 | 167 |
| v1 hard | 74 | 9 | 366 |
| new hard | 160 | 22 | 817 |

Median cost rises about 5x from easy to the new hard items. Easy and medium are
indistinguishable, so the suite's easy/medium split does not separate anything
for this model — which matches its 1.00 accuracy on both.

## It is difficulty, not length

The naive correlation between artifact length and reasoning tokens is `+0.308`.
That is a between-group artifact: hard items happen to be both longer and
harder. **Within** each group the correlation disappears:

| group | r(artifact lines, reasoning tokens) |
|---|---|
| easy | −0.100 |
| medium | −0.051 |
| v1 hard | −0.137 |
| new hard | +0.112 |

All four are near zero, and three are negative. Doubling the length of a
snippet does not buy more thinking. Semantic difficulty does. Tokens per line
do not scale either: 70, 32, 58 and 70 (mean over mean) or 40, 19, 41 and 37
(median over median). Neither basis shows a trend.

## Valid code is bimodal; broken code has a lower ceiling

Splitting the 360 items by how much thinking they got:

| slice | valid | broken |
|---|---|---|
| 36 least-reasoned | 30 (83%) | 6 |
| 36 most-reasoned | 22 (61%) | 14 |

Base rate is 50/50. The bottom end is decisive: 83% valid, far outside chance.
The top end at 61% is **not** distinguishable from chance on 36 items, so the
claim does not rest on it.

What does carry the top end is the tail. Valid items have the *higher* mean
(2073 against 1342) and 7 of them exceed 5,000 reasoning tokens against 4
broken ones. Typical valid code is cheap; a minority of it is extremely
expensive.

> **Corrected by the repeats.** This paragraph first claimed a median
> inversion too: valid 540 against broken 890. That does not replicate. Three
> repeats give 733/728, 653/684 and 558/733, which is near-equal and does not
> keep a consistent sign. The *mean* inversion holds in all three. Typical
> valid and typical broken items cost about the same; the difference is
> entirely in the tail.

The reading: most valid code is recognised almost instantly, and the
catastrophic cases are almost all valid too. The model spends enormous effort
convincing itself that strange legal code is legal — PEP 695 generics, C11
flexible array members with anonymous unions, LaTeX `\verb` forms holding
braces, all items this suite added deliberately.

The part that replicates most strongly is the **ceiling**, not the middle.
Across the three repeats the heaviest valid item costs 26135, 26349 and 16929
tokens, while the heaviest broken item costs 4837, 6159 and 8794. Broken code
never reaches the extreme; valid code does, in every run. Since the medians are
near-equal, the honest summary is that the two classes cost about the same
typically and differ only in how far the worst case runs.

The worst single item is `python.parse.h2.00036`: 21 lines, valid, **21,386
reasoning tokens**. It answered correctly. Six of the eight heaviest items are
valid source, and all eight are correct.

> **Corrected by the repeats.** That item is not stable. Across three repeats
> it costs 1329, 3961 and 4682 tokens and flips its answer — true, false, true
> — on valid source. A per-item token count is a single sample with a typical
> coefficient of variation of 0.370. Read the distribution, never one item.

## Reasoning length does not predict errors

| reasoning-token quartile | range | error rate |
|---|---|---|
| Q1 | 16–142 | 0.022 |
| Q2 | 143–376 | 0.022 |
| Q3 | 383–884 | 0.044 |
| Q4 | 887–21386 | 0.044 |

`r(reasoning tokens, is wrong) = +0.004`. Accuracy on the 36 heaviest items is
0.944 against 0.967 overall — two extra errors, which is noise at this size.

**But it does predict instability.** The repeats show that items which change
answer between runs cost 3573 reasoning tokens on average against 1451 for
those that do not — a 4.6x gap at the median. Long reasoning does not mean the
model is wrong. It means the model is near a decision boundary, and which side
it lands on varies between calls. That signal is available without knowing the
answer, so it is the more useful of the two. See
[run-variance.md](run-variance.md).

**A correction worth recording.** On the 200-item seed set alone the picture
looked very different: wrong answers averaged 1347 tokens against 346 for
correct ones, and `r(tokens, is wrong)` was `+0.329`. Adding the 160 hard items
erased it. That earlier signal rested on 6 errors and a few outliers; even
there the *medians* were 295 against 204, far less dramatic than the means.
Long thinking does not mean the model is in trouble.

## Which faults cost the most thought

| fault type | items | mean tokens | missed |
|---|---|---|---|
| `subshell_paren_mismatch` | 2 | 3295 | 0 |
| `brace_mismatch` | 5 | 2997 | 0 |
| `missing_paren` | 4 | 2488 | 0 |
| `bad_preprocessor` | 10 | 2064 | 1 |
| `math_mode_trap` | 6 | 1751 | 3 |
| `bad_double_bracket` | 5 | 1386 | **4** |

Only two types combine heavy cost with a poor hit rate: `math_mode_trap` and
`bad_double_bracket`.

`bad_double_bracket` needs care, because **this run disagrees with the earlier
one**. The earlier run (`results/deepseek`) missed all 5. This run misses 4 and
catches `bash.parse.h2.00025`. Both runs sent `temperature: 0`, so the two
answers differ under identical request parameters. Whether the provider honours
that parameter on a reasoning model is not established here, so read this as an
observed disagreement, not as proof about sampling. Either way the practical
rule holds: the counts above are from the `ds-tok-*` run only, and any
single-item claim about this tag should name its run.

The flip is informative. The four misses cost 105, 142, 245 and 1147 reasoning
tokens; the one catch cost **5289**. The model does not weigh this fault and
reject it — it usually does not see it at all, and answers fast. It only
succeeds when it happens to think hard.

## Latency is an excellent proxy

`r(reasoning tokens, latency) = +0.997`. The answer JSON is a fixed short
shape, so wall time is almost entirely thinking time. This is a direct
measurement and does not depend on the sampling settings, so earlier runs that
recorded only latency can be read as reasoning-cost data after the fact.

## Contrast with Jev

Jev has no reasoning stage. It answers in about 375 ms with a fixed small
output, against DeepSeek's mean 968 reasoning tokens and roughly 2.7 s. That
difference buys DeepSeek 0.94 against Jev's 0.61 on hard items.

DeepSeek is also the better-calibrated of the two on this run: Brier 0.0416
and ECE 0.0335 across the 360 items, against Jev's ECE of 0.254 on hard items.
Its mean confidence is 0.989 at 0.967 accuracy, so it is mildly overconfident,
but far less so than Jev.

The cost is not uniform. Half of DeepSeek's reasoning budget goes to a small
minority of items, and on the hardest fault type it spends heavily and still
fails. A cascade is the obvious shape: Jev first, escalate only what it is
unsure about. This data does not yet say where that threshold belongs — Jev's
calibration on hard items is the blocker, so that needs measuring before it is
built.

## How to repeat it

```bash
python -m jev_eval run --provider deepseek-flash --data data/seed --out results/ds-tok-seed
python -m jev_eval run --provider deepseek-flash --data data/hard --out results/ds-tok-hard
python scripts/analyze_reasoning.py \
    --pred results/ds-tok-seed results/ds-tok-hard \
    --data data/seed data/hard
```

## Limits

- One run per item, with `temperature: 0` sent. No repeat, so per-item token
  counts have an unknown spread. The distribution shape is robust; a single
  item's count is not. The harness now sends no sampling parameter, so a repeat
  run is not a like-for-like comparison with these numbers.
- 12 errors in 360 items. Any claim about what the model gets *wrong* rests on
  a very small sample, which is exactly what the seed-only correction above
  shows.
- The heavy tail is driven by items this project authored to look broken while
  being legal. That is a deliberate property of the set, not a neutral sample
  of real code.
- `reasoning_tokens` is the provider's own count and was not independently
  checked.
