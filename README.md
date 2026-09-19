# jev-experiments

Experiments on TypeSafe AI's **Jev**, a System One model, measured against a
reasoning model on the same items, the same contract and the same metrics.

## The short version

We asked two models one machine-checkable question 360 times — *does this
parse?* — across Bash, C, Python and LaTeX. Every label comes from a real
parser or compiler. Nothing was hand-labelled.

| difficulty | items | Jev | DeepSeek Flash |
|---|---|---|---|
| easy | 55 | 0.818 | 1.000 |
| medium | 71 | 0.803 | 1.000 |
| hard, short | 74 | 0.743 | 0.919 |
| hard, long | 160 | **0.544** | 0.963 |

The hard-long set is balanced 80/80, so **chance is 0.500**. On the LaTeX part
of it Jev scored exactly 0.500.

The pattern: a System One model answers what the artifact makes obvious. When
the answer needs state tracked across distance — a brace opened 40 lines
earlier, a macro redefined at the top — it falls to chance, and it says yes
with high confidence while doing so. Jev answered yes to 278 of 360 items when
only 180 were valid.

Read the write-up: **[Does This Even Parse?](articles/system-one-vs-reasoning.md)**

## Contents

| path | what it is |
|---|---|
| [`articles/`](articles/) | Write-ups for a general audience. |
| [`jev-testing/`](jev-testing/) | Experiment 1: the eval harness, the datasets, the oracles, and every raw run. |

## Reproducing

Everything in the article is recomputable from this repo. The raw predictions
are committed, including DeepSeek's full reasoning traces.

```bash
cd jev-testing
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[jev]'

# Re-score the committed runs — no API keys, no network
python scripts/counts.py
python scripts/analyze_reasoning.py \
    --pred results/ds-tok-seed results/ds-tok-hard \
    --data data/seed data/hard
python scripts/analyze_variance.py \
    --pred results/ds-def-r1 results/ds-def-r2 results/ds-def-r3 \
    --data data/hard
```

Making new calls needs `JEV_API_KEY` or `DEEPSEEK_API_KEY` in the environment,
and the oracle tools (`bash`, `gcc`, `pdflatex`, Python 3.12) for regenerating
or revalidating data. See [`jev-testing/README.md`](jev-testing/README.md).

## Caveats worth reading before quoting any number

- **Jev's figures are a single run per item and have no error bar.** DeepSeek's
  hard-set figure is a mean of three runs that scored 0.9187, 0.9313 and
  0.9437, so a single number on this project is worth about ±0.025. Jev is the
  *less* well established of the two columns, not the more.
- **Jev was asked one locked question with no `criteria`.** Supplying criteria
  is a different experiment and may well go better for it.
- Half the hard set is valid code written to look broken. That is a deliberate
  stress test, not a sample of code in the wild.
- The labels are only as good as the local toolchain. Versions are recorded in
  each item's `oracle_meta`.

## Results in detail

- [Evaluation, full 200-item seed set](jev-testing/docs/evaluation.md)
- [Extended hard set](jev-testing/docs/hard-set-v2.md)
- [DeepSeek reasoning tokens](jev-testing/docs/reasoning-tokens.md)
- [Run-to-run variance](jev-testing/docs/run-variance.md)
