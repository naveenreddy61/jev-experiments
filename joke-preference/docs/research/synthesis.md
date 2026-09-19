# Synthesis: a personal joke recommender from Jev criteria and GEPA

Date: 2026-09-19. This document joins the three research reports in this
folder into one design proposal. The scaffold, dataset, labeling page and
GEPA loop now exist; see the project README for the current state.

Reports:

- [jev-api.md](jev-api.md) — the Jev SDK and API, verified against
  `typesafe_sdk` 0.7.0 source.
- [dspy-gepa.md](dspy-gepa.md) — DSPy 3.3.1, GEPA, and the standalone `gepa`
  0.1.4 package.
- [parsing-experiments-report.md](parsing-experiments-report.md) — how
  experiment 1 called Jev and DeepSeek, and what it can lend.

## The question

Can a System One model become a personal recommender when the only thing we
train is the text of its `criteria`? The training signal is one person's
labels. The optimizer is GEPA. The scorer is Jev. No LLM runs at inference
time.

## Six facts that shape the design

1. **Jev has three primitives, and one of them is a graded rubric.** `Noul`
   returns one probability. `Choice` picks a label. `Score` takes an ordered
   list of level descriptions and returns an expected score, a confidence,
   and a probability per level. The TypeSafe docs say a graded judgment
   belongs in `Score`, not `Noul`. Bad / good / great is a three-level rubric.
   `Score` fits it directly.

2. **`criteria` is a field on each question.** For `Noul` it is an optional
   `{"true": ..., "false": ...}` pair. For `Score` it is a required list, one
   entry per level. Each entry can be a string, a dict, or a list. There is no
   documented length limit below the 64k-token context.

3. **One call has one `state` but many questions.** The state can be a string
   or a dict. Several questions ride together with no extra latency. So one
   call can hold one premise and one `Score` question per punchline. That is
   a comparative scoring mode that a single-joke call does not have.

4. **Jev is not deterministic.** The docs' own repeat test shows a standard
   deviation near 0.01 on `noul`, and spreads that cross 0.5 on subjective
   questions. Repeat calls and average at test time.

5. **Experiment 1 found Jev confidently wrong, not uncertain.** Brier 0.259 and
   ECE 0.254 on hard parsing items. It said yes to 278 of 360 items. Treat the
   raw probability as a ranking signal first and a calibrated value second.

6. **`dspy.GEPA` cannot wrap this directly.** It optimizes text inside a DSPy
   program whose predictors call an LM. Jev is not an LM. The standalone
   `gepa` package, which DSPy uses underneath, accepts any text component and
   a custom adapter. Its `evaluate()` can call Jev. Its
   `make_reflective_dataset()` gives the reflection LM the joke, the label,
   the Jev output, and a diagnosis. Only the reflection LM is an LLM, and it
   runs a few dozen times per optimization, never at inference.

## Cost and time

| item | value |
|---|---|
| one full pass over ~420 punchlines | about $0.02 |
| Jev rate limit | about 1,200 requests/min |
| one full pass, parallel | about 20 s |
| GEPA run, 30 iterations, valset of 20 premises | about 2,000 Jev calls, a few minutes |
| reflection LM calls per run | a few dozen |

Cost is not a constraint. Naveen's labeling time is the constraint.

## Proposed design

### The unit of data is a premise

A record holds one premise and 2 to 5 punchlines. Each punchline carries one
label: `bad`, `good`, or `great`. Split by premise, never by punchline, so a
punchline in the test set never shares a premise with training data.

| split | premises | use |
|---|---|---|
| train | 70 | GEPA reflection minibatches |
| val | 20 | GEPA Pareto selection |
| test | 30 | held out, scored once at the end |

Target: 120 premises, about 420 punchlines. Generate the punchlines with an
LLM in deliberately different styles (pun, absurdist, dark, anti-joke, meta,
observational, dad joke). Style is a latent feature that criteria text can
name. Record the style tag but hide it from Jev.

### Label reliability sets the ceiling

Relabel 20% of the punchlines about a week later. Intra-rater agreement is the
upper bound for any model. If Naveen agrees with himself on 70% of items, a
model at 70% is perfect, not weak. Report this number with every result.

### Two scoring modes

- **Mode A, single joke.** State is one full joke (premise plus one
  punchline). One `Score` question with three criteria entries.
- **Mode B, comparative.** State is the premise plus all its punchlines as a
  dict. One `Score` question per punchline, same criteria on each. Jev sees the
  alternatives. This is the recommendation shape.

Run `Noul` with `{"true", "false"}` criteria as a secondary arm, so the
result also speaks to the parsing experiment, which used `Noul` only.

### What GEPA optimizes

The candidate is a dict of text components. For `Score`: `level_bad`,
`level_good`, `level_great`, and `instructions`. For `Noul`: `true`, `false`,
`instructions`. GEPA can update one component at a time or all together.

### The metric

Per-example score for GEPA, where an example is one premise:

- **Within-premise ranking.** Fraction of punchline pairs with different labels
  that Jev orders the same way Naveen did. This is the recommender goal.
- **Level agreement.** Mean probability that Jev assigns to the true level.
  This is a proper scoring signal that keeps the levels meaningful.
- Blend about 0.6 ranking and 0.4 level agreement.

Side metrics logged once per full pass: Spearman between expected score and
label, top-1 hit rate per premise (did the top-scored punchline carry
Naveen's best label), exact-level accuracy, Brier and ECE reused from
`parsing-experiments/src/jev_eval/metrics.py`.

The feedback text for each example names the premise, each punchline, the
label, the Jev probabilities, and a one-sentence diagnosis such as "Jev rated
the pun above the absurdist line; Naveen did the opposite."

### The arms that make the result mean something

| arm | what it shows |
|---|---|
| Jev, no criteria, "is this joke funny?" | generic funniness baseline |
| Jev, Naveen's hand-written seed criteria | the starting point for GEPA |
| Jev, GEPA criteria | the claim under test |
| DeepSeek Flash with the seed criteria | an LLM judge at the same start |
| DeepSeek Flash with the GEPA criteria | does the criteria text transfer |
| label-frequency baseline | chance level |

If the GEPA arm beats the generic arm, the criteria carried personal taste,
not general humor. If DeepSeek with the GEPA criteria also improves, the
criteria text is a portable description of taste. If it does not, the gain
is tied to Jev's reading of the text.

### Go/no-go probe before any labeling

Run 20 jokes under five very different criteria texts and measure how much
the Jev output moves. If criteria barely changes the probabilities, GEPA has
nothing to steer, and the experiment stops there with a clear negative
result. This costs 100 calls and ten minutes.

## Proposed layout

```text
joke-preference/
  README.md
  pyproject.toml          # deps: typesafe-sdk, gepa, dspy (reflection LM only)
  data/
    premises.jsonl        # premise, punchlines with style tags
    labels.jsonl          # Naveen's labels, one row per punchline
    splits.json           # premise ids per split, fixed seed
  src/joke_pref/
    schema.py             # Premise, Punchline, Label
    jev_scorer.py         # Score and Noul calls, modes A and B, repeat/average
    adapter.py            # GEPAAdapter for gepa.optimize
    metric.py             # ranking + level agreement, side metrics
    label_ui.py           # local labeling tool
    cli.py                # probe, label, optimize, evaluate
  tests/                  # fake typesafe_sdk namespace, copied from experiment 1
  results/                # gitignored
  docs/
    research/             # these reports
```

## Reuse from experiment 1

Verbatim: `metrics.py` functions (`brier_score`, `expected_calibration_error`,
`score_rows`), the Jev client construction and error policy in
`runners/jev.py`, the fake-SDK test harness, the `results/` gitignore, and
the commit style.

New: everything about jokes, labels, `Score`, `criteria`, and GEPA.

## Open decisions for Naveen

1. `Score` as the primary primitive, `Noul` as the secondary arm. Agree?
2. LLM-generated punchlines in tagged styles, or a curated human source?
3. Labeling tool: terminal prompt, or a local web page with three buttons?
4. Commit the personal labels to the public repo, or keep `data/labels.jsonl`
   local and commit only the aggregate results?
5. Reflection LM for GEPA: Claude Sonnet 5 through DSPy, or DeepSeek to keep
   one vendor in the write-up?
