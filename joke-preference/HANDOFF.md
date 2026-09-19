# HANDOFF: joke-preference

Read this file first. `STATUS.md` holds the detailed state.

## What this project is

An experiment on Jev, the TypeSafe AI "System One" model. The question: can
the text of a `Score` rubric, and nothing else, hold one person's taste well
enough to recommend text items to that person better than a generic rubric?
The optimizer is GEPA (standalone `gepa` package). It rewrites the three level
descriptions of the rubric from that person's labels. DeepSeek Flash writes
the text. Jev reads it. No LLM runs at recommendation time.

The repo is the `jev-experiments` monorepo. This experiment is
`joke-preference/`. Experiment 1 is `parsing-experiments/`. Its write-up is in
`articles/`.

## Result so far, in one paragraph

On Naveen's own labels (412 punchlines, 120 premises, bad/good/great) the
GEPA rubric reaches 0.69 pairwise concordance on 30 held-out premises. A
generic "not funny / somewhat funny / very funny" rubric reaches 0.58, and a
null run with shuffled labels reaches 0.56. So the personal lift is 11 points
of concordance, 13 points of top-1 hit rate. It is real (the null run is
flat) but Naveen's bar is 20 points, and the lift sits in one place only:
bad-vs-good pairs, 0.83 against 0.63. The "great" label does not transfer at
all (good-vs-great pairs 0.11 under every rubric). The rubric fits its own
training data no better than the test data (0.675 vs 0.687), so the limit is
expressivity or label noise, not overfitting. Full numbers in `STATUS.md`.

## Direction decided on 2026-09-19

Naveen's framing: the aim is to find out whether there is signal in the
process, that is, whether an optimized rubric beats a default rubric for one
person by a wide margin. Not variance across seeds. Not small wins.

The next step is to move to public data where many people rated the same
text items, so the same test can run on many users, with controls we cannot
run on one person's labels. Three research reports rank the candidates:

- `docs/research/datasets-recsys.md`
- `docs/research/datasets-humor-shorttext.md`
- `docs/research/datasets-personal-preference.md`

Order of work:

1. **Jester** (100 jokes, 7,200 users rated all 100, continuous -10..+10,
   text available, research-use license with citation). Same domain. A
   complete user-by-item matrix lets us measure, with zero model calls, how
   much of a user's ratings the crowd mean explains and what collaborative
   filtering can reach. That tells us if a 20-point personal gap exists in the
   data at all. A subagent was started on this at the end of the session; see
   "First action".
2. **GPPL humour** (UKPLab, Apache-2.0, 4,030 one-liners, 433 workers with
   100+ pairwise votes). Pairwise, matches our metric. Crowd consensus of four
   raters predicts a fifth only 70.9 % of the time.
3. **DICES-350** (Google, CC BY 4.0, 123 raters x 350 short chatbot turns,
   safe/unsure/unsafe). Non-humor generalization test of "personal judgment".

Controls to add on any multi-user dataset: rubric trained on user A tested on
user B; users chosen for low agreement with the crowd (selected on one half of
their data, evaluated on the other); item-mean predictor as the strongest
generic baseline; shuffled-label null run.

## Jester outcome (2026-09-19, late)

The Jester subagent finished. Read `docs/jester-signal.md`. Headline: for a
typical Jester user the whole personal gap is 4 points of concordance (crowd
item-mean predictor 0.612, k=80 kNN 0.659, median over 7,200 complete
users). Only 1.9 % of users show a gap of 0.20 or more, and only a
train-selected top 40 reach +0.198 at the collaborative-filtering ceiling.
A text-only rubric reads no ratings and must land under that ceiling. So the
20-point bar is not available on Jester. Naveen's own +11 over generic is in
line with what a rating-based CF wins on Jester. Not yet discussed with
Naveen at the time of writing; see the last chat message of the session.

Naveen's reading (same evening): the 20-point bar was hyperbole. The aim is
a meaningful personal signal from Jev + GEPA for text recommendation, a
method that sits between ML models (need ratings) and LLM judges (cost).
On Jester the reference is the kNN ceiling, so "meaningful" = personal
rubric beats generic on a paired test across users, cross-user control
flat, and closes a good share of the gap to kNN, with no ratings read at
inference.

## Phase 1 done (2026-09-20)

`joke-pref jester-evaluate` scores the 100 Jester jokes with one rubric and
reports per-user concordance for all 7,200 users and for the 60 selected
users, next to crowd and kNN on the same tercile labels. Result in
`docs/jester-results.md`: the plain three-word rubric gives 0.604 (all) and
0.553 (selected 60); crowd 0.650 / 0.524; kNN 0.710 / 0.734. Three generic
rubrics tried, the plain one is best.

## First action for the next session

Read `PLAN.md`. Phase 2 is next: Jev + GEPA per user on the 60 selected
users. The pieces exist: `jester.user_items` gives the groups of 5 jokes as
`LabeledPremise`, `runs.optimize` runs GEPA, `jester_eval.evaluate_rubric`
scores a rubric for all users. Add a `jester-optimize` command and a driver
over the 60 users, with the cross-user and shuffled-label controls. Then
Phase 3, DeepSeek Flash as the judge.

## Things not to redo

- The go/no-go probe (does criteria text move Jev at all): passed, mean spread
  1.33 on a 0-2 scale. `results/probe/`.
- GEPA plumbing bugs: fixed and tested (`bbff01e`, `f8c01ff`).
- The label server, the dataset build, the metric, the Jester loaders and
  evaluation: all tested, 51 tests.
- Naveen's own labeling is complete. A relabel pass for an intra-rater ceiling
  was proposed and not yet asked for.
