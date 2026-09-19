# Jester results

The Jester arm of the joke-preference experiment. Plan in `PLAN.md`, data and
the zero-model-call ceilings in `jester-signal.md`. Numbers are on the 7,200
users who rated all 100 jokes, the fixed item split (seed 20260919, 60 train
jokes, 40 test jokes), and tercile labels: each user's 60 train ratings set
two cut points, and every joke gets bad / good / great from those cut points.

Reproduce:

```bash
.venv/bin/joke-pref jester-evaluate --criteria criteria/probe/a-generic.json --name generic-a-generic
.venv/bin/joke-pref jester-evaluate --criteria criteria/seed.json --name generic-seed
.venv/bin/joke-pref jester-evaluate --criteria criteria/jester/crowd.json --name generic-crowd
```

Each run writes `results/jester/<name>/{summary.json,users.csv,jokes.jsonl,criteria.json}`.

## Summary

- Jev with a one-line generic rubric orders held-out jokes for a random
  Jester user at 0.60 concordance, 4.6 points under the crowd mean that
  reads 7,199 other users' ratings (Phase 1).
- A rubric written by GEPA from one user's 60 ratings lifts that user 11
  points over generic when the crowd predicts them badly (40 users, 30 wins
  to 10, p = 0.002), beats the crowd mean by 17 points there, and closes
  about half of the distance to a k-nearest-neighbours filter, with no
  ratings at inference. The same text read for another user is at the
  generic level, and shuffled labels give nothing. For typical users there
  is no gain (Phase 2).
- DeepSeek Flash reading the same generic rubric is 6 to 10 points under
  Jev on Jester at 1.4 to 8 times the latency, and the GEPA rubric does not
  transfer to it (Phase 3).

## Metric

Per user, pairwise concordance on the 40 test jokes: the share of joke pairs
with different tercile labels that the predictor orders as the user did, ties
one half. Same definition as `metric.py`; `jester_eval.concordance_rows` is
its vectorized twin and a test checks they agree. One user (18708) has no
pair with different labels and is left out.

The crowd and kNN columns are recomputed on the same tercile truth. They are
higher than in `jester-signal.md` (0.612 and 0.659), because the tercile
truth drops the pairs whose ratings are close, and those are the pairs the
predictors get wrong most often.

The rubric columns read no ratings. Jev scores each joke once with the
rubric, and the expected level is the prediction for every user. So the
rubric gives one global order of the 40 jokes. The crowd column reads the
ratings of the other 7,199 users on the same test jokes. The kNN column
reads them and the user's own 60 train ratings. Both are ceilings for a
text-only method, not competitors.

## Phase 1: Jev with a generic rubric

Three rubrics, 100 Jev calls each, about 0.5 to 0.7 s per call, no invalid
result.

| Rubric | All users, median | Selected 60, median | Spearman with crowd mean, 100 jokes |
| --- | --- | --- | --- |
| `probe/a-generic.json` (not / somewhat / very funny) | **0.604** | **0.553** | 0.61 |
| `seed.json` (joke-preference seed) | 0.587 | 0.532 | 0.60 |
| `jester/crowd.json` (written from the 10 best and 10 worst train jokes) | 0.599 | 0.537 | 0.65 |
| Crowd, leave-one-out item mean (reads ratings) | 0.650 | 0.524 | 1.00 |
| kNN, k = 80 (reads ratings) | 0.710 | 0.734 | |

The selected 60 are the 40 users with the largest train-only kNN-minus-crowd
gap in `data/jester/user_stats.csv` plus 20 random users (seed 20260919).
The list is in each `summary.json`.

Three readings:

- **Generic text sits 4.6 points under the crowd mean for the typical
  user.** The plain three-word rubric is the best of the three. The rubric
  that names the crowd's taste in words (long story, named roles, reversal
  in the last line) tracks the crowd mean better joke by joke (Spearman
  0.65 against 0.61) but does not order pairs better for a user. The
  joke-preference seed, written for a different item set, is the weakest.
- **On the selected 60, the generic rubric already beats the crowd
  mean** (0.553 against 0.524, 57 % of these users). These users were
  chosen because the crowd predicts them badly, so this is expected, and it
  is the reason the crowd column is not the right reference for Phase 2. The
  reference is kNN at 0.734, 18 points above the generic rubric.
- **The rubric ranks the plain three-word rubric above two more specific
  ones, on 7,200 users.** This is the first Jev result on public data: a
  System One call with a one-line rubric orders held-out jokes for a random
  person at 0.60 concordance, with no ratings, where the ratings-based crowd
  mean gives 0.65 and the best collaborative filter 0.71.

Per-user spread of the a-generic rubric over all users: 10th percentile
0.47, 90th percentile 0.72. Between rubrics, a-generic beats seed for 64 %
of users and the crowd rubric for 54 %.

Phase 2 below tests whether a rubric optimized on a user's 60 train jokes
beats a-generic on that user's 40 test jokes, paired over the 60 users.

## Phase 2: Jev + GEPA per user

For each of the 60 selected users: GEPA on that user's 60 train jokes (12
groups of 5, 9 groups train, 3 groups validation), seed rubric
`probe/a-generic.json`, joint rewrite of the three levels by DeepSeek Flash,
minibatch 6 groups, budget 300 metric calls. The best rubric is then scored
on all 100 jokes once and evaluated for every user. The user's own row is
the personal result. The row of a fixed random partner among the 60 is the
cross-user control: the same text read for a person it was not written
for. Ten users were also run with the labels shuffled inside each train
group (null run). Budget 300 was set after a pilot at 600 ran 75 minutes for
one user; DeepSeek Flash spends 12 to 16 thousand reasoning tokens per
rewrite.

```bash
.venv/bin/joke-pref jester-optimize --shard 0/10 --max-metric-calls 300 --name personal --cache-name jev-cache-p0.sqlite
.venv/bin/joke-pref jester-optimize --user 79 ... --shuffle-labels --max-metric-calls 300 --name null
.venv/bin/python scripts/jester_phase2_report.py
```

Median concordance on the 40 test jokes:

| Users | Crowd | Generic rubric | Personal rubric | Cross-user | kNN |
| --- | --- | --- | --- | --- | --- |
| All 60 | 0.524 | 0.553 | **0.618** | 0.509 | 0.734 |
| 40 atypical (selected on train) | 0.489 | 0.529 | **0.664** | 0.498 | 0.758 |
| 20 random | 0.656 | 0.568 | 0.581 | 0.521 | 0.716 |
| Null run, 10 users, shuffled labels | 0.549 | 0.586 | 0.553 | 0.544 | 0.787 |

Paired differences over users, mean with a 95 % bootstrap interval, wins /
losses and the exact sign test:

| Comparison | Users | Mean | Interval | Wins / losses | p |
| --- | --- | --- | --- | --- | --- |
| personal - generic | all 60 | +0.067 | +0.031 to +0.105 | 38 / 22 | 0.052 |
| personal - generic | 40 atypical | **+0.111** | +0.062 to +0.160 | 30 / 10 | 0.002 |
| personal - generic | 20 random | -0.020 | | 8 / 12 | |
| personal - cross-user partner | all 60 | +0.097 | +0.060 to +0.133 | 42 / 18 | 0.003 |
| cross-user partner - generic | all 60 | -0.030 | -0.064 to +0.007 | 25 / 35 | 0.25 |
| personal - generic, null run | 10 | -0.052 | -0.128 to +0.013 | 4 / 6 | 0.75 |

Against the success test in `PLAN.md`:

- **Personal beats generic on the paired test.** Yes for the 40 atypical
  users (+11 points, 30 wins to 10, p = 0.002). For all 60 the interval
  excludes zero but the sign test is at 0.05, because the 20 random users
  gain nothing (-2 points).
- **Cross-user control is flat.** Yes. A rubric written for user A reads
  user B 3 points under the generic rubric, and the personal rubric beats
  its cross-user twin by 10 points (p = 0.003). The gain is in the text
  written for that person, not in longer text as such. The null run with
  shuffled labels is 5 points under generic.
- **Closes half of the gap to kNN.** Almost, for the atypical users: the
  median user closes 46 % of the distance from generic to kNN, with no
  ratings read at inference. Over all 60 the median is 32 %. Seven users
  reach or pass their kNN ceiling.

Where the signal lives: the personal gain correlates 0.70 with the user's
own kNN-minus-generic gap. The rubric helps exactly the people the crowd
predicts badly. For them it also beats the crowd mean itself, 0.664 against
0.489, which a generic rubric cannot do. For a typical user, whose taste is
the crowd's taste, there is nothing personal to learn and the rubric stays
at the generic level. This matches the Jester signal study: the median
personal gap is small, the tail is where personalization pays.

The learned rubrics are long (median 2,200 characters over the three
levels; three users kept the seed) and name concrete joke features and
short quotes from train jokes: for user 23550 (generic 0.239, personal
0.771) the bad level lists "long story with a flatly practical pay-off" and
the great level lists compact wordplay, the reverse of the crowd's taste.
Cost per user: about 920 Jev calls and 20 DeepSeek rewrites, 21 minutes
wall clock with the reasoning trace. Results: `results/jester/personal/`
(one directory per user with `best_criteria.json`, `users.csv`, the GEPA
run), `results/jester/null/`, and `users.csv` in each.

## Phase 3: DeepSeek Flash as the judge

Same 100 jokes, same rubric `probe/a-generic.json`, DeepSeek Flash reads the
rubric instead of Jev. Two answer formats: one word (bad / good / great) and
an integer 0 to 100 that maps to an expected level. With and without the
model's reasoning trace (`thinking`). Temperature 0, one call per joke.

```bash
.venv/bin/joke-pref judge --jester --criteria criteria/probe/a-generic.json [--answer score] [--thinking]
```

| Reader | Answer | All users, median | Selected 60 | Latency per joke | Tokens per joke (in / out) |
| --- | --- | --- | --- | --- | --- |
| Jev | 3 probabilities | **0.604** | **0.553** | 0.5 s | |
| DeepSeek Flash, no thinking | one word | 0.500 | 0.531 | 0.7 s | 147 / 1 |
| DeepSeek Flash, no thinking | 0 to 100 | 0.512 | 0.551 | 0.9 s | 193 / 1 |
| DeepSeek Flash, thinking | one word | 0.530 | 0.513 | 2.3 s | 173 / 287 |
| DeepSeek Flash, thinking | 0 to 100 | 0.543 | 0.550 | 4.2 s | 219 / 651 |

The one-word judge without thinking says "good" for 76 of the 100 jokes, so
most pairs tie and it sits at chance. The graded answer and the reasoning
trace each help a little. The best DeepSeek variant is 6 points under Jev on
all users, at 8 times the latency and about 650 reasoning tokens per joke.
The 60 selected users are the ones the crowd predicts badly, and there every
reader lands near 0.55.

The same comparison on Naveen's own test split (30 premises, 101
punchlines), where the GEPA rubric was written for Jev:

| Reader | Rubric | Concordance | Top-1 | Reasoning tokens |
| --- | --- | --- | --- | --- |
| Jev | generic | 0.576 | 0.46 | |
| Jev | GEPA run 2 | **0.687** | 0.58 | |
| DeepSeek, no thinking, one word | generic | 0.564 | 0.50 | 0 |
| DeepSeek, no thinking, one word | GEPA run 2 | 0.562 | 0.54 | 0 |
| DeepSeek, no thinking, 0 to 100 | generic | 0.586 | 0.58 | 0 |
| DeepSeek, no thinking, 0 to 100 | GEPA run 2 | 0.595 | 0.58 | 0 |
| DeepSeek, thinking, 0 to 100 | generic | 0.666 | 0.67 | 576 per joke |
| DeepSeek, thinking, 0 to 100 | GEPA run 2 | 0.634 | 0.62 | 2,457 per joke |

Two readings:

- **The GEPA rubric does not transfer to the LLM reader.** It gives
  DeepSeek at most 1 point without thinking and costs it 3 points with
  thinking. The rubric text encodes how Jev reads level descriptions, not a
  portable description of Naveen's taste. Section "Known weaknesses" in
  `STATUS.md` already suspected this.
- **A thinking LLM with a generic rubric reaches 0.666 on Naveen's
  split**, 2 points under Jev with the personal rubric, at about 600
  reasoning tokens per joke against a single Jev call. On Jester the same
  judge is 6 points under Jev with the same generic rubric. So the System
  One tier is not only cheaper here; on Jester it also reads the generic
  rubric better than the LLM does.

Results: `results/jester/judge-*/` and `results/naveen/judge-*/`.
Judge answers are cached in `results/judge-cache.sqlite`.
