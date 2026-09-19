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

What Phase 2 must show: a rubric optimized on a user's 60 train jokes
beats a-generic on that user's 40 test jokes (paired over the 60 users),
the cross-user control does not, and the gain covers a good share of the
18-point distance to kNN on the selected users.
