# Jester signal study

Reproduce with:

```bash
.venv/bin/python scripts/fetch_jester.py
.venv/bin/python scripts/jester_signal.py
```

Data: `jester-data-1`, the 7,200 users who rated all 100 jokes. No model calls.
Seed 20260919. Item split fixed for every user: the ten gauge jokes
{5, 7, 8, 13, 15, 16, 17, 18, 19, 20} plus 50 more in train, 40 in test.

## Result

**No for a typical user. Yes, but only at the ceiling, for a rare selected
user.** The median user holds a 4-point personal gap. A careful selection of
40 users from 7,200 holds a 20-point gap, but that 20 points is what a
collaborative filter wins, and a text-only rubric cannot reach it.

The numbers, all on the 40 test jokes, all per user, median over 7,200 users:

| Predictor | Concordance | NMAE |
| --- | --- | --- |
| Crowd (leave-one-out item mean) | 0.612 | 0.196 |
| Eigentaste style (2 components, 64 cells) | 0.623 | 0.160 |
| kNN, k = 80 | **0.659** | **0.148** |

The crowd predictor is a strong proxy for the **ceiling** of any generic
rubric. (It is not the exact ceiling: a rubric gives one global order, and
the concordance-optimal global order is by majority pairwise preference, not
by item mean. The difference is small.) The kNN predictor is the **ceiling**
of a personal model that already reads 60 of the user's own ratings. The
distance between the two is the whole personal gap the data holds:

- median gap **+0.041**, that is 4 points, not 20;
- 15.4 percent of users reach a gap of 0.10 or more;
- 1.9 percent of users (137 of 7,200) reach a gap of 0.20 or more;
- for 17.6 percent of users the gap is negative.

A text-only rubric reads no ratings, so it must land below the kNN ceiling.
For the median user the 20-point bar asks the rubric to beat a 4-point
ceiling.

Selection helps. Pick the users with the largest gap measured **inside the
train jokes only** (an honest selection, independent of the test items):

| Selected group | Median crowd conc | Median kNN conc | Median test gap |
| --- | --- | --- | --- |
| top 40 of 7,200 (0.6 %) | 0.496 | 0.691 | **+0.198** |
| top 200 of 7,200 (2.8 %) | 0.539 | 0.696 | +0.157 |

So the 20-point gap does exist, but only at the collaborative-filtering
ceiling and only for about the top 0.6 percent of users. A text-only rubric
must stay under that ceiling, because it reads no ratings. Read the top-40
number as "the most a perfect personal model could win here", not as a
target for Jev.

## (a) Crowd agreement

Per user, Pearson correlation between their 100 ratings and the
leave-one-out item mean over the other 7,199 complete users.

| | p05 | p10 | p25 | p50 | p75 | p90 | p95 | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r | 0.050 | 0.121 | 0.237 | 0.358 | 0.466 | 0.548 | 0.595 | 0.344 |
| r^2 | 0.006 | 0.017 | 0.057 | 0.128 | 0.218 | 0.301 | 0.354 | 0.147 |

The crowd explains **14.7 percent** of the average user's rating variance,
and 19.2 percent of users have r below 0.2. Most of the variance is personal
or noise. Sections (c) and (d) show how little of it a model recovers.

## (b) Crowd concordance on the test jokes

Pairwise concordance, defined as in `src/joke_pref/metric.py`: a pair with
equal user ratings is skipped, a tie in the prediction counts one half. The
vectorized version in the script is checked against `metric.pairs` and
`metric.concordance` on five users at every run.

| | p05 | p10 | p25 | p50 | p75 | p90 | p95 | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crowd concordance | 0.475 | 0.508 | 0.560 | 0.612 | 0.663 | 0.701 | 0.723 | 0.608 |
| crowd NMAE | 0.101 | 0.118 | 0.152 | 0.196 | 0.248 | 0.298 | 0.329 | 0.203 |

A perfect generic rubric reaches about 0.61 for the median user. Chance is
0.50. For 5 percent of users even a perfect generic rubric stays below 0.475.

## (c) Collaborative filtering ceiling

kNN: Pearson similarity on the 60 train jokes, k = 80, mean-centred weighted
average of the neighbours. Eigentaste style: two principal components of the
train block, an 8 by 8 quantile grid, then the cell mean of the test joke
without the user itself.

| Predictor | conc p25 | conc p50 | conc p75 | NMAE p50 | NMAE mean |
| --- | --- | --- | --- | --- | --- |
| crowd | 0.560 | 0.612 | 0.663 | 0.196 | 0.203 |
| user mean only | — | — | — | 0.180 | 0.184 |
| Eigentaste style | 0.574 | 0.623 | 0.671 | 0.160 | 0.167 |
| kNN k = 80 | 0.610 | 0.659 | 0.706 | 0.148 | 0.155 |

Published numbers: POP 0.203 and Eigentaste 0.187. **The split differs.** The
published run uses about 2.5 million ratings from 57,000 users of all
densities and a different item holdout. Our run uses only the 7,200 complete
users and a fixed 60/40 item split, which is easier. Our crowd NMAE of 0.203
lands on the published POP number, which is a good sign for the parse, but it
is a coincidence of two protocols. Do not call it a reproduction.

## (d) Where the personal gap is largest

Gap = kNN concordance minus crowd concordance, per user, on the 40 test jokes.

| | p05 | p10 | p25 | p50 | p75 | p90 | p95 | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gap | -0.031 | -0.015 | 0.010 | 0.041 | 0.078 | 0.121 | 0.154 | 0.048 |

| Threshold | Users | Share |
| --- | --- | --- |
| gap >= 0.20 | 137 | 1.9 % |
| gap >= 0.10 | 1,111 | 15.4 % |
| gap >= 0.05 | 3,080 | 42.8 % |
| gap >= 0.00 | 5,931 | 82.4 % |

Users with a low crowd correlation are the better target. Bottom decile by
crowd r (r < 0.121, n = 720): median crowd concordance 0.513, median kNN
concordance 0.615, median gap **+0.094**. Their generic ceiling is 10 points
lower, so a rubric has more room, but only 9 points are learnable.

**What 20 points would demand.** A typical user needs 0.612 + 0.20 = **0.812**.
An atypical user (bottom decile by crowd r) needs 0.513 + 0.20 = **0.713**.
Compare those with the kNN ceilings of 0.659 and 0.615. For both groups the
20-point target is **above** what a model with full access to the user's own
ratings achieves.

**Is a large gap real or noise?** Two checks say it is mostly noise at this
item count.

1. Split the 40 test jokes into two halves of 20. The per-user gap on one
   half correlates with the gap on the other at r = 0.362. The top 137 users
   by half A show a median gap of +0.286 on half A but only +0.174 on half B.
2. Measure the gap inside the 60 train jokes (30 fit, 30 held, both ways).
   Median train gap +0.030. Correlation with the test gap r = 0.478.

So about half of a large observed gap is regression to the mean. Select users
on the train jokes, never on the test jokes.

`data/jester/user_stats.csv` holds one row per complete user, sorted by the
test gap: `user_id, crowd_r, crowd_concordance, knn_concordance,
eigentaste_concordance, gap_knn_minus_crowd, gap_train_only, crowd_nmae,
knn_nmae, train_mean, train_std`. Use `gap_train_only` to select.

## (e) Item facts

Across the 7,200 complete users:

- item mean from -3.52 (joke 58) to +3.93 (joke 50);
- item variance from 16.86 to 34.07, so no joke is close to unanimous;
- text length 12 to 218 words, median 41.

The ten highest-variance jokes, the ones where taste divides:

| Joke | Variance | Mean | Words | In test | Topic |
| --- | --- | --- | --- | --- | --- |
| 71 | 34.07 | -0.55 | 62 | no | DOS `Format C:` |
| 80 | 30.90 | +1.15 | 72 | yes | Clintons and the Pope |
| 2 | 30.23 | +0.71 | 71 | no | paedophilia punchline |
| 75 | 29.99 | -0.13 | 24 | yes | men are stupid |
| 41 | 29.83 | +0.09 | 12 | yes | atheist orgasm |
| 7 | 29.72 | -0.13 | 15 | no | feminist light bulb |
| 51 | 29.02 | -0.43 | 16 | no | Clinton, sex |
| 57 | 28.88 | -1.57 | 15 | no | phone book, weak pun |
| 60 | 28.80 | +0.01 | 15 | no | Buddhist hot dog |
| 98 | 28.73 | +0.95 | 94 | yes | women rated by age |

**Warning on item quality.** The jokes are from 1999 to 2003. The
highest-variance set is dated, offensive, or both: US politics of that decade
(80, 51), sexist material (75, 98), a paedophilia punchline (2), a feminist
joke (7), and an obsolete computer command (71). These items divide taste, so
they carry the most signal, but they measure tolerance of the subject more
than a sense of humour, and a modern scoring model can refuse or flatten
them. Watch the invalid-answer rate on jokes 2, 75, 98 and 41.

## (f) Three-level and five-level labels

Each user's 100 ratings binned by that user's own terciles or quintiles, then
measured on the 40 test jokes.

| Binning | Class balance | Usable pairs | Crowd conc p50 | kNN conc p50 | Median gap |
| --- | --- | --- | --- | --- | --- |
| tercile | 0.298 / 0.334 / 0.368 | 66.9 % | 0.645 | 0.706 | +0.052 |
| quintile | 0.173 / 0.195 / 0.202 / 0.210 / 0.220 | 80.5 % | 0.632 | 0.688 | +0.048 |

Binning does not create a personal gap. It raises both numbers by about 3
points, because the coarse label drops the near-tie pairs that are hard for
every predictor, and it destroys one third (tercile) or one fifth (quintile)
of the pairs. The balance is near even, so the levels are usable as they are.
Three levels match the current rubric; five levels keep more pairs.

## Recommended protocol for the Jev plus GEPA run

1. **Users.** Rank the 7,200 complete users by `gap_train_only` in
   `data/jester/user_stats.csv`. Reject any user with a train standard
   deviation below 2.0 (165 users of 7,200; none of them is in the top 40).
   Take the top 40. This group has a median crowd concordance of 0.496, a
   median kNN ceiling of 0.691, and a median test gap of +0.198.
2. **Control group.** Add 20 users drawn at random from the rest, with the
   same seed. Without them the study cannot say whether selection, not the
   rubric, produced the gain.
3. **Split.** The split in this report. The gauge jokes stay in train. GEPA
   optimizes on the 60 train jokes. The 40 test jokes are untouched.
4. **Labels.** Three levels by the user's own terciles, computed on the 60
   train jokes only. Section (f) binned on all 100 ratings, so its numbers
   are a close guide, not the same quantity. Run five levels later if the
   three-level result is positive.
5. **Metric.** Per-user pairwise concordance on the 40 test jokes, the
   definition in `src/joke_pref/metric.py`. Also report NMAE, with the level
   midpoints mapped to -6.67, 0 and +6.67.
6. **Baselines, all four.** Generic rubric (the real baseline, text only),
   crowd predictor (the generic ceiling), kNN k = 80 (the personal ceiling),
   and a shuffled-label GEPA run (the null).
7. **Success.** Per user, personal rubric minus generic rubric, on the test
   jokes. Because 40 items give roughly 780 pairs and the split-half
   correlation of a per-user gap is only 0.36, judge the **median over the
   40 selected users**, not one user.
   - **+0.20 median: at the ceiling, so treat it as not reachable.** The kNN
     ceiling for this group is +0.198, and kNN reads 60 of the user's own
     ratings while the rubric reads none.
   - **+0.10 median, with the control group near zero: a real result.** That
     is half the honest CF ceiling, from text alone.
   - **+0.03 median or less: no signal in the process.**

## Limits

1. The crowd and kNN numbers are **ceilings**, not forecasts for a rubric.
   The kNN predictor reads 60 of the user's own ratings and every other
   user's rating of the test jokes. A rubric reads joke text only.
2. 100 jokes is a small pool. 40 test items give a noisy per-user number.
   Read the median over users, not one user.
3. The 7,200 complete users are self-selected: they sat through all 100
   jokes. They may agree with the crowd more than a normal user does.
4. Our split is not the published split, so our NMAE numbers are not a
   reproduction of Eigentaste 0.187 or POP 0.203.
5. The Eigentaste style predictor here is a simplification: a quantile grid,
   not the recursive rectangular clustering of the paper.
6. The jokes are 20 to 25 years old. Taste in the test set is partly taste in
   1999 US politics and gender humour. A result on Jester may not carry over
   to fresh material.
7. Jester gives no repeated ratings, so we cannot separate rater noise from
   taste in the unexplained variance.
