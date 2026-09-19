# PLAN: Jester run, three phases

Active chunk. Delete a phase when it is complete. Detail and current state
are in `STATUS.md`; the session summary is in `HANDOFF.md`.

## Goal

Show, on public data with a measured ceiling, that Jev plus an optimized
rubric gives a meaningful personal signal for text recommendation, at System
One cost, with no ratings at inference time. The method sits between
collaborative filtering (needs ratings on the item) and an LLM judge (costs
100 to 1000 times more per item).

Order fixed by Naveen on 2026-09-19: **(1) a Jev result with a generic
rubric, (2) an optimized Jev + GEPA result, (3) a DeepSeek Flash benchmark.**

## Data and split (fixed, from `docs/jester-signal.md`)

- `jester-data-1`, the 7,200 users who rated all 100 jokes.
  `data/jester/jokes.jsonl`, `data/jester/ratings.csv.gz`,
  loaders in `src/joke_pref/jester.py`.
- Item split, same for every user, seed 20260919: gauge jokes
  {5,7,8,13,15,16,17,18,19,20} plus 50 more in train (60), 40 in test.
  `scripts/jester_signal.py` defines it; expose it from `jester.py`.
- Users: the top 40 by `gap_train_only` in `data/jester/user_stats.csv`
  (atypical users, selected on train jokes only) plus 20 random users
  (seeded). 60 users in total.
- Levels per user: bad / good / great by that user's own terciles of the 60
  **train** ratings. Test jokes are binned with the same train cut points.
- Per-user reference numbers on the tercile truth: `crowd_concordance` and
  `knn_concordance` in `results/jester/generic-a-generic/users.csv`. The
  numbers in `user_stats.csv` are on the raw ratings and are lower.

## Metric

Primary: per-user pairwise concordance on the 40 test jokes, all pairs with
different tercile labels, ties 0.5 (same definition as `metric.py`). Report
median and a paired test across the 60 users. Secondary: NMAE from the
expected level mapped to the user's tercile midpoints; Spearman with the raw
rating. Always show four columns per arm group: crowd, generic rubric,
personal rubric, kNN. `jester_eval.concordance_rows` computes it for all
users at once.

## Phase 1: done (2026-09-20)

Result in `docs/jester-results.md`. Generic rubric `probe/a-generic.json`
is the baseline to beat: median concordance 0.604 on all users, 0.553 on
the selected 60. References on the same tercile truth: crowd 0.650 / 0.524,
kNN 0.710 / 0.734. Command: `joke-pref jester-evaluate`. Grouping for
Phase 2: `jester.user_items(ratings, jokes, user_id, split)`.

## Phase 2: Jev + GEPA per user

For each of the 60 users: `optimize` on the 12 train groups (60 jokes) with
val = a 3-group hold-out of the train set (GEPA needs a val set; do not
touch the 40 test jokes), joint proposer, minibatch 6 groups (30 jokes),
budget 600 metric calls, DeepSeek Flash as writer. Score the 40 test jokes
with the best rubric.

Work items: `joke-pref jester-optimize --user U` built on `runs.optimize`
with `user_items(..., "train")` split 9 groups train / 3 groups val, then
`jester_eval.evaluate_rubric` on the best rubric for that one user; a driver
that loops over the 60 selected users (`selected_user_ids` in
`results/jester/generic-a-generic/summary.json`) and appends one row per
user to `results/jester/personal/users.csv`.

Controls, same test jokes:

- cross-user: the best rubric of user A scored on user B (pair users
  randomly); expected at the generic level.
- shuffled-label null for 10 users.

Deliverable: per-user table crowd / generic / personal / cross-user / kNN,
paired test personal vs generic, share of the (kNN minus crowd) gap that
the personal rubric closes, `docs/jester-results.md` updated. Cost estimate:
60 users x about 2,000 Jev calls plus 40 DeepSeek calls, under two dollars.
Run in the background with `nohup`, one user after another, a monitor on
the log. Success: personal beats generic on the paired test with the
cross-user control flat, and closes half or more of the gap to kNN.

## Phase 3: DeepSeek Flash as the judge

Same 40 test jokes, same rubrics, DeepSeek Flash asked to return the level
(prompt = instructions + the three level texts + the joke, answer one word,
temperature 0). Arms: generic rubric, each user's personal rubric. Report
concordance next to Jev, plus cost per 1,000 jokes and latency per call for
both. This answers two questions: does the rubric text transfer to an LLM
reader, and what does the System One tier buy. Also run it on Naveen's own
test split with `results/naveen/gepa-2/best_criteria.json`.

Work items: `src/joke_pref/llm_judge.py` (reuse `_post_json` from
`reflection.py`), `joke-pref judge --provider deepseek ...`, cache by
(model, rubric, joke).

## Phase 4: write-up

`docs/jester-results.md` becomes the basis of the article section. Keep
Naveen's personal result as the motivating case (412 labels, +11 over
generic, great level not learned), Jester as the multi-user result with a
ceiling, DeepSeek as the cost comparison. Open items to state: no
intra-rater ceiling anywhere; humor has a small personal gap; the ten
highest-variance Jester jokes are dated or offensive.

## Not in this chunk

GPPL and DICES runs. Relabel pass on Naveen's data. Comparative scoring
mode. Stronger reflection model.
