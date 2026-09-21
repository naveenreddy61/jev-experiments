# PLAN: joke-preference

Active chunk. Detail and current state are in `STATUS.md`; the session
summary is in `HANDOFF.md`.

## Jester chunk: complete (2026-09-20)

All three phases ran. Result in `docs/jester-results.md`. The chunk is
closed; git holds the plan text (commit `2e9345c`).

Naveen closed the experiment as a negative result on 2026-09-21. The
article is `articles/a-rubric-is-not-a-recommender.md`. A new experiment
idea comes next; nothing below is scheduled.

## Ideas left over from this experiment (not started)

1. Typical users gain nothing. Test a two-stage rubric (crowd rubric as
   seed instead of a-generic), or a larger GEPA budget for the 20 random
   users only, to see if that is a budget limit or a data limit.
2. GPPL humour (pairwise, 433 workers) and DICES-350 (safety, 123 raters)
   as the second and third datasets; see `docs/research/`.
3. Relabel pass on Naveen's data for an intra-rater ceiling.
4. DeepSeek Flash without the reasoning trace as the GEPA writer, to cut the
   21 minutes per user to about 3, and check the quality cost.
