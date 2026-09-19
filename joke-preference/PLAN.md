# PLAN: joke-preference

Active chunk. Detail and current state are in `STATUS.md`; the session
summary is in `HANDOFF.md`.

## Jester chunk: complete (2026-09-20)

All three phases ran. Result in `docs/jester-results.md`. The chunk is
closed; git holds the plan text (commit `2e9345c`).

## Candidates for the next chunk (not started, Naveen decides)

1. Write-up: an article section from `docs/jester-results.md`, with
   Naveen's own result as the motivating case (+11 over generic, great level
   not learned), Jester as the multi-user result with a ceiling, DeepSeek
   as the cost comparison. Open items to state: no intra-rater ceiling
   anywhere; the personal gain is in the atypical tail; the ten
   highest-variance Jester jokes are dated or offensive.
2. Typical users gain nothing. Test a two-stage rubric (crowd rubric as
   seed instead of a-generic), or a larger GEPA budget for the 20 random
   users only, to see if that is a budget limit or a data limit.
3. GPPL humour (pairwise, 433 workers) and DICES-350 (safety, 123 raters)
   as the second and third datasets; see `docs/research/`.
4. Relabel pass on Naveen's data for an intra-rater ceiling.
5. DeepSeek Flash without the reasoning trace as the GEPA writer, to cut the
   21 minutes per user to about 3, and check the quality cost.
