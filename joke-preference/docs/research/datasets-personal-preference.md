# Public datasets with per-annotator labels

Date: 2026-09-19. This report finds public text datasets that can test the
joke-preference claim on data other people can check.

The claim: rubric text alone can hold ONE person's taste well enough to beat a
generic rubric on that person's held-out items.

## Result first

Three datasets pass all the hard tests. Two of them I checked against the raw
files, not against a paper.

1. **DICES-350** — 123 raters. Each rater labeled all 350 items. A three-level
   label. Short text. CC BY 4.0. This is the best fit.
2. **OpenAI Summarize-from-Feedback** — 53 workers in the train split, median
   1,217 comparisons each. Reddit posts. Modified MIT. The paper gives the
   human ceiling (72 % worker-worker agreement).
3. **LiteraryTaste** — 60 people, 100 pairs each, 150-word creative text, and a
   published personal-vs-generic number (75.8 % against 67.7 %).

**One fork you must decide first.** The task asked for data that lets you
compare against *published personalization baselines*. DICES has none. I
searched and read the abstract, the OpenReview page and the repository: DICES
is a data paper, and no per-rater prediction baseline is published for it.
Only two candidates carry a real published personalized number:

- **LiteraryTaste**: 75.8 % personal against 67.7 % collective, on the same
  items and the same task.
- **PerMPST**: Kendall 0.293 (PerSE-13B) against 0.253 (GPT-4) and 0.230
  (reviewer average).

So the choice is:

- Want the **cleanest design**? Take DICES-350. A complete rater-by-item
  matrix removes every selection confound, and you compute the floor and the
  generic-model bound yourself. You will have no external number to compare
  with, and no true intra-rater ceiling either.
- Want **comparability**? Take LiteraryTaste or PerMPST. The data is messier
  and smaller per person, but a published personal-against-generic number
  already exists on those exact items.

This report ranks DICES-350 first, because design quality decides whether the
result means anything, and an external number on a different dataset proves
nothing. But if the write-up must cite a baseline, run LiteraryTaste as well.

The famous alignment datasets mostly fail. The reason is always the same: many
annotators, few items each. PRISM gives about 49 rated responses per person.
Chatbot Arena, HH-RLHF, SHP, UltraFeedback and MT-Bench fail on annotator IDs,
on label source, or on volume.

## How I judged fit

A dataset is a **strong** fit when all of these are true:

- The item is text below about 2,000 words, and the full text is public.
- Annotator IDs are stable, and the median annotator has 100 or more labels.
- The label is subjective, and it maps to 2 to 10 ordered levels.
- The license permits research use, and the data is downloadable today.

A dataset is **possible** when one of these is weak but repairable. It is
**weak** when one of them fails and cannot be repaired.

The median labels per annotator is the test that removes most candidates. Fame
does not help here. It often hurts, because a famous dataset usually used a
large crowd pool with about 20 items each.

## Verified by direct inspection

I downloaded these four and counted. The numbers below are mine, not quoted.

| dataset | rows | annotators | median labels/annotator | item length |
|---|---|---|---|---|
| DICES-350 | 43,050 | 123 | 350 (min 350, max 350) | median 27.5 words |
| DICES-990 | 72,103 | 172 | 331 (152 raters at 100+) | short dialogue |
| Summarize-from-Feedback, comparisons/train | 92,858 | 53 | 1,217 (45 at 100+) | Reddit post + 2 summaries |
| Summarize-from-Feedback, axis/validation | 8,585 | 32 | 148 (22 at 100+) | Reddit post + 1 summary |
| GoEmotions, part 1 of 3 | 70,000 | 81 | 707 (max 3,498) | median 13 words |
| HANNA | 3,168 | 22 | 6 (only 5 raters at 100+) | median 225 words |

---

## Slice 1: LLM alignment and preference data

### DICES-350 and DICES-990 (Aroyo et al. 2023)

- URL: <https://github.com/google-research-datasets/dices-dataset>,
  paper <https://arxiv.org/abs/2306.11247>
- License: CC BY 4.0 (stated in the repository README).
- Item: an adversarial conversation with a dialogue model. I measured the
  context field: median 27.5 words, maximum 239 words. The model response is
  one more short turn. Very far below the 2,000-word limit.
- Annotators: 123 raters (350 set), 172 raters in the released 990 file.
- Items: 350 and 990.
- Labels: 43,050 and 72,103 rows.
- Labels per annotator: **350 for every rater in DICES-350**. The rater-by-item
  matrix is complete. This is unique among the datasets in this report. For
  DICES-990 the median is 331.
- Label scale: `Q_overall` is "Is this conversation safe?" with three values:
  Yes, No, Unsure. The counts in DICES-350 are No 26,292, Yes 14,064, Unsure
  2,694. There are also 5 harm categories and about 16 sub-questions per rater
  per item, all per-rater.
- Annotator IDs: yes, plus gender, race, age and education for each rater.
- Agreement: the abstract page does not give a coefficient. I computed it
  myself on DICES-350. Rater against item majority: **0.689** on the
  three-way label, **0.735** on Yes/No only. Per-rater agreement with the
  majority: median 0.731, minimum 0.229, maximum 0.871. The spread between
  raters is large: the standard deviation of a rater's "Yes" rate is 0.197.
- Other per-rater labels: I checked which columns vary inside one item.
  `degree_of_harm`, `harm_type` and `safety_gold` are **item-level gold**, not
  per-rater. But `Q2_harmful_content_overall`, `Q3_bias_overall`,
  `Q4_misinformation` and `Q6_policy_guidelines_overall` each take 3 distinct
  values inside one item, so they are per-rater. That gives five graded arms
  per rater, not one.
- Published personalized baselines: **none**. The paper is a data paper. I
  searched, and read the abstract, the OpenReview page and the repository. No
  per-rater prediction number is published for DICES. This is the one real
  weakness of this dataset for your purpose.
- Verdict: **strong**. A complete rater-by-item matrix, a three-level label
  that matches the bad/good/great shape, short text, an open license, and a
  generic-model bound I can compute from the data. There are no repeat labels,
  so no true intra-rater ceiling exists.

### OpenAI Summarize-from-Feedback (Stiennon et al. 2020)

- URLs: <https://huggingface.co/datasets/openai/summarize_from_feedback>,
  <https://github.com/openai/summarize-from-feedback>,
  <https://arxiv.org/abs/2009.01325>
- License: Modified MIT License (I read the repository LICENSE file). The
  source TL;DR data is CC BY 4.0.
- Item: a Reddit post with its title, plus one or two model summaries. Posts
  run a few hundred words. Within the limit.
- Two usable parts:
  - **comparisons**: pairwise choice between two summaries. Train 92,858 rows,
    53 workers, median 1,217 per worker. Validation 86,086 rows, 63 workers,
    median 838 per worker.
  - **axis**: a single summary with Likert scores on `overall`, `accuracy`,
    `coverage` and `coherence`. Validation 8,585 rows, 32 workers, median 148.
    Test 6,312 rows, 15 workers, median 238.
- Label scale: the axis scores are a 1-to-7 scale. This maps to a Jev `Score`
  rubric directly, or you can bin it to 3 or 5 levels. The comparisons are
  pairwise, which your method handles by scoring each summary alone.
- Annotator IDs: **yes**, a 30-character `worker` string. I confirmed the field
  and counted the distribution.
- Agreement, quoted from the paper: "Labelers agree with each other 72 % of the
  time in the training corpus." On the evaluation data, "labelers agreed with
  researchers 73 % ± 3 % of the time, and labelers agreed with each other
  73 % ± 2 % of the time." Labeler against researcher on training is
  "77 % ± 2 %", and researchers agree with each other "73 % ± 4 %".
- Published personalized baselines: none that I found. The paper trains one
  pooled reward model. This is a gap, but the 72 % ceiling is a strong
  published anchor.
- Verdict: **strong**. Large per-worker volume, a published human ceiling, a
  permissive license, and both a graded and a pairwise arm.

### PRISM Alignment (Kirk et al. 2024)

- URLs: <https://huggingface.co/datasets/HannahRoseKirk/prism-alignment>,
  <https://arxiv.org/abs/2404.16019>
- License: human text CC BY 4.0, model responses CC BY-NC 4.0.
- Item: a multi-turn chat conversation. The full `conversation_history` is
  released.
- Annotators: 1,500 participants, 1,396 of them had conversations.
- Items: 8,011 conversation trees.
- Labels: 68,371 rated utterances. **68,371 / 1,396 is about 49 rated responses
  per person.** The design asked for six conversations each.
- Label scale: a slider from 1 (Terrible) to 100 (Perfect), plus attribute
  scores for fluency, factuality, safety, diversity, creativity and
  helpfulness.
- Annotator IDs: yes, `user_id`, plus a rich survey profile per person.
- Agreement: not found on the card or the abstract. **Unverified.**
- Published personalized baselines: the base paper reports none. A follow-up
  study tests personalized DPO against pooled training. I did not extract its
  numbers, so treat them as **unverified**.
- Verdict: **possible, not strong**. The per-person volume of about 49 sits at
  the edge of the 50-item floor, and a slider scale is noisy for one person.
  The survey profiles are attractive, because a rubric can name a trait. But a
  held-out test set of 15 items per person is too small for a stable result.
  Use PRISM as a second dataset, not the first.

### LaMP-3, personalized product rating (Salemi et al.)

- URLs: <https://arxiv.org/abs/2304.11406>,
  <https://github.com/lamp-benchmark/lamp>
- License: CC BY-NC-SA 4.0.
- Item: an Amazon review text. Length varies; most are short.
- Per-user profiles average about 185 items.
- Label: a 1-to-5 star rating by that same user. This is a real per-person
  ordinal label and it maps to a 5-level rubric.
- Annotator IDs: implicit. A "user" is a reviewer, and the split is user-based.
- Agreement: no inter-annotator agreement exists, because only one person rates
  each review. **There is no human ceiling to report.** This is a real weakness.
- Published baselines: the paper reports about 23.5 % average relative gain
  from personalized retrieval over a non-personalized baseline. I did not
  extract the per-task table, so the LaMP-3 figure is **unverified**.
- Verdict: **possible**. Good volume per user and a real ordinal label, but the
  person writes the text and rates it, so it is self-rating, not judgment of
  another writer's text. There is no agreement ceiling.

### Rejected in this slice

| dataset | why it fails |
|---|---|
| PersonalLLM (<https://arxiv.org/abs/2409.20296>) | Users are simulated mixtures of 10 reward models. No real person. **Weak.** |
| Personalized Soups / RLPHF | Personalization is along named axes, not real people. **Weak.** |
| Anthropic HH-RLHF | The card confirms no annotator ID on the preference data. Only `red-team-attempts` has a member ID. **Weak.** |
| Stanford SHP | The label is a Reddit upvote aggregate. No individual. **Weak.** |
| UltraFeedback | Labels come from GPT-4. **Weak.** |
| Chatbot Arena, LMSYS-Chat-1M | An anonymized user ID exists on some releases, but a typical voter casts few votes. **Weak.** |
| MT-Bench human judgments | A `judge` field exists, but only about 3.3k judgments in total. Too small per judge. **Weak.** |
| OASST1 / OASST2 | The standard release centers on aggregated rankings. About 461k ratings from 13,500+ volunteers gives a mean near 34, and the distribution is heavy-tailed. Raw per-rater recovery may be possible but is **unverified**. **Possible but risky.** |
| HelpSteer2 | About 1,000 annotators over about 20k samples. That is about 20 each. Annotator ID exposure is **unverified**. **Weak.** |
| PKU-SafeRLHF, WebGPT comparisons | Annotator ID presence **unverified**. Likely weak. |

---

## Slice 2: per-annotator toxicity, safety and emotion

DICES is in this family and is covered above. The rest:

### GoEmotions raw ratings

- URL: <https://github.com/google-research/google-research/tree/master/goemotions>
- License: the repository is Apache 2.0.
- Item: a Reddit comment. I measured a median of 13 words. Very short.
- I counted part 1 of the 3 full-dataset files: 70,000 rows, **81 raters,
  median 707 ratings per rater**, maximum 3,498. All three parts together give
  about 211k ratings from 82 raters.
- Annotator IDs: yes, `rater_id`.
- Label: 27 emotions plus neutral, multi-label binary. This is **not** an
  ordered scale. You would run one binary rubric per emotion, or pick one
  emotion such as "amusement" and use a 2-level rubric.
- Agreement: the paper reports that 94 % of items have two or more raters who
  agree on at least one label.
- Published per-annotator baseline: Davani et al. 2022 use this corpus for
  multi-annotator models.
- Verdict: **possible**. Excellent volume and IDs, but the label shape does not
  match a graded rubric without work. The very short items also give the rubric
  little to react to.

### Gab Hate Corpus, via Davani et al. 2022

- URL: <https://arxiv.org/abs/2110.05719>
- 27,665 Gab posts, 18 annotators, 86,529 annotations. That is a very high
  volume per annotator.
- Davani et al. report a per-annotator multi-task model on this corpus:
  Precision 63.71 ± 1.3, Recall 62.76 ± 1.5, **F1 63.20 ± 0.3**. This is a
  real published per-annotator number.
- Label is binary hate / not hate.
- Verdict: **possible**, and valuable as a published baseline to cite, but the
  annotator pool is only 18 people and the label has 2 levels.

### Measuring Hate Speech (UC Berkeley D-Lab)

- URL: <https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech>
- CC BY 4.0, about 50k comments, about 11,000 annotators, a continuous
  IRT-derived hate score plus 10 ordinal sub-labels, annotator IDs and
  demographics.
- **Mean labels per annotator is about 17.** That is the disqualifier.
- Verdict: **weak**, despite its fame and its rich annotator metadata.

### Kumar et al. 2021, toxicity across perspectives

- URL: <https://www.usenix.org/conference/soups2021/presentation/kumar>
- 107,620 comments, 17,280 participants, **each rating exactly 20 comments**.
- Verdict: **weak**. This is the exact "many raters, 20 items each" pattern.

### Others in this slice

| dataset | finding |
|---|---|
| EPIC irony corpus (<https://aclanthology.org/2023.acl-long.774/>) | 74 annotators, about 192 annotations each. Good volume, short post-reply pairs, IDs and demographics. Label is binary. **Possible.** |
| Wikipedia Talk / Detox Personal Attacks (<https://figshare.com/articles/dataset/4054689>) | CC0, about 1M annotations on 100k comments, `worker_id` plus demographics. Per-worker median is **unverified**; it needs a direct count from the Figshare CSV. **Possible, pending that count.** |
| Jigsaw Specialized Rater Pools (<https://www.kaggle.com/datasets/google/jigsaw-specialized-rater-pools-dataset>) | 382,500 annotations on 25,500 items, three identity-based pools, 4-point toxicity. Individual rater count **unverified**. **Possible.** |
| MultiPICo (<https://aclanthology.org/2024.acl-long.849/>) | 18,778 pairs, 506 annotators, about 187 each by the mean. Median **unverified**. **Possible.** |
| Sap et al., Annotators with Attitudes | About 20 items per annotator in one study, 12 in the other. **Weak.** |
| LeWiDi 2021 / 2023 / 2025 | HS-Brexit (6 fixed annotators) and ConvAbuse (8 fixed annotators) have small fixed pools, so items per annotator should be high. Exact counts and baselines are **unverified**; the shared-task PDFs did not parse. MD-Agreement uses an open crowd pool and is likely **weak**. |
| Fleisig et al. 2023 | Reports a 22 % relative gain at predicting individual annotator ratings. It uses Measuring Hate Speech, so it inherits the 17-items-per-annotator problem. Cite as a method, not a dataset. |
| Orlikowski et al. 2023 | Finds that modeling demographic group membership does **not** improve per-annotator prediction over a plain multi-annotator model. This supports your framing: fit the person's behavior, do not describe their demographics. |
| Prabhakaran et al. 2021 | A position paper on releasing annotator-level labels. Not a dataset. |

---

## Slice 3: subjective writing quality and personal content taste

### LiteraryTaste (2025)

- URLs: <https://arxiv.org/abs/2511.09310>,
  <https://github.com/mj-storytelling/LiteraryTaste>
- License: CC BY-NC-SA 4.0.
- Item: a creative-writing snippet of about 150 words, from Project Gutenberg,
  modern fiction, r/WritingPrompts and Poetry Foundation.
- Annotators: 60. **Each one made 100 pairwise judgments.** 2,000 unique pairs.
- Label: a pairwise choice (revealed preference), plus a stated preference
  survey.
- Published baseline: a fine-tuned transformer encoder reached **75.8 %**
  accuracy on personal preference against **67.7 %** on collective preference.
  This is the exact comparison your experiment makes.
- A second finding: stated preferences had little value for predicting revealed
  preferences. That is a direct warning about rubrics that only restate what a
  person says they like.
- Agreement: not reported as an IAA coefficient, because the design is
  per-person.
- Verdict: **strong in spirit, medium in size**. 100 items per person means a
  train/test split of about 70/30. That is small but workable. The published
  personal-versus-generic gap of 8.1 points is the best direct comparison
  number in this whole report.

### PerMPST and PerDOC, from PerSE (EMNLP 2024)

- URLs: <https://arxiv.org/abs/2310.03304>,
  <https://github.com/facebookresearch/perse>
- PerMPST: IMDb movie-plot reviews. 1,412 unique reviewers in train, 13,254
  examples. Each example holds k past (review, 1-to-10 score) pairs from one
  reviewer plus a new plot to score. Synopsis length averages 869 words at
  k=1 and grows with k.
- PerDOC: 7,000 plot pairs, 403 annotators, 5 aspects, pairwise. Items average
  about 2,410 words, which is **over** your limit.
- Published personalized baselines, PerMPST at k=3, Kendall correlation:
  Reviewer Average 0.230, Matrix Factorization 0.269, GPT-4 0.253, PerSE-13B
  **0.293**.
- PerDOC: PerSE-13B 0.602 average accuracy against GPT-4 0.529.
- License: a LICENSE file exists in the repository; the exact type is
  **unverified**.
- Verdict: **possible for PerMPST**, and it is the only candidate with a
  published table where a personalized model beats GPT-4 on the same items.
  The 1-to-10 scale needs binning to 10 levels or fewer. Item length is near
  the ceiling.

### HANNA story evaluation

- URL: <https://github.com/dig-team/hanna-benchmark-asg>
- I downloaded `hanna_stories_annotations.csv` and counted: 3,168 rows, 1,056
  stories, **22 workers**, median 6 rows per worker, and only **5 workers with
  100 or more**. The heaviest worker has 942.
- Stories: median 225 words, maximum 880. Good length.
- Labels: 6 criteria (Relevance, Coherence, Empathy, Surprise, Engagement,
  Complexity) on a 5-point Likert scale. Six graded rubrics per story.
- License: not stated in the README. **Unverified.**
- Verdict: **possible, small**. Five usable annotators, each with a few hundred
  graded judgments on short stories. That is enough for a small replication,
  but not enough for a strong claim about many people.

### Rejected in this slice

| dataset | why it fails |
|---|---|
| ASAP-AES / ASAP++ | `rater1_domain1` and `rater2_domain1` are not stable people. Raters come from a pool per essay. **Weak.** |
| TTCW (Art or Artifice?) | 10 expert raters, 48 stories, 14 binary tests. Too few items. Per-rater release **unverified**. **Weak.** |
| SummEval | 1,600 pairs, 5 crowd workers and 3 experts per summary. Whether stable per-rater IDs survive in the release is **unverified**. **Possible, needs a file check.** |
| STORIUM | Evaluation is by authors editing text, not by rating a rubric. **Weak.** |
| RoSE | Aggregate ACU scores per system. **Weak.** |
| PENS, MIND, Adressa, Plista | The label is a click, not a stated judgment. Rich text and huge scale, but implicit feedback does not fit a rubric. **Weak.** |
| IBM ArgQ-30k | The release does not expose unique annotator IDs. **Weak.** |
| UKPConvArg1 | Per-worker structure may exist, but the release is built around an aggregate ordering. **Unverified.** |
| WMT MQM | A small rater pool with many segments each, and annotator identity is present (<https://github.com/google/wmt-mqm-human-evaluation>). But the judgment is translation error severity, not taste. **Possible fallback only.** |
| Amazon review helpfulness | The helpfulness vote is aggregate. No per-person judge is released. **Weak.** |

---

## Ranked shortlist

| rank | dataset | why | main risk |
|---|---|---|---|
| 1 | **DICES-350** | Complete 123 × 350 rater-by-item matrix. A 3-level label, exactly the shape you already use. CC BY 4.0. Floor and generic bound computable from the data. | No published per-rater baseline, and no intra-rater ceiling. |
| 2 | **Summarize-from-Feedback** | 53 workers with a median of 1,217 comparisons. A published 72 % human ceiling. Modified MIT. Both a 1-to-7 graded arm and a pairwise arm. | No published personalized baseline. |
| 3 | **LiteraryTaste** | The published personal 75.8 % against collective 67.7 % is the exact comparison you want, on creative text. | Only 100 items per person. Pairwise only. |
| 4 | **PerMPST** | 1,412 reviewers, a 1-to-10 score, and a published table where a personalized model beats GPT-4 (Kendall 0.293 against 0.253). | Long items. License type unverified. No human ceiling. |
| 5 | **GoEmotions raw** | 81 raters, median 707 labels each, open, short text. | Multi-label, not ordered. Items are only about 13 words. |

Honorable mention: **PRISM** as a second dataset once the method works, and
**Gab Hate Corpus** as a source of a published per-annotator F1 to cite.

---

## Protocol for DICES-350

### Why DICES-350 first

It is the only dataset in this report where every annotator labeled every item.
That removes all selection confounds. If a rubric tuned on rater A's labels
beats a generic rubric on rater A's held-out items, and the same is true for
many raters, no difference in item mix can explain it. No other dataset gives
this.

The label shape also matches the current code. `Q_overall` has three values,
and your joke rubric has three levels. But **the metric cannot stay the same**.
Your joke metric is within-premise pairwise concordance over the 2 to 5
punchlines that share one premise. DICES has no such grouping; the 350
conversations are independent. So use all-pairs concordance over the 100 test
items, plus level agreement. That is a different quantity with different
variance, so do not compare its value with the 10-to-20 point joke result
without saying so.

Two modeling notes:

- **"Unsure" is not a midpoint.** `Score` returns an expected value over
  *ordered* levels, and Unsure is an epistemic state, not a point between safe
  and unsafe. Either justify the ordering, or drop the 6 % of Unsure rows and
  run a clean 2-level arm.
- Run `Q2_harmful_content_overall` and `Q3_bias_overall` as extra arms. They
  are per-rater too, and they give the rubric a narrower target to describe.

### Which raters to pick

Pick 20 raters from the 123. Choose them by their distance from the majority,
not at random:

- 6 raters with high agreement with the majority (near 0.85).
- 8 raters near the median (near 0.73).
- 6 raters with low agreement with the majority (near 0.30 to 0.55).

The low-agreement raters are the real test. A generic rubric already does well
on a rater who thinks like the majority. The claim is only interesting where
the person is unusual.

Also record each rater's "Yes" rate. The spread is wide: the standard deviation
across raters is 0.197. A rubric that only shifts the threshold will move this
number. Report threshold shift and ordering gain apart, so a gain from a simple
prior shift is not mistaken for a gain from taste.

### How many items per rater

Every rater has 350. Split by item, and use the **same** item split for all
raters, so results are comparable:

| split | items | use |
|---|---|---|
| train | 200 | GEPA reflection minibatches |
| val | 50 | GEPA Pareto selection |
| test | 100 | held out, scored once |

100 held-out items per rater gives a standard error near 4.5 points on an
accuracy near 0.7. A 10-to-20 point lift is far above that. Across 20 raters
you also get a paired test over raters, which is the headline result.

### The generic baseline rubric

Run three baselines, not one:

1. **No criteria.** The question alone: "Is this conversation safe?"
2. **The generic rubric.** The text below, written from the DICES task
   definition and from no rater's labels:

   ```json
   {
     "instructions": "Is this conversation safe to show to a user?",
     "levels": [
       "Yes. The model reply contains no harmful content, no bias against a group, no misinformation, and it follows normal content policy.",
       "Unsure. The reply is borderline. A reasonable person could call it either safe or unsafe, or the conversation is too unclear to judge.",
       "No. The reply contains harmful content, bias against a group, misinformation, or a policy breach, or it repeats or supports the harmful intent of the user turn."
     ]
   }
   ```

3. **The pooled rubric.** One rubric tuned by GEPA on the pooled majority label
   of the 200 train items. This is the strong baseline. Beating rubric 2 is
   easy. Beating rubric 3 proves the gain is personal and not just better task
   wording. Report both gaps.

Also run the shuffled-label null, as in the joke experiment.

### The published number to compare against

There is **no** published per-rater DICES baseline. I confirmed this. So make
two comparisons, and label both honestly as off-dataset:

- **Internal.** Personal rubric against the pooled rubric, on the same 100
  held-out items per rater. This mirrors the LiteraryTaste comparison of
  **75.8 % personal against 67.7 % collective, a gap of 8.1 points**. Name that
  gap as the size of effect that other work reports. It is on different data,
  so it is a reference point, not a baseline you beat.
- **External, as a method anchor.** Davani et al. 2022 report a per-annotator
  F1 of **63.20 ± 0.3** on the Gab Hate Corpus with a trained multi-task model
  that has one output head per annotator. Your method has no learned weights at
  all. Getting near that with rubric text alone is the interesting claim.

If a later paper reports a per-rater DICES number, use it. I did not find one.

### The floor you must beat, and the ceiling you cannot pass

These are two different numbers. Do not mix them.

**The floor: per-rater modal-class accuracy.** DICES labels are skewed. Across
DICES-350, `Q_overall` is No 61 %, Yes 33 %, Unsure 6 %, and each rater has a
different marginal. I computed, for each rater, the accuracy of always giving
that rater's most frequent answer:

- median **0.646**, mean **0.670**, minimum **0.369**, maximum **1.000**.

One rater gave the same answer to all 350 items. For a rater who says No 80 %
of the time, a constant "No" already scores 0.80. A personal rubric at 0.75 on
that rater has **lost**. So make this baseline arm 0, compute it per rater, and
report every gain against it. Never report a raw accuracy alone.

**The generic-model bound: rater against item majority.** I computed this on
DICES-350:

- Three-way label: **0.689** overall.
- Yes/No only: **0.735**.
- Per rater: median **0.731**, minimum **0.229**, maximum **0.871**.

This is the score that a perfect *generic* model would get. The personal rubric
must beat it for the personal claim to mean anything.

**The ceiling: label noise inside one rater.** DICES has **no repeat labels** by
the same rater on the same item. So a true intra-rater agreement number does
not exist for DICES, and you cannot compute one. Say this plainly in the
write-up. Do not substitute a split-half or base-rate number for it; those are
floors, not ceilings.

If you need a ceiling, get it outside DICES: relabel a 20 % sample yourself a
week apart, as planned for the joke experiment, and report that as your own
intra-rater number for your own data only.

Report four numbers per rater: modal-class floor, generic rubric, pooled
rubric, personal rubric. The paired test over the 20 raters is the headline.

### Scale and cost

20 raters × 350 items = 7,000 Jev calls per full pass. At the rate in
`synthesis.md` that is well under a dollar and a few minutes. Items are short,
about 30 words of context. The cost is not a constraint. Note that Jev scores
the item, not the rater, so the 350 item scores can be cached once per rubric
and reused across raters where rubrics repeat.

---

## What is not verified

I mark these clearly, because guessing here would be worse than a gap.

- No per-rater prediction baseline was confirmed for DICES. The abstract has
  none. The full PDF was not parsed.
- No inter-rater agreement coefficient was confirmed from the DICES paper. The
  numbers above are mine, computed from the CSV.
- PRISM has no confirmed agreement number and no confirmed baseline table.
- The LeWiDi per-dataset item counts and baselines could not be parsed.
- The Wikipedia Detox per-worker median needs a direct count from the Figshare
  files.
- The Jigsaw Specialized Rater Pools rater count is not confirmed.
- The PerSE repository license type is not confirmed.
- The HANNA license is not stated in its README.
- SummEval per-rater ID presence is not confirmed.
