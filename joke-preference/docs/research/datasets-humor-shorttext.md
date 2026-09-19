# Humor and short-text datasets with per-rater labels

Slice: humor, funniness, and other short-text subjective judgments.
Date of research: 2026-09-19.
Purpose: find public data to test the claim that rubric text alone can encode one
person's taste.

## Result

One dataset satisfies all four criteria. It is the **Simpson et al. (2019) GPPL
humour preference data**, which the LeWiDi 2021 shared task later reused. It has
1,059 real rater IDs, 141,050 pairwise "which is funnier" judgments on 4,030
short texts, an Apache-2.0 license, and published baselines. The median rater
gave 60 judgments; 433 raters gave 100 or more.

The second candidate is the **NEXT / nextml caption-contest response data**. It
has a 3-level funniness scale that matches the Jev rubric shape exactly, and a
median of 97 votes per participant ID. Its weakness is that the participant ID
is issued per page visit, not per person, and that the item is a caption that
needs a cartoon image to make sense.

Everything else on the list fails criterion 2. Humicroedit, HaHackathon,
#HashtagWars, rJokes, ColBERT, the Humor Norms, Memotion, and the Upworthy
Archive all publish aggregated labels only. Jester is the one classic exception:
it is per-user and dense, but its labels are recommender-system ratings, and its
100 jokes give a small item pool.

**Verified first-hand fact that sets the target**: in the GPPL humour data, the
majority vote of four raters predicts the fifth rater's A/B choice only **70.9%**
of the time (ties dropped, n = 113,194). This is the **generic-predictor
baseline**, not a ceiling: it is the accuracy of the best impersonal predictor
that crowd consensus can give. A well-written generic rubric tries to reach it. A
personal rubric that goes above it shows real personalization. The ceiling is a
different quantity: the split-half stability of one rater's own taste.

## Ranked shortlist

| # | Dataset | Per-rater IDs | Labels per rater (median) | Fit |
|---|---------|---------------|---------------------------|-----|
| 1 | GPPL humour / LeWiDi 2021 Humour | Yes, MTurk worker IDs | 60 | Strong |
| 2 | NEXT caption-contest responses | Session-scoped only | 97 | Possible |
| 3 | Jester (Dataset 1) | Yes, one row per user | 72 of 100 jokes | Possible |
| 4 | PRISM Alignment | Yes, `user_id` | about 49 (mean) | Weak |
| 5 | Humicroedit / FunLines | No | Not applicable | Weak |

Only #1 is a strong fit. #2 and #3 are usable with stated compromises. #4 and #5
are listed because they are the best of the remainder, not because they work.

---

## 1. GPPL humour preference data (Simpson, Bujel and Gurevych 2019)

- **Name**: humour preference data from "Predicting Humorousness and Metaphor
  Novelty with Gaussian Process Preference Learning" (ACL 2019). Reused as the
  "Humour" dataset in LeWiDi 1st edition / SemEval-2021 Task 12.
- **URL**: https://github.com/UKPLab/acl2019-GPPL-humour-metaphor
  (file `data/pl-humor-full/results.tsv`).
  Paper: https://aclanthology.org/P19-1572/
  LeWiDi overview: https://aclanthology.org/2021.semeval-1.41/
- **License**: Apache-2.0 (LICENSE.txt in the repository).
- **Item type and length**: one-line English texts. Homographic puns,
  heterographic puns, non-pun jokes, proverbs and aphorisms. Verified: 4,030
  unique texts, 2 to 68 words, median 10 words.
- **Raters**: 1,059 unique Amazon Mechanical Turk worker IDs (verified count).
- **Items and labels**: 141,050 pairwise judgments over 28,210 unique unordered
  pairs. Exactly 5 judgments per pair (verified: median = mean = max = 5).
- **Labels per rater**: verified median 60, mean 133, minimum 10, maximum 2,200.
  610 workers gave 50 or more. 433 workers gave 100 or more.
- **Label scale**: 3 answers. A is funnier, B is funnier, or X (tie). Verified
  distribution: A 58,986, B 54,652, X 27,412. The tie rate is 19.4%.
- **Per-rater IDs released**: YES. Verified by printing the file header:
  `Worker ID<TAB>Answer<TAB>Text A<TAB>Text B`, with real worker strings such as
  `A147F5PJTHOB8A`. This is the only humor dataset in this survey where real,
  linkable rater IDs were confirmed inside a downloaded file.
- **Inter-rater agreement**: computed first-hand by leave-one-out. A held-out
  rater agrees with the majority of the other four in 56.3% of all 141,050
  judgments; 63.6% when the held-out rater did not answer "tie"; **70.9%** when
  all ties are dropped on both sides (n = 113,194). The LeWiDi paper gives only
  an observed-agreement figure, not a numeric Krippendorff alpha for Humour;
  that alpha is not verified.
- **Intra-rater agreement**: not measurable. Verified: zero worker-pair repeats
  in 141,050 rows. No worker ever saw the same pair twice.
- **Published baselines**: LeWiDi 2021, Table 2, Humour dataset. GPPL base
  model: hard-label F1 = 0.557, soft-label cross-entropy = 0.728. The one
  participant system ("uor"): F1 = 0.513, cross-entropy = 3.697. Note that the
  baseline beat the submission. Simpson et al. 2019 report GPPL results on the
  same corpus in their own paper.
- **Verdict**: **STRONG FIT.** It is text-only, it has real rater IDs with many
  judgments each, the label is pure taste, and the pairwise form maps directly
  onto the pairwise-concordance metric the joke experiment already uses.
- **Caution**: the LeWiDi packaged release (`humour.zip`) ships only a test-pair
  skeleton plus a script to re-fetch text. Use the UKPLab repository as the
  primary source.

## 2. NEXT / nextml New Yorker caption-contest data

- **URL**: https://github.com/nextml/caption-contest-data-api (raw per-vote
  files under `contests/responses/*.csv.zip`, stored with git-LFS) and
  https://github.com/nextml/caption-contest-data
- **License**: CC BY 4.0 for the data (LICENSE.txt).
- **Item type**: a one-line caption for a New Yorker cartoon. The cartoon is an
  image. The caption alone is often not judgeable.
- **Scale claim from the repository README**: "over 89 million ratings on over
  750,000 unique captions during 155 contests" as of contest 652 (2019-03-03).
- **Label scale**: 3 levels. "unfunny", "somewhat funny", "funny", stored as
  `target_reward` in {1, 2, 3}. A dueling ("which is funnier") variant also
  exists.
- **Per-rater IDs**: a `participant_uid` column exists and repeats heavily.
  Verified on contest 510 (82,627 rows): 372 distinct IDs, votes per ID with
  median 97, mean 222, maximum 4,865. **But** the repository documentation
  states the ID is issued per page visit: "if the same user visits the page
  twice, they will get two participant IDs." So it is a session pseudo-ID, not a
  durable person.
- **Aggregation tier**: the `summaries/` files hold only
  `rank,caption,mean,precision,votes,not_funny,somewhat_funny,funny` with no
  participant column.
- **Published baselines**: Hessel et al. 2023 (ACL best paper),
  https://aclanthology.org/2023.acl-long.41.pdf, Table 2. Quality ranking, crowd
  accuracy: random 50.0, GPT-4 5-shot 73.3, human estimate from pixels 83.7.
  (Per-model rows between these three were read from a secondary extraction of
  Table 2 and are not repeated here; check the table directly if a specific model
  number is needed.) Note: Hessel's ranking ground truth is the **mean** NEXT
  crowd score, not an individual rater.
- **Verdict**: **POSSIBLE FIT.** The 3-level scale and the vote volume per ID
  are ideal. Two problems block a strong rating: the ID does not persist across
  sessions, and the item is not self-contained text. A workaround is to prepend
  the human-written cartoon description from Hessel et al. to each caption, so
  the item becomes one self-contained text block.

## 3. Jester (Goldberg et al., Eigentaste)

- **URL**: https://eigentaste.berkeley.edu/dataset/ (the host refused
  connections during this research; the content was read from the Internet
  Archive and the older mirror https://goldberg.berkeley.edu/jester-data/, which
  is live and served the files).
- **License**: not an OSI license. The stated terms are: "Freely available for
  research use when acknowledged with the following reference: Eigentaste: A
  Constant Time Collaborative Filtering Algorithm. Ken Goldberg, Theresa Roeder,
  Dhruv Gupta, and Chris Perkins. Information Retrieval, 4(2), 133-151. July
  2001." The author also asks for a courtesy email. Research use only. Do not
  describe it as open-license.
- **Joke text**: YES, available. `jester_dataset_1_joke_texts.zip` (92 KB) gives
  100 files `init1.html` to `init100.html`, with the joke between
  `<!--begin of joke -->` markers. Verified: 14 to 220 words, median 38 words.
  Dataset 3 adds text for 150 jokes; Dataset 4 adds jokes 151-158.
- **Structure**: verified by downloading and parsing `jester-data-1.xls`. Shape
  24,983 x 101. Column 1 = count of jokes rated. Columns 2-101 = ratings for
  jokes 1-100. Value 99 means "not rated".
- **Labels per rater**: verified. Minimum 36, median **72**, maximum 100. 19,640
  users rated 50 or more jokes. 7,200 users rated all 100.
- **Label scale**: continuous, -10.00 to +10.00 (verified range -9.95 to 10.00).
- **Per-rater IDs**: YES, implicitly. One row per user. There are no
  demographics and no user names, but the row index is a stable user key.
- **Gauge set**: jokes {5, 7, 8, 13, 15, 16, 17, 18, 19, 20} are near-complete.
  Verified missing fraction 0.018%. This is the "universal query" set from the
  Eigentaste paper.
- **Other releases**: Dataset 3 is 54,905 users x 150 jokes (2006-2015), with 22
  jokes retired in May 2009. Dataset 4 adds over 100,000 ratings from 7,699
  users (2015-2019) and 8 new jokes.
- **Inter-rater agreement**: no agreement statistic is published. Jester gives no
  repeated ratings, so intra-rater reliability is **not measurable**.
- **Published baselines**: the Eigentaste paper reports Normalized Mean Absolute
  Error near 20%, and states in its appendix that uniform random guessing gives
  NMAE = 33%. Recommender-system baselines are large in number but are
  collaborative-filtering results, not content-based ones, so they are not a
  fair comparison for a text-only rubric.
- **Verdict**: **POSSIBLE FIT.** Per-user structure and joke text are both
  confirmed, and 100 jokes with a median of 72 ratings per user is enough for a
  train/test split per person. The weakness is the item pool: 100 jokes is small,
  the jokes are from 1999-2003 and are dated, and the published baselines are
  collaborative-filtering numbers that a content-only method should not be
  compared against directly.

## 4. Humicroedit and FunLines (SemEval-2020 Task 7)

- **URL**: https://cs.rochester.edu/u/nhossain/humicroedit.html ;
  https://github.com/n-hossain/semeval-2020-task-7-humicroedit ;
  https://huggingface.co/datasets/SemEvalWorkshop/humicroedit
- **License**: CC BY 4.0 (stated on page 1 of the overview paper).
- **Item type**: a news headline with one word replaced to make it funny.
- **Size**: 15,095 edited headlines (train 9,652, dev 2,419, test 3,024), plus
  8,248 FunLines rows. 5 graders per headline. Scale 0-3.
- **Per-rater IDs**: **NO.** Verified by printing the CSV header:
  `id,original,edit,grades,meanGrade`. The `grades` field is a bare digit string
  such as `10000` or `22110`. Digit position is not a stable rater across items,
  so no rater can be followed from one headline to the next. The FunLines game
  release uses the identical schema, with no player ID. Any private
  player-linked version is not verified.
- **Published baselines**: overview paper, Tables 3-5. Subtask 1 baseline RMSE
  0.575; best system (Hitachi) RMSE 0.49725. Subtask 2 baseline accuracy 49.0%;
  best system 67.43%.
- **Verdict**: **WEAK FIT.** Excellent as a generic-rubric benchmark with a real
  published number to beat. Useless for per-person optimization.

## 5. HaHackathon (SemEval-2021 Task 7)

- **URL**: https://aclanthology.org/2021.semeval-1.9/
- **Item type**: 10,000 short texts, mixed from Twitter and the Kaggle Short
  Jokes set.
- **Raters**: 20 per text, from 1,569 unique Prolific annotators over 2,062
  valid sessions.
- **Label scale**: humor rating 0-5, offense rating 0-5, plus binary flags.
- **Per-rater IDs**: **NOT RELEASED** as far as can be verified. The public
  train.csv has 6 columns: `id, text, is_humor, humor_rating, humor_controversy,
  offense_rating`. All are aggregates. No raw per-annotator file was found on
  CodaLab, GitHub, Zenodo or OSF.
- **Inter-rater agreement**: Krippendorff alpha, Table 4. Humorous (binary)
  0.736. Humor rating (continuous) **0.124**. Offense rating 0.518. The 0.124
  figure is the most useful single number in this survey: it says that funniness
  ratings are almost entirely personal.
- **Published baselines**: Task 1a BERT baseline F1 0.911 / accuracy 0.928, top
  system PALI F1 0.9820. Task 1b humor rating BERT baseline RMSE 0.800, top
  system abcbpc RMSE 0.4959. Task 1c BERT baseline F1 0.4731, top system 0.4943.
- **Verdict**: **WEAK FIT** for personalization, but worth an email to the task
  organizers. 20 raters per text over 10,000 texts is the richest unreleased
  per-rater humor resource known. Raw data may be available on request.

## 6. SemEval-2017 Task 6 #HashtagWars

- **URL**: https://aclanthology.org/S17-2004/
- 12,734 tweets over 112 hashtags. Labels are curated by the @midnight TV show,
  not by independent raters: 2 = winning tweet, 1 = other top-10 tweet, 0 = rest.
- **Per-rater IDs**: none exist by design, because there are no raters.
- **Baselines**: Subtask A (pairwise accuracy), best system HumorHawk 0.675,
  random 0.500. Subtask B (rank edit distance, lower is better), best 0.872.
- **Verdict**: **WEAK FIT.** One aggregate taste ("what the show liked"), not a
  person.

## 7. Bulk joke corpora (aggregate or binary only)

| Dataset | URL | License | Labels | Per-rater? |
|---|---|---|---|---|
| rJokes (Weller & Seppi 2020) | https://github.com/orionw/rJokesData | Reddit ToS, no separate license | log upvote score, bucketed; 550k+ jokes | No |
| Short Jokes (Kaggle) | https://www.kaggle.com/datasets/abhinavmoudgil95/short-jokes | GPL v2 | none; 231,657 jokes, 10-200 chars | No |
| ColBERT 200k | https://github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection | MIT | binary, 100k/100k | No |
| CrowdTruth short-text humor | https://github.com/CrowdTruth/Short-Text-Corpus-For-Humor-Detection | not stated | binary | No |
| Pun of the Day (Yang et al. 2015) | https://aclanthology.org/D15-1284/ | not stated | binary, about 2,400 + 2,400 | No |
| 16000 One-Liners (Mihalcea & Strapparava 2005) | no live authoritative mirror found | not verified | binary | No |
| Reddit r/Jokes, r/dadjokes dumps | Pushshift dead since 2023; Arctic Shift / Academic Torrents remain | varies | net upvote score | No. Reddit never exposes per-user votes. |

Reported numbers: ColBERT accuracy 98.2%, F1 0.982 on its own split. Weller &
Seppi (EMNLP 2019) report F-measure 93.1% on Pun of the Day and 98.6% on a Short
Jokes set. rJokes numbers (RoBERTa about 72.4% accuracy; regression RMSE 1.614,
Spearman 0.435) come from secondary summaries and are **not verified** against
the paper tables.

**Verdict for the whole group: WEAK FIT.** Every label here is either a binary
class or an anonymous crowd aggregate. None can train a personal rubric.

## 8. Humor norms and psycholinguistic data

- **Engelthaler & Hills 2018, "Humor norms for 4,997 English words"**,
  https://doi.org/10.3758/s13428-017-0930-6 , data at
  https://github.com/tomasengelthaler/HumorNorms
  821 participants, each rated 211 words, scale 1-5. The released
  `humor_dataset.csv` holds **aggregates only**: mean, SD and n overall, plus
  splits by gender and by age median. The authors state that individual-
  difference analysis beyond those splits is not possible from the release.
  **WEAK FIT.** No per-participant rows. Also, single words are not the item type
  we need.
- **Ruch 3WD (Three Witz Dimensions) and Humor Styles Questionnaire studies**:
  no specific open OSF deposit with raw per-participant joke ratings was located.
  **NOT VERIFIED**, not ruled out. This is the clearest remaining gap and is
  worth a targeted OSF search.

## 9. Memes and multimodal

- **Memotion (SemEval-2020 Task 8)**, https://aclanthology.org/2020.semeval-1.99/
  About 10,000 memes, humor intensity labels (not funny / funny / very funny /
  hilarious), 2 annotators per meme. Whether the release carries each
  annotator's own vote is **not verified**; the visible release appears to hold
  the resolved label. **WEAK FIT** in any case: two raters per item is far too
  few, and the item is an image.
- **MemeCap**, https://github.com/eujhwang/meme-cap — 6,384 memes, about
  captioning and visual metaphor. It has no funniness scale at all. **NOT
  APPLICABLE.**
- **Stand-up and TED laughter timing** (for example Chen & Lee 2017): audience
  laughter is an aggregate room response with no rater identity. **WEAK FIT**,
  noted only.

## 10. Beyond humor: short text with subjective liking

- **Upworthy Research Archive**, https://osf.io/jd64p/ , paper
  https://www.nature.com/articles/s41597-021-00934-7 , CC BY 4.0.
  The paper states 32,487 experiments, 150,817 arms and 538,272,878 assignments.
  A 2024 data-quality revision supersedes those counts with 27,616 tests and
  128,217 packages; cite the 2024 figures. The label is aggregate
  click-through rate. The documentation states it "does not include any
  individual-level information to differentiate between viewers."
  **WEAK FIT.** No rater exists at all.
- **PRISM Alignment (Kirk et al., NeurIPS 2024)**,
  https://huggingface.co/datasets/HannahRoseKirk/prism-alignment , CC BY 4.0 for
  human-written text. 1,500 participants surveyed, 1,396 in conversations, 8,011
  conversation trees, 68,371 rated utterances, scale 1-100. `user_id` **is**
  released. Mean is about 49 utterances per participant; the true median is
  **not verified** and is probably lower. The item is a multi-turn conversation,
  and the label is response quality, not enjoyment.
  **WEAK FIT**, but the best non-humor text candidate.
- **Chatbot Arena Conversations**,
  https://huggingface.co/datasets/lmsys/chatbot_arena_conversations — 33k
  conversations, anonymized voter IDs tied to browser sessions from about 13k
  IPs. Not verified that any ID reaches 50 votes. Pairwise model preference, not
  taste. **WEAK FIT.**
- **OpinionQA (Santurkar et al. 2023)**,
  https://proceedings.mlr.press/v202/santurkar23a/ — per-respondent Pew survey
  data, 500 questions per respondent. Real per-person IDs and many items each,
  but the label is an opinion, not enjoyment. **WEAK FIT** for this claim, though
  it is a good structural template.
- **PersonalLLM**, https://arxiv.org/pdf/2409.20296 — preferences are simulated
  from reward models, not real people. **NOT APPLICABLE.**
- **Poetry**: an EEG study, "An EEG Dataset on Aesthetic and Creative Judgments
  of Brief Structured Poetry", Scientific Data 2025,
  https://www.nature.com/articles/s41597-025-06189-w — 51 participants, 210
  short texts, five 7-point rating dimensions including aesthetic appeal. This
  clears the items-per-rater bar. The release format and license are **not
  verified**. This is the strongest open lead in the survey and deserves a direct
  check of its data-availability statement.
  Wassiliwizky / Menninghaus poem studies use about 48 poems per participant;
  public per-participant release **not verified**.
- **Stories**: StoryER (Chen et al., EMNLP 2022),
  https://aclanthology.org/2022.emnlp-main.114/ — 100k ranked pairs and 46k
  aspect ratings by MTurk workers. Stable rater identity over 50+ items is **not
  verified** and is unlikely given the crowd fan-out design. HANNA and Per-DOC:
  **not verified**.
- **Quotes, ad copy, retweet intent, lyrics, book blurbs**: nothing found with
  per-rater IDs. Probably does not exist in public form.

## 11. Image analogues (not text, for reference)

These are listed only because they prove the design works when the data exists.

- **PARA**, https://arxiv.org/abs/2203.16754 — 31,220 images, 438 annotators,
  about 25 labels per image, so roughly 1,780 labels per annotator. Scale 1-5,
  with per-user demographics and Big-5 personality data.
- **FLICKR-AES**, https://github.com/alanspike/personalizedImageAesthetics —
  40,000 images, 210 workers, test-split workers rated about 137 images each.

---

## Suggested protocol for the best candidate

Dataset: **GPPL humour preference data**.

### Why this one

It is the only dataset in the survey where all four criteria hold at the same
time: self-contained short text, real rater IDs, many judgments per rater, an
open license, and a published baseline. The label is a pairwise funniness
preference, which converts to the pairwise-concordance metric the joke
experiment already reports, with no change of metric.

### Which raters to pick

Use the 433 workers with 100 or more judgments. From these, select 20 workers
for the main run:

- 10 workers chosen at random, to avoid cherry-picking.
- 10 workers chosen for **low agreement with the crowd**. These are the people
  whose taste a generic rubric cannot capture, so they are where a personal
  rubric must show its lift. Report both groups separately.

**Guard against a selection artifact.** Low crowd agreement has two causes:
genuinely unusual taste, and noise. If you select on agreement and then measure
lift against a crowd-derived baseline on the same judgments, regression to the
mean makes the baseline look weak and inflates the lift. The cross-worker control
does not catch this, because it degrades on noisy raters too.

Use split selection. For each worker, split the judgments in half at random.
Compute crowd agreement on half A only. Select the low-agreement workers from
half A. Optimize and evaluate on half B only. Selection and evaluation then use
independent data.

Also report each selected worker's split-half stability next to the agreement
score. A worker with low agreement **and** low self-stability is noise, not
taste. Exclude that worker instead of studying them.

Drop the "tie" answers from both training and test. The tie rate is 19.4% and a
tie carries no preference signal.

### How many items per rater

Per worker, with a median of 60 and a selected floor of 100 non-tie judgments:

- Optimization set for GEPA: 40 pairs.
- Held-out test set: the rest, at least 40 pairs.
- Split by pair, not by judgment, so no text appears in both halves.

Each pair gives two Jev calls, one per text. Score both texts with the same
rubric, take the level probabilities, compute an expected score, and predict the
higher-scoring text as "funnier". This keeps the one-item-per-call constraint.

Use a 3-level rubric to match the current joke experiment: not funny / mildly
funny / very funny.

### Generic baseline rubric

Write one rubric with no personal content, for example:

> Level 1, not funny: the text states something plainly, or the wordplay does
> not land.
> Level 2, mildly funny: there is a pun or a twist, but it is predictable.
> Level 3, very funny: the pun or twist is surprising and the two readings both
> work cleanly.

Run this same rubric, unchanged, for every worker. The difference between the
generic rubric and the GEPA-optimized personal rubric, measured on the same
held-out pairs, is the lift under test. Also run a second control: a rubric
optimized on **another** worker's labels and tested on this worker. If the
cross-worker rubric performs as well as the personal one, the lift is generic
quality, not personalization.

### What published number to compare against

Three reference points, in order of usefulness:

1. **Generic-predictor baseline, 70.9%.** Computed first-hand from the data by
   leave-one-out over the 5 judgments per pair, ties dropped. This is the
   accuracy of the best impersonal predictor available: the consensus of four
   other raters. It is a baseline, not a ceiling. A personal rubric that goes
   above it has learned something the crowd does not know.
2. **Chance, 50.0%** on the tie-dropped pairwise task.
3. **LeWiDi 2021 Humour baselines**: GPPL hard-label F1 = 0.557, soft-label
   cross-entropy = 0.728; best participant system F1 = 0.513. These are
   aggregate-label numbers, so they are a benchmark for the generic rubric, not
   for the personal one. Say this clearly when reporting.

### How to estimate the human ceiling

Two measures are possible, and one is not.

**Inter-rater (possible).** Every pair has exactly 5 judgments. Leave-one-out
over all pairs gives 70.9% (ties dropped). Also compute it per worker, which
gives a per-person "how typical is this person" score. Use it to sort workers
into the typical and atypical groups described above.

**Split-half within a worker (possible).** Split one worker's judgments into two
halves at random, fit a simple preference model such as Bradley-Terry on each
half, and correlate the two item score vectors. Apply the Spearman-Brown
correction. This measures how stable that person's taste is, which is the real
ceiling for any personal model of that person.

**Intra-rater (NOT possible).** Verified: zero worker-pair repeats in 141,050
rows. No worker ever judged the same pair twice, so test-retest reliability
cannot be computed from this data. Do not report an intra-rater figure. The same
limit applies to Jester, which has no repeated ratings either. If a true
intra-rater number is needed, it must come from a small new collection where
selected pairs are shown to the same worker twice.

### Fallback if the primary run is inconclusive

Use the NEXT caption-contest data with the Hessel et al. cartoon descriptions
prepended, so each item becomes self-contained text. Restrict to
`participant_uid` values with 100 or more votes inside one contest. Accept that
a "rater" is one sitting, not one person, and state that limit in the writeup.

---

## Facts that could not be verified

- Whether the HaHackathon organizers will release raw per-annotator data.
- Whether Memotion ships each annotator's own vote or only the resolved label.
- Whether a Ruch 3WD or Humor Styles Questionnaire study has an open OSF deposit
  with raw per-participant joke ratings.
- The release format and license of the 2025 poetry EEG dataset.
- The true median of ratings per participant in PRISM.
- The exact rJokes baseline numbers from the paper tables.
- A numeric Krippendorff alpha for the LeWiDi Humour dataset.
- Shahaf et al. 2015: no public per-rater release was found.
- Radev et al. 2016 collected 7 MTurk judgments per pair (a 4-vote difference
  occurred in 5,594 of 15,154 pairs, 36.9%), but a raw per-Turker release was
  not found.
