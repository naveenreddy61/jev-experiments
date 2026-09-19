# Public datasets for the per-user rubric experiment: recommender-system and review slice

Date: 2026-09-19. Scope: classic recommender-system datasets and review datasets that
include item text. The goal is a dataset where we can optimize one rubric per user and
then hold out that user's items.

## Result first

Jester is the best fit, and it is the only candidate that satisfies every criterion.
The joke text is downloadable today. I downloaded it and read it. Per-user density is
higher than in all other candidates. The domain is subjective humor, which matches the
first experiment. Published baselines exist on the same data, with exact numbers.

The second group of candidates trades domain match for scale. MyAnimeList, Goodreads,
and Food.com give many ratings per user and real item text. The item text is a synopsis
or a blurb, not the object of enjoyment itself.

One warning applies to almost every candidate. No published work measures "per-user
optimized rubric against generic rubric, by pairwise concordance". For most datasets the
honest answer is: no directly comparable number exists, and we must run a per-user
collaborative-filtering baseline ourselves.

---

## Verification status

I verified these facts directly:

- The Jester joke text archive. I downloaded it and read the files.
- The Eigentaste result table. I extracted the text of the published PDF.
- The New Yorker caption contest file format. I downloaded a summary CSV and read the header.
- The Goodreads book record fields, the genre subset counts, and the license text.
- The Amazon Reviews 2023 per-category counts.
- The density of `jester-data-1`. I downloaded the file and counted every cell.
- The license of the R package `recommenderlab`, from the CRAN DESCRIPTION file.

I could not verify these facts:

- The Jester 150-joke versions. The host `eigentaste.berkeley.edu` refused the connection
  from this machine, over HTTPS and over HTTP.
- The license of the Kaggle MyAnimeList mirror. Kaggle pages did not render for me.
- The license of the Kaggle Food.com mirror. Same reason.
- Whether the per-annotator HaHackathon labels are in the public release.

---

## Candidate 1: Jester

| Field | Value |
| --- | --- |
| Name | Jester joke ratings, Goldberg et al. |
| Ratings URL | https://goldberg.berkeley.edu/jester-data/ |
| Joke text URL | https://raw.githubusercontent.com/BerkeleyAutomation/jester/master/jokes/jesterjoketext.zip |
| Alternative text copy | `JesterJokes` in the R package `recommenderlab`, https://rdrr.io/cran/recommenderlab/man/Jester5k.html |
| License | "Freely available for research use when acknowledged", with a citation to the Eigentaste paper |
| Item type | Jokes. 28 to 69 words in the three files I measured |
| Items | 100 jokes in the classic release |
| Users | 73,421 across the three files |
| Ratings | 4.1 million |
| Ratings per user | 36 or more for `jester-data-1` and `jester-data-2`. 15 to 35 for `jester-data-3` |
| Scale | Continuous, -10.00 to +10.00. The value 99 marks "not rated" |
| Item text bundled | No, but see the two text URLs above |

I downloaded `jesterjoketext.zip` and unpacked it. It holds 100 HTML files, `init1.html`
to `init100.html`. Each file marks the joke body with `<!--begin of joke -->` and
`<!--end of joke -->`. A simple regular expression extracts the text. The three files I
measured hold 28, 69, and 29 words.

### Measured density of `jester-data-1`

I downloaded `jester-data-1.zip` and counted the non-99 values in the 24,983 by 100
matrix. The results follow.

- 7,200 users rated all 100 jokes. That pool is large enough for our experiment.
- The deciles of the per-user rated count are 36, 42, 48, 58, 70, 72, 74, 95, 100, 100, 100.
- Ten jokes are rated by almost every user. These are jokes 5, 7, 8, 13, 15, 16, 17, 18,
  19 and 20, each with 24,975 to 24,981 ratings. This set matches the gauge set that the
  Eigentaste paper describes.
- About 25 more jokes have 24,700 or more ratings. The density falls off gradually, not sharply.
- The sparsest jokes are 71 to 80, with 8,532 to 9,074 ratings each, that is about 35 percent
  of users.

The gauge set matters for the experimental design. The Eigentaste paper states that the
authors chose the gauge jokes "based on a combination of their correlations and variances".
The gauge jokes are therefore selected items, not a random sample. Do not put them in the
holdout.

Note the license split. Berkeley publishes the ratings for research use with
acknowledgement. The joke text copy in the GitHub repository carries no separate license
file. The `recommenderlab` copy carries the package license. I read the CRAN DESCRIPTION
file for `recommenderlab` version 1.0.7. The license is GPL-2. That copy is therefore the
cleanest redistributable source of the joke text. State the source you use in any publication.

### Published baselines, from the Eigentaste paper

I extracted these numbers from the PDF at
https://goldberg.berkeley.edu/pubs/Eigentaste-Info-Retrieval-Journal.pdf

The metric is Normalized Mean Absolute Error (NMAE). Lower is better.

| Algorithm | NMAE |
| --- | --- |
| Random, uniform model | 0.333 |
| Random, normal model | 0.282 |
| POP, global item mean | 0.203 |
| 1 nearest neighbor | 0.238 |
| 80 nearest neighbors | 0.187 |
| Eigentaste | 0.187 |

The paper used about 2,500,000 ratings from 57,000 users.

Verdict: strong fit. The domain is identical to our first experiment. The item text
carries the reason a person laughs. Density is the highest of all candidates. Two
published numbers, POP at 0.203 and Eigentaste at 0.187, give a real comparison point.
The weakness is the item pool. 100 jokes limit the holdout to a few tens of items per user.

---

## Candidate 2: MyAnimeList 2020

| Field | Value |
| --- | --- |
| Name | Anime Recommendation Database 2020, Hernan4444 |
| URL | https://github.com/Hernan4444/MyAnimeList-Database and https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020 |
| License | Not stated in the README. Treat it as unclear |
| Item type | Anime synopsis, a few hundred words |
| Items | 17,562 anime. 16,872 in `rating_complete.csv` |
| Users | 325,772 in the full list. 310,059 in `rating_complete.csv` |
| Ratings | 109 million list rows. 57 million completed-and-scored ratings |
| Ratings per user | About 184 on average in `rating_complete.csv` |
| Scale | 1 to 10 |
| Item text bundled | Yes. `anime_with_synopsis.csv` holds the synopsis |

Published baselines: none that I can cite with a number. Many Kaggle notebooks report
RMSE, but they are not primary sources and the splits differ.

Verdict: possible fit, and the best of the large-scale options. Density is excellent and
the text is bundled. Two problems: the license is unclear, and a synopsis describes the
plot rather than the experience. Taste in anime depends on animation, music, and pacing,
which the synopsis does not carry.

---

## Candidate 3: Goodreads, UCSD Book Graph

| Field | Value |
| --- | --- |
| Name | Goodreads Book Graph datasets, Wan and McAuley |
| URL | https://mengtingwan.github.io/data/goodreads.html and https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html |
| License | "Collected for academic use only. Please do not redistribute them or use for commercial purposes" |
| Item type | Book description, that is, a publisher blurb |
| Items | About 2,360,000 books |
| Users | 876,145 |
| Interactions | 228,648,342 shelf interactions. 112,131,203 reads. 104,551,549 ratings |
| Ratings per user | About 119 on average, if you count only the 104.5 million ratings |
| Scale | 1 to 5 |
| Item text bundled | Yes. `goodreads_books.json.gz` holds a `description` field |

Genre subsets, books and interactions and detailed reviews:

| Genre | Books | Interactions | Reviews |
| --- | --- | --- | --- |
| Poetry | 36,514 | 2,734,350 | 154,555 |
| Comics and Graphic | 89,411 | 7,347,630 | 542,338 |
| Children | 124,082 | 10,059,349 | 734,640 |
| Fantasy and Paranormal | 258,585 | 55,397,550 | 3,424,641 |
| History and Biography | 302,935 | 31,479,229 | 2,066,193 |
| Mystery, Thriller and Crime | 219,235 | 24,799,896 | 1,849,236 |
| Romance | 335,449 | 42,792,856 | 3,565,378 |
| Young Adult | 93,398 | 34,919,254 | 2,389,900 |

Citations: Wan and McAuley, RecSys 2018. Wan, Misra, Nakashole and McAuley, ACL 2019.

Published baselines: the RecSys 2018 paper reports item-recommendation metrics for
monotonic behavior chains. I did not extract the numbers, and the task is ranking from
implicit chains, not per-user rating prediction. Treat it as "no directly comparable number".

Verdict: possible fit. The Poetry subset is the most interesting slice, because a poetry
collection blurb carries more of the reading experience than a thriller blurb does. The
counts are large enough that a "100 or more ratings" user pool is easy to build.

---

## Candidate 4: Personalized humor datasets with per-annotator labels

These datasets keep the label of each annotator instead of a mean. The profile table
below comes from Bielaniewicz et al., "From Generalized Laughter to Personalized
Chuckles", https://arxiv.org/pdf/2312.11296 , Table I.

| Dataset | Text type | Texts | Annotations | Annotators | Annotations per annotator | Language |
| --- | --- | --- | --- | --- | --- | --- |
| Cockamamie Gobbledegook | 1 to 2 words | 10,884 | 40,673 | 351 | 115.88 | English |
| Humor | Tweets | 8,284 | 26,967 | 3,137 | 57.44 | Spanish |
| Humicroedit | News headline pairs | 14,886 | 74,430 | 5 | 14,886 | English |
| Doccano 1 | Comments | 880 | 31,521 | 39 | 808.23 | Polish |
| Doccano 2 | Comments | 8,891 | 17,533 | 49 | 357.81 | Polish |

Two cautions. The Humicroedit "5 annotators" number is an artifact. Humicroedit stores
five grader slots per headline, not five persistent people. Do not treat it as a per-user
dataset. The Cockamamie items are one or two words, which gives our model almost no text
to read.

The Spanish Humor dataset with text, not tweet IDs, is at
https://github.com/pln-fing-udelar/humor/tree/main/previous

HaHackathon, SemEval 2021 Task 7, is the most attractive member of this family:
10,000 texts, 202,369 ratings, 1,821 annotators, 111.13 texts per annotator on average,
and no text with fewer than 17 votes. Source: Meaney et al.,
https://aclanthology.org/2021.semeval-1.9/ and the follow-up analysis
https://arxiv.org/pdf/2208.10898 . I could not confirm that the public release keeps the
annotator identifier. The 2312.11296 paper lists HaHackathon under "generalized"
datasets, which suggests that the public files hold only the aggregate. Confirm this
before you plan work on it.

Published baselines: the 2312.11296 paper reports macro F1 for generalized models against
personalized models, per dataset. That is the closest published comparison to our claim
that I found. The task is binary "funny or not", not a level scale.

Verdict: strong fit in principle, weak fit in practice today. The domain and the label
structure match our claim exactly. Availability is the blocker for the English datasets.

---

## Candidate 5: Amazon Reviews 2023

| Field | Value |
| --- | --- |
| Name | Amazon Reviews 2023, McAuley Lab |
| URL | https://amazon-reviews-2023.github.io/ and https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 |
| License | Not stated on the project page. Check the Hugging Face card before use |
| Item type | Product title and description, that is, marketing text |
| Items | 48 million |
| Users | Hundreds of millions across 33 categories |
| Ratings | 571.54 million |
| Scale | 1 to 5 |
| Item text bundled | Yes, in the item metadata files |

Per-category counts, and the ratings-per-user ratio that I computed from them:

| Category | Users | Items | Ratings | Ratings per user |
| --- | --- | --- | --- | --- |
| Clothing, Shoes and Jewelry | 22.6M | 7.2M | 66.0M | 2.9 |
| Home and Kitchen | 23.2M | 3.7M | 67.4M | 2.9 |
| Beauty and Personal Care | 11.3M | 1.0M | 23.9M | 2.1 |
| Books | 10.3M | 4.4M | 29.5M | 2.9 |
| Electronics | 18.3M | 1.6M | 43.9M | 2.4 |

Verdict: weak fit as a rating dataset. The average user rates about three products. A
"100 or more ratings" pool exists only in the extreme tail, and that tail is not a normal
user. The item text is marketing copy, while the rich text is the review the user wrote,
which is an output and not an input.

The LaMP benchmark reuses this data in a more useful shape. LaMP-3 is personalized
product rating prediction, 1 to 5, evaluated with MAE and RMSE, with 2,500 test samples
and a leaderboard. See https://arxiv.org/html/2304.11406 . One reported result is a MAE
improvement from 0.275 to 0.236 for a fine-tuned model with retrieval augmentation. I did
not verify that number against the leaderboard. LaMP-3 is close to our claim in shape,
because the model reads one text and returns a level. The task is different, because the
text is the user's own review and the level is the star count the user gave it.

---

## Other candidates, in brief

### Yelp Open Dataset

https://business.yelp.com/data/resources/open-dataset/ . 6,990,280 reviews, 150,346
businesses, 11 metropolitan areas. The license permits academic use only, and forbids
redistribution and commercial use. Business text is thin: name, category, and attributes.
The review is the user's own writing. Verdict: weak fit, for the same reason as Amazon.

### MovieLens with plot summaries

https://files.grouplens.org/datasets/movielens/ml-25m-README.html . The ratings are
dense per user, but MovieLens carries no plot text. `links.csv` holds an IMDb identifier
and a TMDB identifier. The MovieLens license forbids redistribution and commercial use.
A join to IMDb or TMDB adds a second license, and each provider sets its own terms.
Verdict: possible, but the license work is a real cost and the plot summary does not
carry taste well.

MPST, Movie Plot Synopses with Tags, https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags ,
holds 14,828 synopses with 71 tags. It has no per-user ratings. Use it only as a text
source for a join.

### Food.com recipes

https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions ,
paper: Majumder et al., EMNLP 2019, https://aclanthology.org/D19-1613/ . The processed
set holds 178,265 recipes, 25,076 users, and 749,053 reviews. That is about 30 ratings
per user. The rating scale is 1 to 5. The recipe text is long: title, ingredients, and
steps. Verdict: possible fit. Density is below our target, and the long item text raises
the cost per call.

### MIND, Microsoft News

https://msnews.github.io/ . About 160,000 English articles, more than 15 million
impression logs, 1 million users. MIND-small samples 50,000 users. Every article carries
a title, an abstract, a body, a category, and entities. The license is the Microsoft
Research License Terms. The feedback is implicit: click or no click. Verdict: possible,
but weaker. A click is not enjoyment. Headlines are short, which keeps the cost low. Use
it only if you accept a click-through metric such as AUC or nDCG@k.

### PENS, personalized news headlines

https://msnews.github.io/pens.html . About 113,000 articles and 500,000 impression logs
from more than 445,000 users. Microsoft Research License Terms. The test set is
hand-made by native speakers. Verdict: weak fit for a rating task, because the label is
a generated headline, not a preference level.

### New Yorker caption contest

I downloaded https://raw.githubusercontent.com/nextml/caption-contest-data/master/summaries/510_RoundRobin.csv
and read the header. The columns are:

`rank,caption,mean,precision,votes,not_funny,somewhat_funny,funny`

That file holds 3,905 captions for one contest. The rows are aggregate counts per
caption. There is no rater identifier. The repository publishes 397 such summary files.
The Hessel corpus, https://github.com/jmhessel/caption_contest_corpus , is a benchmark of
ranking and explanation tasks, not a per-rater log. Verdict: weak fit for the per-user
claim. It is still useful as a generic-rubric baseline, because the three level names
(not funny, somewhat funny, funny) match a three-level rubric exactly.

### Book-Crossing

278,858 users, 1,149,780 ratings, 271,379 books. Ziegler et al., WWW 2005. The book record
holds a title, an author, a year, a publisher, and a cover image URL. There is no blurb.
The average user gives about four ratings. Verdict: weak fit.

### Steam reviews

Several Kaggle mirrors exist, for example
https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam . Game
descriptions are available from the Steam store API. The per-user review count is low,
and the label is a thumbs up or thumbs down. Verdict: weak fit.

### Epinions, CiaoDVD, Douban, Ta-Feng

These are identifier-and-rating datasets with trust graphs. CiaoDVD holds 72.7 thousand
movie ratings. Douban holds 129,490 users and 58,541 movie items. None of them ships
usable item text. Ta-Feng is retail transaction data with no text. Verdict: weak fit for
all four.

### Poems, quotes, short fiction, fanfiction

I found no public dataset of poems, quotes, short fiction, or fanfiction that carries
per-user preference ratings with a user identifier. AO3 kudos counts are aggregate. The
Gutenberg Poetry corpus and the Poetry Foundation scrape carry no ratings at all. If you
want poetry, the Goodreads Poetry subset is the only route with per-user labels.

---

## Ranked shortlist

1. **Jester**, `jester-data-1`. Strong fit. Only candidate that meets every criterion,
   with published numbers on the same data.
2. **MyAnimeList 2020**. Possible fit. About 184 ratings per user and a bundled synopsis.
   The license is the open question.
3. **Goodreads, Poetry or Comics subset**. Possible fit. Clear academic license, book
   descriptions, and about 119 ratings per user across the whole set.
4. **HaHackathon and the personalized humor family**. Strong on domain, blocked on
   availability. Confirm the per-annotator release first.
5. **LaMP-3**. Different task, but it is the one candidate with a live leaderboard for
   per-user level prediction from text.

---

## Protocol for Jester

### Call budget

GEPA uses about 20 to 50 rubric candidates. Each candidate scores about 50 training
items. That gives 1,000 to 2,500 calls for each user, plus the holdout evaluation. At a
total budget of 50,000 to 500,000 calls, we can run about 20 users at the low end and
about 200 users at the high end. Jester items are 28 to 69 words, so the cost per call
is near the minimum.

### Data preparation

1. Download `jester-data-1.zip` from https://goldberg.berkeley.edu/jester-data/ .
2. Download `jesterjoketext.zip` from the BerkeleyAutomation repository.
3. Extract each joke body between the `<!--begin of joke -->` and `<!--end of joke -->`
   markers. Strip the HTML tags.
4. Map the joke file index `init<N>.html` to the rating column `N`. Confirm this mapping
   on three jokes by hand before you trust it.
5. Remove the value 99, which marks "not rated".

### User selection

1. Keep only users who rated all 100 jokes.
2. Remove users whose rating variance is near zero. A flat rater gives the optimizer no signal.
3. Sample 50 users at random from the rest. Fix the random seed.

The pool is 7,200 users, which I counted directly. No threshold change is necessary.

### Split and label

1. Put the 10 gauge jokes 5, 7, 8, 13, 15, 16, 17, 18, 19 and 20 in the training set
   always. They are variance-selected items, so they bias a holdout.
2. Split the other 90 jokes into 50 training jokes and 40 holdout jokes.
3. Use the same split for every user, so that the holdout items are comparable.
4. Convert the continuous rating to five levels by that user's own quintiles. Five levels
   keep more magnitude than three levels, at the same cost per call. NMAE punishes the
   loss of magnitude, and NMAE is the metric that carries the published comparison.
5. If you want a direct comparison with the first joke experiment, run a second pass with
   three levels from the user's terciles.

### Conditions to compare

1. **Generic rubric.** A five-level funniness rubric with no personal information. Write
   one sentence of guidance per level. For the three-level pass, use the New Yorker
   contest wording: not funny, somewhat funny, funny.
2. **Per-user rubric.** The same rubric after GEPA optimizes it on that user's 60
   training jokes.
3. **POP control.** Predict the mean rating of the joke over the training users. This is
   the same control that the Eigentaste paper used.
4. **Neighbor control.** An 80 nearest neighbor predictor on the same holdout, with the
   same user pool.

### Metrics

1. Per-user pairwise concordance on the 40 holdout jokes. This is our primary metric. No
   published number exists for it.
2. NMAE, to compare with the published numbers. Convert the level probabilities to a
   rating estimate by the expected value over the level midpoints. Report NMAE for all
   four conditions.

### The number to beat

Report NMAE against POP at 0.203 and against Eigentaste at 0.187. State clearly that our
NMAE is not measured on the exact split of the 2001 paper, so the comparison is
indicative and not exact. The strict comparison is our own POP and 80 nearest neighbor
runs on the same holdout.

### Expected outcome and risk

The first experiment showed 10 to 20 points of pairwise concordance over a generic
rubric. The risk is that a large part of the Jester signal is the joke and not the person.
The published evidence on this point is favourable to us. The GroupLens team claimed that
users generally agree on joke ratings, from correlations on the `rec.humor` newsgroup. The
Eigentaste paper disputes that claim. It states that 75 percent of `rec.humor` jokes got
the lowest rating, which "skewed all correlations dramatically upward", and it notes "a
substantial number of low and negative correlations". The Eigentaste authors then chose
jokes "with a much higher variance" for Jester. Per-user variance in Jester is therefore
real, by design. Measure the per-joke rating variance yourself before the main run, and
use it to confirm this.

---

## Sources

- https://goldberg.berkeley.edu/jester-data/
- https://raw.githubusercontent.com/BerkeleyAutomation/jester/master/jokes/jesterjoketext.zip
- https://goldberg.berkeley.edu/pubs/Eigentaste-Info-Retrieval-Journal.pdf
- https://rdrr.io/cran/recommenderlab/man/Jester5k.html
- https://grouplens.org/datasets/jester/
- https://mengtingwan.github.io/data/goodreads.html
- https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html
- https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/books
- https://github.com/Hernan4444/MyAnimeList-Database
- https://amazon-reviews-2023.github.io/main.html
- https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- https://arxiv.org/html/2304.11406
- https://arxiv.org/pdf/2312.11296
- https://aclanthology.org/2021.semeval-1.9/
- https://arxiv.org/pdf/2208.10898
- https://github.com/pln-fing-udelar/humor/tree/main/previous
- https://github.com/nextml/caption-contest-data
- https://github.com/jmhessel/caption_contest_corpus
- https://business.yelp.com/data/resources/open-dataset/
- https://files.grouplens.org/datasets/movielens/ml-25m-README.html
- https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags
- https://aclanthology.org/D19-1613/
- https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
- https://msnews.github.io/
- https://msnews.github.io/pens.html
- https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam
