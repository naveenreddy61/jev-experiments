# A Rubric Is Not a Recommender

*A negative result: we tried to turn one person's taste into a paragraph of
text, hand it to a model that does not think, and beat the boring baseline.*

---

## The hook

Here are three levels of a rubric. A machine wrote them from 60 of one
person's joke ratings. Read the last one.

> **bad** — Long, multi-sentence story, dialogue, or fable whose payoff is
> flatly practical: a mundane workplace answer, a literal measurement, an
> itemized bill or fine schedule, a purchased access. Also named political or
> celebrity marital scandal, and harsh mockery where an identity supplies the
> punchline.
>
> **good** — Conventional well-formed jokes: short Q&A stereotype riffs,
> conversational loopholes, political wordplay such as pro-/con-, and longer
> narrative reversals with an occupational or medical loophole.
>
> **great** — Compact jokes whose punchline lands instantly: one-line Q&A
> wordplay flips, absurd literal answers, anti-jokes, tautologies, meta
> refusals. "Shredded tweet." "What is orange and sounds like a parrot? A
> carrot." Exclude multi-line narratives, long conversational reversals,
> sexual innuendo, and engineer billing tales.

This person, user 23550 in the Jester dataset, likes exactly the jokes the
crowd hates. The crowd's favourite Jester joke is a 200-word story about a
US Navy ship and a Canadian lighthouse. Their favourite is the carrot.

With a generic rubric ("not funny / somewhat funny / very funny"), Jev
ordered this person's held-out jokes correctly 24 % of the time. With the
rubric above, 77 %. A nearest-neighbour filter that reads their ratings and
everyone else's got 72 %. For this one person, a paragraph of text beat
collaborative filtering.

That is the good news. This article is about why it is not enough.

## What we were trying to do

[Jev](https://typesafe.ai) is TypeSafe AI's **System One** model. It does not
generate text. You give it a *state* and a *question*, and it returns typed
values. Its `Score` primitive takes an ordered list of level descriptions,
the *criteria*, and returns a probability for each level. One call, under a
second, a fraction of a cent.

The criteria are text. Text can be optimized. So the question was:

> Can the text of a `Score` rubric, and nothing else, hold one person's taste
> well enough to recommend items to that person better than a generic rubric?

If yes, you get a recommender with an unusual shape. It scores an item it
has never seen, with no ratings on that item, for one call. It sits between
the two tools people use today: collaborative filtering, which needs ratings
on the item, and an LLM judge, which costs a hundred times more per call.
And the trained artifact is a paragraph you can read.

The optimizer is [GEPA](https://github.com/gepa-ai/gepa). It shows a
reflection model a batch of items, the person's labels, and the current
rubric's mistakes, and asks for a better rubric. DeepSeek Flash wrote the
text. Jev read it. No LLM runs at recommendation time.

## Experiment one: one person, 412 punchlines

We generated 120 joke premises with two to five punchlines each in
deliberately different styles, 412 punchlines in total. One of us labelled
every punchline bad, good or great. We split by premise, 70 train, 20
validation, 30 test, and ran GEPA.

The metric is pairwise concordance: of all punchline pairs the person
labelled differently, what share does Jev order the same way? Chance is
0.50.

| Rubric | Concordance | Top-1 hit |
|---|---|---|
| Seed, hand-written | 0.48 | 0.33 |
| Generic, three words | 0.58 | 0.46 |
| Shuffled labels (null run) | 0.56 | 0.42 |
| GEPA, 400 metric calls | **0.69** | **0.58** |

Eleven points over generic, and the null run is flat, so the lift is real.
But look at where it lives. Split the pairs by which labels they compare:

| Pair type | Generic | GEPA |
|---|---|---|
| bad vs good | 0.63 | **0.83** |
| bad vs great | 0.68 | 0.50 |
| good vs great | 0.11 | 0.11 |

The rubric learned to separate bad from good, and got slightly worse at
everything that involves great. It never learned great.
Across every rubric we tried, Jev put a mean probability of 0.07 on the
great level for bad, good and great punchlines alike. The training set's
great punchlines were "a procedural routine from another domain" and the
test set's were terse deadpan puns. Thirty-eight great labels was not enough
to describe a taste that has two mechanisms.

Train concordance was 0.675 and test 0.687. The rubric was not overfitting.
It had run out of things it could say.

## Experiment two: 7,200 people, 100 jokes

One person's labels cannot tell you whether a method works. So we moved to
[Jester](https://eigentaste.berkeley.edu/dataset/), where 7,200 people rated
the same 100 jokes on a -10 to +10 scale. A complete matrix lets you compute
things with no model calls at all.

The first thing we computed was the ceiling. For each user we split the
jokes 60 train, 40 test, binned their ratings into terciles, and asked: how
well does the crowd mean order this person's test jokes? How well does a
k-nearest-neighbour filter that reads their 60 train ratings? The difference
is the whole personal gap a text-only method could ever claim.

| Predictor, median over 7,200 users | Concordance |
|---|---|
| Crowd item mean (reads 7,199 other users) | 0.650 |
| kNN, k = 80 (reads those plus your 60 ratings) | 0.710 |

Six points. For the typical Jester user, being you instead of the crowd is
worth six points of concordance. Only 2 % of users have a gap of 20 points
or more. Humour, in this data, is mostly shared.

Then Jev with the generic three-word rubric, one call per joke, no ratings
anywhere:

| Reader | Median concordance |
|---|---|
| Jev, "not / somewhat / very funny" | **0.604** |
| Jev, our hand-written seed | 0.587 |
| Jev, a rubric written from the crowd's ten favourite and ten least favourite jokes | 0.599 |

A one-line rubric and a sub-second call gets you within five points of the
crowd mean, for anyone, on jokes nobody has rated. That is a fine result for
a generic quality filter. It is not personal.

## The personal run

We picked 60 users: the 40 whose train ratings disagreed most with the
crowd, plus 20 at random. For each, GEPA rewrote the rubric from their 60
train jokes. Then we scored the 40 test jokes. Three controls:

- **Cross-user.** Take the rubric written for user A and read it for user B.
  If long, specific text helps by itself, this will beat generic too.
- **Null run.** Shuffle the labels inside each training batch and run GEPA
  anyway, for 10 users.
- **kNN.** The ceiling that reads ratings.

| Users | Crowd | Generic | Personal | Cross-user | kNN |
|---|---|---|---|---|---|
| 40 atypical | 0.489 | 0.529 | **0.664** | 0.498 | 0.758 |
| 20 random | 0.656 | 0.568 | 0.581 | 0.521 | 0.716 |
| 10 null | 0.549 | 0.586 | 0.553 | 0.544 | 0.787 |

For the atypical 40, the personal rubric gains 11 points over generic, 30
wins to 10, p = 0.002. It beats the crowd mean by 17 points, which a generic
rubric cannot do. The cross-user rubric sits at the generic level, so the
gain is in the text written for that person, not in longer text. The null
run loses 5 points. The pipeline works.

And for the 20 random users it gains nothing. Minus two points, 8 wins to
12. The personal gain correlates 0.70 with how far the user is from the
crowd. If your taste is the crowd's taste, there is nothing personal to
learn, and the rubric stays generic.

## Why we call this negative

Three reasons, in order of weight.

**It does not beat the boring baseline.** kNN is the cheapest thing you
would actually deploy. It read the personal rubric's own training data plus
everyone else's ratings and stayed ahead for 53 of 60 users, by a median 12
points. The personal rubric closed about half the distance from generic to
kNN for the atypical users, a third for everyone. Half of a small gap is a
small number.

**It only works for people who are not like the crowd.** Which is a real
population, and the population a recommender most needs to serve. But it is
the tail. For the median user the method is a generic quality filter with
extra steps.

**The text is not portable.** We took the rubric GEPA wrote for our own
labels and handed it to DeepSeek Flash instead of Jev. With no reasoning it
helped the LLM by one point. With reasoning on, it cost it three. The rubric
encodes how Jev reads level descriptions, not a description of a person a
human or another model could use. The paragraph is readable; it is not
transferable.

There is a fourth reason, smaller. The great level never learned on our own
labels. A three-level rubric with 38 examples of the top level is asking a
paragraph to hold a taste with more structure than a paragraph has.

## What did work, and what it is for

Read the wins again, because they are not nothing.

- Jev with a one-line rubric orders unseen jokes for a random person at
  0.60, five points under a crowd mean that needed 7,199 other people's
  ratings. As a cold-start quality score, that is cheap and decent.
- DeepSeek Flash reading the same rubric got 0.50 to 0.54 on Jester, at 1.4
  to 8 times the latency and, with reasoning on, 650 tokens per joke. The
  System One model read the generic rubric *better* than the LLM did.
- For the people the crowd predicts badly, a paragraph written from 60
  ratings beat the crowd mean by 17 points with no ratings at inference.
  User 23550 and the carrot are real.

So the shape we were after does exist. It is just narrower than a
recommender: a **readable, optimizable, sub-second quality filter whose
personal component only matters for outliers**. If you have a cold-start
problem and a population with a long tail of taste, this is a tool. If you
have ratings, use them.

## What we would say in a design meeting

- **Rubric text is a real optimization target.** GEPA moved Jev from 0.58 to
  0.69 on one person and from 0.53 to 0.66 on forty. Controls confirm it.
  Text can be trained.
- **Text holds surface features, not people.** Every rubric GEPA wrote
  describes the *form* of a joke: length, Q&A shape, whether the punchline is
  literal. That is what Jev can see in one pass, and it is exactly the finding
  from [our first experiment](system-one-vs-reasoning.md): System One answers
  what the artifact makes obvious. A person's taste is partly form. The rest
  did not fit.
- **Ratings beat descriptions when you have them.** Twelve points of median
  gap to kNN, on the users chosen to favour the rubric. Do not build a
  recommender out of rubric text where a rating matrix exists.
- **Do not expect the text to travel.** Optimize for the reader you will
  deploy.

Summary in one line: **a rubric can learn what a joke looks like, and for
most people that is the crowd's taste already.**

## Caveats, and the case against this experiment

- **Forty test jokes per user.** One user's number moves by several points
  between splits. The paired statistics over 60 users are the result; any
  single row is an anecdote, including the carrot.
- **The 40 atypical users were selected.** On train jokes only, so the test
  is fair, but they are still a chosen tail. The random 20 are the honest
  sample and they gained nothing.
- **Jester is old, and the jokes with the most disagreement are dated or
  offensive.** Disagreement in this data is partly about what people will
  tolerate, not only what they find funny.
- **Nobody rated anything twice.** We have no intra-rater ceiling, on Jester
  or on our own labels. Some of the gap to kNN may be noise no method can
  close. We proposed a relabel pass and did not do it.
- **DeepSeek Flash wrote every rubric.** A stronger writer might describe a
  taste better. We did not try one, and we did not try a rubric written once
  from all the labels by a strong model.
- **Budget was 300 metric calls per user**, about 20 GEPA iterations,
  because the writer spends 12 to 16 thousand reasoning tokens per rewrite
  and one user took 21 minutes. A pilot at 600 was stopped for time. More
  budget might help the random 20. We doubt it: those users have nothing
  personal in their labels to find.
- **Terciles throw away information.** Binning each user's ratings into
  bad / good / great drops the pairs whose ratings are close. That raises
  every number in the Jester tables, kNN included, relative to the raw
  ratings. It does not change the ordering of methods.

## Try it yourself

The Jester arm is fully reproducible. Our own labels are not committed, so
the first experiment's numbers are reported but cannot be re-run from the
repository.

```bash
git clone https://github.com/naveenreddy61/jev-experiments.git
cd jev-experiments/joke-preference
uv sync
.venv/bin/python scripts/fetch_jester.py        # downloads and parses Jester
.venv/bin/python scripts/jester_signal.py       # the ceilings, no model calls

export JEV_API_KEY=... DEEPSEEK_API_KEY=...
.venv/bin/joke-pref jester-evaluate --criteria criteria/probe/a-generic.json --name generic-a-generic
.venv/bin/joke-pref jester-optimize --user 23550 --max-metric-calls 300
.venv/bin/joke-pref judge --jester --criteria criteria/probe/a-generic.json --thinking --answer score
.venv/bin/python scripts/jester_phase2_report.py
```

| path | what |
|---|---|
| `docs/jester-results.md` | every table in this article, with the paired statistics |
| `docs/jester-signal.md` | the no-model-call ceiling study |
| `src/joke_pref/adapter.py` | the GEPA adapter and the joint rewrite prompt |
| `src/joke_pref/llm_judge.py` | DeepSeek Flash in Jev's place |
| `criteria/` | the seed, generic and crowd rubrics |

Pick a Jester user whose ratings you find strange, run the optimizer, and
read the paragraph it writes. Then check whether it says anything about the
person, or only about the jokes.
