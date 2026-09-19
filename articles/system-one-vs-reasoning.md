# Does This Even Parse?

*A bake-off between a System One model and a reasoning model, on the most
boring question in computing.*

---

We asked two models one question, 360 times:

> Does this parse?

Not "is this good code". Not "what does it do". Just: will the parser accept
it, yes or no. A question with a machine-checkable answer, which is the whole
point — we never hand-labelled anything. Four real tools decided every label:

| language | the judge |
|---|---|
| Bash | `bash -n` |
| C | `gcc -std=c11 -fsyntax-only` |
| Python | `ast.parse` |
| LaTeX | `pdflatex -draftmode` |

If the tool exits 0, the answer is yes. No opinions involved.

The two contestants:

- **Jev**, a *System One* model from TypeSafe AI. It does not write an answer.
  It returns one number — a probability — and it returns it in about 345
  milliseconds (median over 360 items). There is no chain of thought, because there is no chain.
- **DeepSeek Flash**, a reasoning model. It thinks first, at length, and then
  answers. A median of 2.4 seconds per item, of which almost all is thinking.

The name "System One" is a promise about what kind of thinking the model does.
Daniel Kahneman's System 1 is the fast, automatic one: you see a face, you know
it is angry. System 2 is the slow one: you multiply 17 by 24. We wanted to find
out whether the promise holds — and, more usefully, where it stops holding.

It holds. It stops holding at a very specific place.

## Round 1: the easy stuff

We started with short snippets. A missing quote. An unclosed brace two lines
down. The sort of thing you spot while scrolling past.

| difficulty | items | Jev | DeepSeek |
|---|---|---|---|
| easy | 55 | **0.818** | 1.000 |
| medium | 71 | **0.803** | 1.000 |

DeepSeek got every single one. Jev got about 80%, in a seventh of the time,
without generating a single reasoning token.

That is a real result and it deserves to be said plainly: **for faults you can
see, Jev works.** It is not as accurate as the reasoning model, but it is fast
and cheap and it is clearly doing something. Eighty percent on a balanced
yes/no task is not luck.

Hold on to that number. It is the high-water mark.

## Round 2: hiding the fault

Then we got mean.

We wrote 160 new items, averaging 22 lines instead of 3, built on one rule:
**put the fault far away from the thing it breaks.** Open a brace at line 4 and
never close it. Redefine a LaTeX macro at the top so that a command 30 lines
later takes the wrong number of arguments. Open a subshell and close it with
the wrong bracket, 40 lines apart.

Nothing exotic. No obscure language corners. Just distance.

Half the items were broken. The other half were valid code written to *look*
broken — C11 flexible array members, Python 3.12 generics, LaTeX `\verb` with
braces inside. We wanted both failure directions available.

| difficulty | items | Jev | DeepSeek |
|---|---|---|---|
| easy | 55 | 0.818 | 1.000 |
| medium | 71 | 0.803 | 1.000 |
| hard (short) | 74 | 0.743 | 0.919 |
| **hard (long)** | **160** | **0.544** | **0.963** |

The set is balanced, 80 yes and 80 no. **Chance is 0.500.**

Jev scored 0.544.

DeepSeek barely noticed. It went from 1.000 to 0.963 and spent more time
thinking. Jev fell off a cliff.

On the LaTeX subset it is even cleaner. Twenty valid fragments, twenty broken
ones, and Jev got exactly 20 right.

**0.500. Not approximately chance. Chance.**

## What actually happened

Here is the number that explains everything else.

Across all 360 items — of which exactly 180 are valid — Jev answered **yes 278
times**.

| | says yes when it is yes | says yes when it is broken |
|---|---|---|
| Jev | 171 of 180 | **107 of 180** |
| DeepSeek | 180 of 180 | 12 of 180 |

Jev's score on valid code is superb: 171 out of 180. But that is not skill, or
not only skill. It is a model that leans hard toward yes, being graded on the
half of the test where yes is correct.

On the long hard items Jev said yes to 143 of 160. A model that simply answered
"yes" to everything, with no input at all, would have scored 0.500 on that set.
Jev scored 0.544.

That is the finding. Not "Jev is bad" — it is 0.818 on the easy set and it is
7x faster. The finding is **what Jev does when it cannot tell.** It does not
hedge, and it does not say it is unsure. It says yes, confidently, and moves
on. Two whole categories were total wipeouts: every broken medium bash script
(0 of 9), every broken long LaTeX fragment (0 of 20).

## Why this is the expected result, not a bug

To know that a brace opened on line 4 is never closed, you have to hold a
counter in your head from line 4 to line 40. That is not perception. That is
bookkeeping. It is System 2, and Jev does not have a System 2 — that is the
product, not a defect in it.

Meanwhile, what is DeepSeek doing with all that time? We measured it. Across
360 items it burned **348,335 reasoning tokens — 98% of everything it
generated.** The visible answer, `{"answer": true, "p_yes": 0.99}`, is the
other 2%.

And that reasoning is not padding. It scales with difficulty and *only* with
difficulty:

| difficulty | median reasoning tokens |
|---|---|
| easy | 160 |
| medium | 167 |
| hard (short) | 366 |
| hard (long) | 817 |

Not with length, notably. Longer artifacts do not get more thought; harder ones
do. Within each difficulty band, the correlation between how long an artifact
is and how long the model thinks about it is essentially zero.

So DeepSeek is buying its accuracy with a resource Jev structurally does not
have. This is not two models of different quality. It is two different machines.

## The funniest thing we found

The single most expensive item in the whole run cost **21,386 reasoning
tokens** on 21 lines of Python. DeepSeek ground away at it for half a minute.

The code was *fine*. It was one of our valid-but-weird items, using Python 3.12
generic syntax. The model spent an enormous amount of effort convincing itself
that strange legal code was legal — and got it right.

That pattern held. The most expensive items were mostly *valid* source, not
broken source. Recognising that nothing is wrong turns out to be harder than
finding something wrong, once "nothing is wrong" looks alarming.

Jev disposed of that same item in 322 milliseconds — and got it wrong. It said
no. One of the handful of times all day that Jev rejected something, and it
picked the item a reasoning model needed 21,386 tokens to be sure about.

## One honest complication

We re-ran DeepSeek three times on the hard set to see how stable any of this
is. It scored 0.9187, 0.9313 and 0.9437 on identical input. So a single number
here is worth about ±0.025, and anyone quoting three decimal places (including
us, earlier) is kidding themselves.

Out of that fell the nicest result in the whole project. The spread came
entirely from 11 items. The other 149 scored identically in all three runs. And
those 11 unstable items cost **3573 reasoning tokens on average, against 1451
for the stable ones.**

Long thinking does not predict a *wrong* answer. It predicts an *unstable*
one — the model is near a decision boundary and falls off different sides on
different days. That signal is available without knowing the right answer,
which makes it genuinely useful: reasoning length is a live confidence
estimate, and a better one than the confidence the model reports.

That 21,386-token item? Across the repeats it answered true, then false, then
true. On valid code.

## So what is a System One model for

Not this. Or rather: not the second half of this.

The result we would take to a design meeting:

- **Use a System One model where the answer is a property of the surface.** A
  missing quote, a malformed line, a shape that is wrong on sight. Jev is 0.818
  there and returns in about 345 ms. That is a real tool.
- **Do not use it where the answer requires tracking state across distance.**
  Not because it is inaccurate there, but because it is *at chance* there, and
  it will tell you yes with high confidence while being at chance.
- **The second failure is worse than the first.** A model that is wrong is
  manageable. A model that is wrong and confident, on exactly the inputs you
  cannot check by eye, is a trap.

The obvious shape is a cascade: Jev first, escalate the uncertain cases to a
reasoning model. We have not built it, and our data does not yet say where the
threshold goes — Jev's probabilities are poorly calibrated on precisely the
hard items where you would need them. That is the next experiment.

For now the summary fits on one line: **System One answers what the artifact
makes obvious. Everything else needs something that can count.**

---

### Small print

- 360 items, 4 tasks, every label from a real parser or compiler. No
  hand-labelling, ever.
- Jev numbers are a single run per item. DeepSeek's hard-set numbers are a mean
  of three. Jev was never repeated, so its figures have no error bar and should
  be read as less well established than DeepSeek's, not more.
- Jev was asked one locked question per item with no extra criteria. Supplying
  criteria is a different experiment and might well go better for it.
- "Valid code that looks broken" is half of the hard set by construction. That
  is a deliberate stress test, not a sample of code in the wild.
- The harness, the data, the raw predictions and the full reasoning traces are
  all in [`parsing-experiments/`](../parsing-experiments/). Every number above is
  recomputable from it.
