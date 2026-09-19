# Does This Even Parse?

*A playful experiment with a model that does not think, on a job that
sometimes needs thinking.*

---

## The hook

Here is a 22-line shell script. You have one third of a second. Does it parse?

```bash
#!/usr/bin/env bash
watch_queue() {
    local q=$1
    local idle=0
    while :; do
        local depth
        depth=$(redis-cli llen "$q" 2>/dev/null || echo 0)
        if (( depth > 0 )); then
            idle=0
            echo "queue $q depth=$depth"
        else
            idle=$(( idle + 1 ))
            if (( idle > 12 )); then
                echo "queue $q idle for 60s"
                break
            fi
        fi
        sleep 5
    fi
}

watch_queue jobs
```

It does not. The `while` on line 5 is closed with `fi` on line 19 instead of
`done`. To see that, you have to remember what opened fourteen lines earlier.
Nothing on line 19 looks wrong by itself.

That is the whole experiment in one script. Some faults are visible on the
surface. Some are only visible if you keep a running tally. We wanted to know
which kind a model with no scratchpad can catch.

## What Jev is

[Jev](https://typesafe.ai) is a model from TypeSafe AI. They call it a
**System One** model, and their docs describe the category as "a class of AI
models built to make fast, structured decisions that software can use
directly."

The important part is what it does *not* do. It does not generate text. There
is no answer to read, no explanation, no chain of thought. You give it some
input, which TypeSafe calls the *state*, and one or more *questions*. It
returns typed values. A yes/no question comes back as a single probability:

```json
{"type": "noul", "noul": 0.92}
```

That is the entire response. No boolean, no separate confidence score. The
number is the answer and the confidence at the same time. TypeSafe says the
probabilities are "optimized against outcomes to reflect uncertainty," which is
a promise that 0.92 means something closer to 92% than to "very yes."

The name is a nod to Daniel Kahneman. System 1 is the fast, automatic mode of
thought: you see a face and you know it is angry. System 2 is the slow,
effortful mode: you multiply 17 by 24. TypeSafe's examples are all System 1
jobs. Is this support ticket about billing? Is the customer angry? Does this
résumé mention Python?

We wanted to find the edge. Where does "obvious from looking" stop, and
"you have to work it out" begin? And what does a System One model do when it
crosses that line?

## Why parsing

We needed a question with three properties:

1. **A machine decides the answer.** No human labels, no judgement calls, no
   arguing about the gold standard.
2. **The same question spans both modes.** Some answers should be obvious at a
   glance; others should need bookkeeping.
3. **It is cheap to make hundreds of examples.**

"Does this parse?" has all three. A missing closing quote is visible from
across the room. An unclosed brace forty lines from where it opened is not.
And the judge is a real parser that either accepts the input or does not.

We used four languages and four judges:

| language | the judge | yes means |
|---|---|---|
| Bash | `bash -n` | exit code 0 |
| C | `gcc -std=c11 -fsyntax-only` | exit code 0 |
| Python | `ast.parse` | no `SyntaxError` |
| LaTeX | `pdflatex -draftmode` | exit code 0 |

Every label in the dataset came from one of those tools. Not a single item was
labelled by hand.

## The setup

**The items.** 360 in total, half valid and half broken, 90 per language.
They come in two batches.

The first batch is 200 short snippets, tagged easy, medium or hard by
construction. A missing quote. A stray `}`. A `\frac` with one argument.
Median length: 7 lines.

The second batch is 160 longer items built on one rule: **put the fault far
from the thing it breaks.** Open a brace early and never close it. Close a
`while` with `fi`. Redefine a LaTeX macro at the top so a call thirty lines
later has the wrong number of arguments. Median length: 22 lines.

Half of that second batch is a trap in the other direction. It is valid code
written to look broken. Python 3.12 generic syntax that most people have never
seen. C11 flexible array members inside anonymous unions. A LaTeX `verbatim`
block full of unbalanced-looking build output. We wanted both ways of being
wrong to be available.

**The question.** One fixed sentence per language, never changed:

> Does this parse as bash?

**The contestants.**

*Jev* got the artifact as its state and that sentence as its question. Nothing
else. The full call is this:

```python
from typesafe_sdk import TypeSafeClient, Noul

response = client.system_one(
    artifact,
    {"parses": Noul(instructions="Does this parse as bash?")},
    model="jev-latest",
)
p_yes = response.answers["parses"].noul   # a float in [0, 1]
```

We call it a yes if the number is 0.5 or above. That threshold is our choice,
not the model's.

*DeepSeek Flash* is a reasoning model. It got a short rubric, the artifact,
and an instruction to reply in JSON with an answer and a probability. It
thinks first, at whatever length it likes, and then answers. We logged every
reasoning trace.

Neither model got a parser. Neither could run anything. Both were reading
code and guessing.

## Round one: faults you can see

The 126 easy and medium items first.

| difficulty | items | Jev | DeepSeek |
|---|---|---|---|
| easy | 55 | 82% | 100% |
| medium | 71 | 80% | 100% |

DeepSeek got every single one. Jev got four out of five.

Four out of five is a real result. Jev answered in a median of 344
milliseconds. DeepSeek took 2.4 seconds, and on one item it took 77. Jev
generated nothing, thought about nothing, and was right most of the time on
faults a human spots while scrolling past.

Hold on to that 80%. It is the high-water mark.

## Round two: faults you have to count

Now the 160 long items, where the fault is far from the damage.

| difficulty | items | Jev | DeepSeek |
|---|---|---|---|
| hard, short | 74 | 74% | 92% |
| **hard, long** | **160** | **54%** | **96%** |

The long set is exactly half valid and half broken, so a coin scores 50%.

Jev scored 54%.

DeepSeek barely noticed the change. Jev fell off a cliff. On the LaTeX part of
the long set, twenty valid and twenty broken, Jev got exactly twenty right.
That is not "close to chance". It is chance.

## What Jev actually does when it cannot tell

This is the part worth remembering.

Jev does not become uncertain on the hard items. It does not drift toward 0.5.
It says **yes**.

Across all 360 items, of which exactly 180 are valid, Jev said yes 278 times.
On the long hard set it said yes to 143 of 160. On the long LaTeX items it
said yes to all forty, broken ones included, with an average probability
around 0.8.

Look at it as two questions instead of one:

| | valid items: how many did it accept? | broken items: how many did it catch? |
|---|---|---|
| Jev | 171 of 180 | **73 of 180** |
| DeepSeek | 180 of 180 | 168 of 180 |

Jev is excellent at accepting valid code. But a model that says yes to
everything is also excellent at accepting valid code. The 171 is mostly the
yes bias being graded on the half of the exam where yes is the right answer.

The intuition we ended up with: Jev is a pattern detector. When the input
contains a pattern it has learned to associate with "broken", it fires. When
the input contains no such pattern, it reports "looks fine". A `while` closed
by `fi` fifteen lines later does not look like anything. Every line, on its
own, is a perfectly normal line of bash.

Take the script at the top of this article. Jev gave it 0.93. Then again, so
did DeepSeek. That one fooled everybody. But swap the fault for an extra `fi`
that closes nothing, and DeepSeek catches it while Jev still says 0.61 yes.

## Why this is the expected result, not a bug

To know that a brace opened on line 4 is never closed, you have to hold a
counter from line 4 to line 40. That is not perception. That is bookkeeping.
It is System 2 work, and Jev has no System 2. That is the product, not a
defect in it.

DeepSeek does have one, and we can watch it run. Across the 360 items it
produced about 354,000 tokens. **98% of them were reasoning.** The visible
answer, `{"answer": true, "p_yes": 0.99}`, is the other 2%.

That reasoning is not padding, either. Easy items got a couple of hundred
tokens of thought. Long hard items got about eight hundred at the median, and
a few got tens of thousands. The model spends more when the question is
harder, and it does not spend more just because the input is longer.

So the two models are not two grades of the same thing. One looks. The other
counts. On a task that needs counting, only one of them can play.

## The funniest item in the set

The single most expensive item in the whole run was 21 lines of Python:

```python
type Pair[A, B] = tuple[A, B]
type Handler[**P, R] = Callable[P, R]

class Buffer[*Ts]:
    def __init__(self, *items: *Ts) -> None:
        ...
```

It is valid Python 3.12. It also looks like a keyboard accident. DeepSeek spent
**21,386 reasoning tokens** and about half a minute talking itself into
"yes, this is legal." It got it right.

Jev looked at it for 322 milliseconds and said no, with probability 0.25. One
of the few times in the whole experiment that Jev rejected anything, and it
picked the item that a reasoning model needed twenty thousand tokens to
accept.

Which is, if you think about it, exactly what a pattern detector should do
with code that pattern-matches to "wrong".

## One honest complication about the reasoning model

We ran DeepSeek three more times on the long hard set, same inputs, to see how
stable it is. It scored 92%, 93% and 94%.

The spread came entirely from 11 items. The other 149 got the same answer all
three times. And those 11 wobbly items cost, on average, two and a half times
the reasoning tokens of the stable ones.

So a long chain of thought does not mean the answer is wrong. It means the
model is standing near the edge of a decision and could fall either way. That
21,386-token Python item? Across the repeats it answered yes, then no, then
yes. On valid code.

Jev, for what it is worth, has no such tell. Its probability on the hard items
is confidently high whether it is right or wrong.

## What we would say in a design meeting

- **A System One model is for answers that are a property of the surface.** A
  missing quote. A malformed line. A ticket that is obviously about billing.
  Jev is around 80% there and answers in a third of a second with no
  generated text to parse. That is a real tool.
- **It is not for answers that require state tracked across distance.** Not
  because it gets them somewhat wrong, but because it is at chance, and it
  reports high confidence while at chance.
- **The second failure is the dangerous one.** A model that is wrong is
  manageable. A model that is wrong and sure, on exactly the inputs a human
  cannot check at a glance, is a trap.

The natural next shape is a cascade: Jev first, escalate anything it is unsure
about to a reasoning model. We have not built it, and our data does not say
where the threshold should go, because Jev is not unsure on the items where
you would need it to be.

Summary in one line: **System One answers what the artifact makes obvious.
Anything that needs counting needs something that can count.**

## Caveats, and the case against this experiment

We had fun. That does not make it a fair fight. Read these before quoting a
number.

- **This is not what Jev is for.** TypeSafe's examples are customer messages,
  résumés, and PII detection. We pointed a ticket-classifier at a compiler's
  job. The interesting finding is *where* the edge is, not that an edge
  exists.
- **Jev got eight words. DeepSeek got a rubric.** The Noul primitive has an
  optional `criteria` field to describe what yes and no mean. We did not use
  it, because we wanted the same locked question everywhere. A criteria
  variant might do better. It is a separate experiment we have not run.
- **Jev's numbers are one run per item.** DeepSeek's hard-set figure is a mean
  of three, and those three spread by two and a half points. Jev was never
  repeated. So Jev's column is the *less* well-measured of the two, not the
  more. The 54% could plausibly be 51% or 57%. It could not plausibly be 80%.
- **The hard set is adversarial by construction.** Half of it is valid code
  written to look broken. That is a stress test, not a sample of code in the
  wild. Real code is mostly valid and mostly boring, and on that distribution
  a say-yes bias looks a lot better.
- **Both models lean toward yes.** DeepSeek missed 12 broken items and zero
  valid ones. It is a much smaller lean, but it is the same lean. One bash
  fault type, a `[[ ... ]` with the wrong closer, fooled Jev every time and
  DeepSeek in most runs.
- **The 0.5 threshold is ours, and it matters.** Jev returns a probability
  and we chose where to cut. On the long hard set, valid items average about
  0.83 and broken ones about 0.71, so there *is* some signal in the number.
  Cutting at 0.8 instead of 0.5 lifts Jev from 54% to about 66% on that set.
  That is the fairest number to hold against the "chance" headline. It is
  still thirty points behind the reasoning model, and you would only know to
  cut at 0.8 by peeking at the answers.
- **One reasoning model, and not the strongest one.** DeepSeek Flash is the
  fast, cheap end of that family. A bigger model would presumably close the
  remaining 4%, which does not change the story.
- **One author for the hard items.** Another author would find different
  faults, and probably a different LaTeX-shaped hole.
- **We measured time, not money.** Jev is about seven times faster at the
  median. We did not compute a per-item cost for either API.

## Try it yourself

Everything is in the repo, including every raw prediction and every DeepSeek
reasoning trace. No API key is needed to re-score the committed runs.

```bash
git clone https://github.com/naveenreddy61/jev-experiments.git
cd jev-experiments/parsing-experiments

python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[jev]'

# The raw counts behind every table in this article
python scripts/counts.py
```

To make new calls, export `JEV_API_KEY` or `DEEPSEEK_API_KEY` and run:

```bash
python -m jev_eval run --provider jev --data data/hard --out results/my-jev-run
python -m jev_eval run --provider deepseek-flash --data data/hard --out results/my-ds-run
python -m jev_eval score results/my-jev-run
```

To re-check the labels, you need the four judges installed locally:

```bash
python -m jev_eval validate data/hard
```

Where things live:

| path | what |
|---|---|
| `data/seed/` | the 200 short items, one JSONL file per language |
| `data/hard/` | the 160 long items |
| `results/` | every run: predictions, latencies, reasoning traces |
| `docs/` | the detailed write-ups, with the numbers we left out here |
| `src/jev_eval/runners/jev.py` | the Jev call, all twenty lines of it |

Each item is one JSON line with the artifact, the question, the oracle's
answer, and a tag naming the fault if there is one. Pick any item, read it, and
see if you can beat 54%.
