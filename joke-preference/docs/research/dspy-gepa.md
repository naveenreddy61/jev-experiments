# DSPy 3.x and GEPA: a guide for the Jev criteria-optimization task

## Purpose

This report is for an engineer who wants to find good "criteria" text. Jev, an
external non-LLM scoring model, reads a joke and the criteria text. Jev returns
one probability in [0, 1] for a question about the joke. The engineer has a set
of jokes. Each joke has a premise and a punchline. Each joke has a user label:
bad, good, or great. The goal is this: find criteria text so that the Jev
probability predicts the user label.

This report explains DSPy 3.x basics, the `dspy.GEPA` optimizer, and the
standalone `gepa` package. It gives a full code skeleton for a custom `gepa`
adapter. The adapter calls Jev, not an LLM, inside `evaluate()`. It recommends
a metric for the three-level label. It lists budgets, pitfalls, and a
comparison with MIPROv2 and with simple search methods.

---

## 1. DSPy basics needed for this task

DSPy is a framework. It treats prompts as compiled output, not as hand-written
text. You write a typed contract (a signature). You write a strategy (a
module). You write a metric. An optimizer then rewrites the prompt text to
raise the metric score.

For the Jev task, you likely do **not** need a DSPy `Signature` or `Module` at
all, because the "program" is a plain text string (the criteria), not an LLM
call. But the concepts below still apply, because `dspy.GEPA` uses them under
the hood, and because the standalone `gepa` package reuses the same ideas
(metric with feedback, trainset, valset).

### 1.1 Signature

A signature is a typed input/output contract for one LM call.

```python
import dspy

class WriteCriteria(dspy.Signature):
    """Write criteria text that Jev can use to score a joke."""
    joke_premise: str = dspy.InputField()
    joke_punchline: str = dspy.InputField()
    criteria: str = dspy.OutputField(desc="Free-text criteria for the Jev scorer")
```

You do not need this signature for the Jev task if the criteria text is your
only optimization target. You need it only if you want an LLM module to
*produce* the criteria as one step in a larger DSPy program.

### 1.2 Module and Predict

A module is a strategy that fulfills a signature. `dspy.Predict` makes one LM
call.

```python
predict = dspy.Predict("question -> answer")
result = predict(question="What is 2+2?")
print(result.answer)
```

### 1.3 dspy.LM configuration

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)
```

`dspy.LM` wraps LiteLLM. The string format is `<provider>/<model-name>`. You
set a global default with `dspy.configure(lm=...)`. You can override it per
call with the `lm=` keyword, or with `dspy.context(lm=...)` for a block of
code.

For the Jev task, `dspy.GEPA` needs an LM only for the **reflection** step (it
does not need an LM to run Jev, because Jev is an external API, not an LM
call). See section 5.

### 1.4 Example

`dspy.Example` holds one training row. It marks which fields are inputs.

```python
example = dspy.Example(
    premise="Why did the chicken cross the road?",
    punchline="To get to the other side.",
    label="bad",
).with_inputs("premise", "punchline")
```

`.with_inputs(...)` is required. Without it, DSPy may pass the label itself
into the program by mistake.

### 1.5 Evaluate

`dspy.Evaluate` runs a program over a list of examples and averages the
metric.

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(devset=devset, metric=my_metric, num_threads=8)
score = evaluator(my_program)
```

For the Jev task, you likely will not call `dspy.Evaluate` directly, because
you are not using a `dspy.Module` as the program. The standalone `gepa`
package has its own evaluation loop, driven by your adapter's `evaluate()`
method (section 3).

### 1.6 Metrics with feedback

A plain DSPy metric returns a float or a bool:
`metric(example, pred, trace=None) -> float`.

A **GEPA-compatible** metric returns a score **and** a feedback string. In
`dspy.GEPA`, you return this as `dspy.Prediction(score=..., feedback=...)`:

```python
def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score = 1.0 if gold.label == pred.label else 0.0
    feedback = "Correct." if score == 1.0 else (
        f"Expected {gold.label!r}, got {pred.label!r}."
    )
    return dspy.Prediction(score=score, feedback=feedback)
```

The feedback text is the whole point of GEPA. GEPA's reflection LM reads this
text to decide how to change the prompt (or, in the Jev task, the criteria
text). A feedback string like "wrong" is close to useless. A feedback string
that names the failure mode is valuable — for example: "Jev gave 0.82
(near-great) for a joke the user rated bad. The criteria may reward wordplay
too strongly and not check for coherence."

---

## 2. dspy.GEPA: constructor, metric contract, and workflow

`dspy.GEPA` is DSPy's wrapper around the standalone `gepa` package's
optimizer (see section 3). It optimizes the natural-language instructions
inside a `dspy.Module` — for example, the docstring of a `dspy.Signature`.
Internally, `compile()` builds a `DspyAdapter` (an implementation of
`GEPAAdapter`, see section 3) and calls `gepa.optimize(...)`.

Source: `dspy/teleprompt/gepa/gepa.py` in `stanfordnlp/dspy`
(https://github.com/stanfordnlp/dspy), and
https://dspy.ai/api/optimizers/GEPA/overview/.

### 2.1 Constructor arguments

```python
import dspy

optimizer = dspy.GEPA(
    metric=gepa_metric,                    # required: GEPAFeedbackMetric callable
    auto="medium",                         # "light" | "medium" | "heavy" | None
    max_metric_calls=None,                 # int, exact budget in metric calls
    reflection_lm=dspy.LM("openai/gpt-4o"),# strong model for reflection (required for quality)
    reflection_minibatch_size=3,            # size of minibatch shown to reflection LM per step
    candidate_selection_strategy="pareto",  # "pareto" | "current_best"
    skip_perfect_score=True,                # skip a candidate that already scores perfect
    use_merge=False,                        # merge lineages of two good candidates
    max_merge_invocations=5,                # cap on merge attempts
    num_threads=8,                          # parallel evaluation threads
    track_stats=True,                       # populate compiled.detailed_results
    track_best_outputs=False,               # also keep the best raw output per val instance
    log_dir="./gepa_logs",                  # where GEPA writes run artifacts
    seed=0,                                 # RNG seed, for reproducibility
    failure_score=0.0,                      # score assigned on an exception/format failure
    perfect_score=1.0,                      # the score value GEPA treats as "perfect"
    add_format_failure_as_feedback=True,    # feed format failures back to the reflection LM
    warn_on_score_mismatch=True,            # sanity check between aggregate and per-item scores
)
```

Notes on the arguments that matter most for this task:

- **`metric`** — must accept `(gold, pred, trace, pred_name, pred_trace,
  program_trace=None)` and return either a float or a
  `dspy.Prediction(score=..., feedback=...)` (a `ScoreWithFeedback`). `pred_name`
  tells you which sub-module/predictor is being scored, useful only when your
  program has more than one predictor.
- **`auto`** — sets `max_metric_calls` automatically from trainset/valset size
  and the number of predictors in your program. Use `auto` for a first run.
  Use an explicit `max_metric_calls` once you know your budget.
- **`reflection_lm`** — this is the model that reads feedback and rewrites
  text. It should be a strong model (for example, GPT-4o, Claude Sonnet, or
  similar). It is called far less often than the task itself runs, so its cost
  usually does not dominate.
- **`candidate_selection_strategy`** — `"pareto"` keeps a frontier of
  candidates, each of which is best on at least one validation example, and
  samples from that frontier before each mutation. `"current_best"` only
  evolves the single top candidate. `"pareto"` gives more diversity and is the
  default; it usually finds better instructions on varied data such as jokes.
- **`track_stats=True`** — after `compile()`, the returned program has
  `.detailed_results`, an instance of `DspyGEPAResult`, with: `candidates`
  (all proposed candidates), `parents` (lineage), `val_aggregate_scores`,
  `val_subscores`, `per_val_instance_best_candidates`, `discovery_eval_counts`,
  `best_idx`, `best_candidate`, `total_metric_calls`, `num_full_val_evals`,
  `log_dir`, `seed`. Set `track_best_outputs=True` as well if you also want
  `best_outputs_valset`.

### 2.2 Compile and inspect

```python
compiled = optimizer.compile(my_program, trainset=trainset, valset=valset)

# The evolved instruction text (for a single-predictor program):
print(compiled.predictors()[0].signature.instructions)

# Full run detail (requires track_stats=True):
result = compiled.detailed_results
print(result.best_idx, result.val_aggregate_scores[result.best_idx])
print(result.candidates[result.best_idx])
```

### 2.3 Why dspy.GEPA is a poor direct fit for the Jev task

`dspy.GEPA` optimizes the instructions of `dspy.Predict`/`dspy.ChainOfThought`
predictors *inside a DSPy program*. Its adapter (`DspyAdapter`) runs your
`dspy.Module`, which means it runs an LM at some point to fill in output
fields. In the Jev task, the thing you evaluate on each candidate is **not**
an LM call. It is a call to an external, non-LLM scoring model (Jev). There is
no LM in the loop that produces the "prediction" — Jev directly returns a
probability given the joke and the criteria text.

You *can* force this into DSPy by wrapping the criteria in a one-field
`dspy.Signature` whose only "predictor" is a trivial pass-through, but this
adds an artificial layer for no benefit. The standalone `gepa` package (next
section) was built for exactly this situation: optimizing an arbitrary text
artifact against an arbitrary evaluator, with no LM required in the evaluation
path. Recommendation: use the standalone `gepa` package, not `dspy.GEPA`, for
this task.

---

## 3. The standalone gepa package

Source: https://github.com/gepa-ai/gepa. Install:

```bash
pip install gepa
```

The latest released version, as of September 2026, was **`gepa==0.1.4`**
(released 2026-07-15), from https://pypi.org/project/gepa/. Check
`pip index versions gepa` before you start a real run, because this project
releases often.

### 3.1 gepa.optimize()

```python
from gepa import optimize

result = optimize(
    seed_candidate=seed_candidate,   # dict[str, str]: component name -> text
    trainset=trainset,               # list[DataInst] used for reflective updates
    valset=valset,                   # list[DataInst] used to track Pareto scores
    adapter=my_adapter,              # your GEPAAdapter implementation
    reflection_lm="openai/gpt-4o",   # or a LanguageModel object
    max_metric_calls=300,            # evaluation budget
    candidate_selection_strategy="pareto",  # or "current_best", "epsilon_greedy", "top_k_pareto"
    skip_perfect_score=True,
    perfect_score=1.0,
    reflection_minibatch_size=3,
    use_merge=False,
    max_merge_invocations=5,
    run_dir="./gepa_run",
    seed=0,
    display_progress_bar=True,
    raise_on_exception=True,
    track_best_outputs=True,
)

print(result.best_candidate)   # dict[str, str] — the evolved criteria text
```

The **seed candidate** is a dict, not a single string, because GEPA supports
programs with more than one text component (for example, a system prompt and
a set of tool docstrings). For this task, use one key, for example
`{"criteria": "<your starting criteria text>"}`.

Key parameters not present in `dspy.GEPA`:

- **`task_lm`** — only used when you pass **no** adapter (GEPA then falls back
  to `DefaultAdapter`, a single-turn "optimize a system prompt for an LLM"
  adapter). You pass a **custom adapter** instead, so you do not set
  `task_lm`. Jev, not an LM, executes the "program."
- **`frontier_type`** — `"instance"` by default; the Pareto frontier is
  computed per validation instance (an example is on the frontier if some
  candidate is best on it).
- **`batch_sampler`** — `"epoch_shuffled"` by default; controls how
  minibatches are drawn from `trainset` for reflection steps.

### 3.2 The GEPAAdapter protocol

`GEPAAdapter` is a `Protocol` with two required methods.

```python
from gepa.core.adapter import GEPAAdapter, EvaluationBatch

class GEPAAdapter(Protocol[DataInst, Trajectory, RolloutOutput]):

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:
        ...

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, RolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        ...
```

- **`evaluate`** runs the current candidate (here: a criteria string) against
  a batch of examples (here: jokes with labels). It returns an
  `EvaluationBatch`, a dataclass with: `outputs` (raw per-example output —
  here, Jev's probability), `scores` (a float per example — your metric),
  `trajectories` (optional per-example trace data, used only by
  `make_reflective_dataset`, needed when `capture_traces=True`),
  `objective_scores` (optional, for multi-objective setups), and
  `num_metric_calls`.
- **`make_reflective_dataset`** turns the trajectories from one `evaluate()`
  call into a JSON-serializable list of records per component. GEPA's
  reflection LM reads these records — typically with the keys `"Inputs"`,
  `"Generated Outputs"`, and `"Feedback"` — to propose a new candidate text.

`DefaultAdapter` is the built-in adapter for the simplest case: a single
system-prompt string is sent to an LLM (`task_lm`), and the LLM's answer is
compared against a gold label. It does not fit this task, because Jev is not
an LLM and does not take the criteria as a "system prompt" in a chat call —
it takes the criteria and the joke and returns a probability directly through
its own API.

### 3.3 Worked adapter for the Jev scorer

This is the central code of this report. The "program" GEPA evolves is a
single text component, `"criteria"`. `evaluate()` calls Jev, not an LLM.
`make_reflective_dataset()` gives the reflection LM the joke, the user label,
Jev's probability, and a plain-language diagnosis of the gap.

```python
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from gepa.core.adapter import GEPAAdapter, EvaluationBatch


# ---- 1. Data types -------------------------------------------------------

LABEL_TO_TARGET = {"bad": 0.15, "good": 0.55, "great": 0.9}
LABEL_ORDER = {"bad": 0, "good": 1, "great": 2}


@dataclass
class JokeExample:
    premise: str
    punchline: str
    label: str          # "bad" | "good" | "great"


@dataclass
class JokeTrajectory:
    premise: str
    punchline: str
    label: str
    probability: float
    residual: float      # probability - target, signed
    diagnosis: str        # plain-text explanation of the gap


# ---- 2. Jev call (external, non-LLM) ------------------------------------

def call_jev(premise: str, punchline: str, criteria: str) -> float:
    """
    Calls TypeSafe AI's Jev with the joke text and the criteria question.
    Returns a probability in [0, 1]. Replace this body with the real
    Jev client call; keep the signature stable so the adapter below
    does not change.
    """
    from jev_client import JevClient   # placeholder import; use the real client

    client = JevClient()
    response = client.score(
        text=f"Premise: {premise}\nPunchline: {punchline}",
        question=criteria,
    )
    return float(response.probability)


# ---- 3. The adapter -------------------------------------------------------

class JevCriteriaAdapter(GEPAAdapter[JokeExample, JokeTrajectory, float]):
    """
    Optimizes a single free-text component, "criteria", so that
    Jev's probability, given a joke and this criteria text, tracks
    the user's bad/good/great label.
    """

    def evaluate(
        self,
        batch: list[JokeExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[JokeTrajectory, float]:
        criteria = candidate["criteria"]

        outputs: list[float] = []
        scores: list[float] = []
        trajectories: list[JokeTrajectory] | None = [] if capture_traces else None

        for ex in batch:
            prob = call_jev(ex.premise, ex.punchline, criteria)
            outputs.append(prob)

            target = LABEL_TO_TARGET[ex.label]
            residual = prob - target
            score = 1.0 - min(abs(residual), 1.0)   # closeness to the label's target band; see section 4
            scores.append(score)

            if capture_traces:
                diagnosis = self._diagnose(ex, prob, residual)
                trajectories.append(JokeTrajectory(
                    premise=ex.premise,
                    punchline=ex.punchline,
                    label=ex.label,
                    probability=prob,
                    residual=residual,
                    diagnosis=diagnosis,
                ))

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[JokeTrajectory, float],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        records = []
        for traj, score in zip(eval_batch.trajectories, eval_batch.scores):
            records.append({
                "Inputs": {
                    "premise": traj.premise,
                    "punchline": traj.punchline,
                    "user_label": traj.label,
                },
                "Generated Outputs": {
                    "jev_probability": round(traj.probability, 3),
                },
                "Feedback": (
                    f"User rated this joke '{traj.label}'. Jev returned "
                    f"probability {traj.probability:.2f} for the current criteria. "
                    f"{traj.diagnosis} Score for this example: {score:.2f}."
                ),
            })
        # One component in this task: "criteria".
        return {name: records for name in components_to_update}

    @staticmethod
    def _diagnose(ex: JokeExample, prob: float, residual: float) -> str:
        if abs(residual) < 0.1:
            return "This is well-calibrated; keep whatever the criteria rewards here."
        if residual > 0:
            return (
                f"Jev over-scored a '{ex.label}' joke. The criteria may be too "
                "lenient, or it may reward a surface feature (e.g. wordplay, "
                "length) that this joke has despite being weak."
            )
        return (
            f"Jev under-scored a '{ex.label}' joke. The criteria may be missing "
            "a quality this joke has (e.g. surprise, timing, coherence) or may "
            "be too strict about an irrelevant feature."
        )


# ---- 4. Run GEPA ----------------------------------------------------------

from gepa import optimize

trainset = [JokeExample(premise=p, punchline=pl, label=l) for p, pl, l in train_rows]
valset = [JokeExample(premise=p, punchline=pl, label=l) for p, pl, l in val_rows]

result = optimize(
    seed_candidate={"criteria": (
        "Does this joke land? Judge whether the punchline resolves the "
        "premise's setup with genuine surprise, without being mean-spirited "
        "or relying on a cliche."
    )},
    trainset=trainset,
    valset=valset,
    adapter=JevCriteriaAdapter(),
    reflection_lm="anthropic/claude-sonnet-4-5",
    max_metric_calls=600,
    candidate_selection_strategy="pareto",
    reflection_minibatch_size=4,
    track_best_outputs=True,
    display_progress_bar=True,
    run_dir="./gepa_jev_run",
    seed=0,
)

print(result.best_candidate["criteria"])
```

Points worth stressing:

- Jev is called inside `evaluate()`, once per example per candidate. No LM
  call happens there. The **only** LM call in the whole run is the reflection
  step, driven by `reflection_lm`.
- `capture_traces=True` is passed by GEPA itself when it needs trajectories
  for reflection; the adapter must honor that flag and skip trajectory
  construction when it is `False`, to save time on plain evaluation passes
  (for example, full-valset scoring).
- `make_reflective_dataset` must return **JSON-serializable** records. Do not
  put raw Python objects with non-trivial `__repr__` in there; stick to
  strings, numbers, and plain dicts/lists.
- If Jev's client call can fail (timeout, malformed input), catch the
  exception inside `evaluate()` and assign a `failure_score` (for example
  0.0) rather than letting the whole run crash, unless you set
  `raise_on_exception=False` at the `optimize()` call and want GEPA itself to
  handle it.

---

## 4. Designing the metric for bad / good / great vs a [0,1] probability

The label is ordinal with three levels. Jev returns a continuous probability.
You need a metric that (a) rewards Jev probabilities that separate the three
levels in the right order, and (b) gives GEPA's reflection LM a clear,
per-example textual reason for any mismatch.

### 4.1 Options considered

1. **Fixed target bands per label, then closeness score.** Map each label to
   a target probability (for example bad → 0.15, good → 0.55, great → 0.9)
   and score `1 - |prob - target|`. Simple, but it hard-codes a guess about
   where each label "should" sit, and Jev's probability scale may not be
   linear or symmetric around these points.

2. **Global rank correlation (Spearman) over the whole batch.** Compute
   Spearman's rho between Jev's probabilities and the ordinal label ranks
   across all examples in the evaluation batch, and use that single number as
   every example's score. This is attractive because it only asks GEPA to get
   the *order* right, not exact probability values, which matches the actual
   requirement ("Jev's probability predicts the label"). The problem: it is a
   **batch-level** statistic, not a per-example score, so `make_reflective_dataset`
   has less to say about any single joke, and GEPA's per-instance Pareto
   frontier (which is built from per-example scores) degrades to one score
   copied across the whole batch.

3. **Pairwise ranking accuracy.** For every pair of examples with different
   labels, score 1 if Jev ranks the higher-labeled joke above the lower-labeled
   one, else 0; average over all pairs that include this example. This gives
   a genuine per-example score (each example's score is its average over the
   pairs it takes part in), it directly measures the property you want
   ("predicts the label" mainly means "orders jokes the way the user would"),
   and it still supports rich per-example feedback ("this bad joke, paired
   against this great joke, was ranked backward by Jev").

4. **Calibrated-threshold accuracy.** Fit (or hand-pick) two thresholds that
   split [0,1] into bad/good/great, then score exact-bucket accuracy. Easy to
   explain, but brittle: it discards the continuous signal that GEPA and the
   reflection LM could use to understand *how far off* Jev was, and thresholds
   picked on a train set of ~80-100 examples will move around a lot between
   optimizer runs.

### 4.2 Recommendation: per-example pairwise ranking score, with rich feedback

Recommended metric: **for each example, sample or use a fixed set of
"reference" examples with a different label, compute the fraction of those
pairs Jev ranks correctly, and use that fraction as the example's GEPA score**.
Combine it with a smaller closeness term against a coarse target band, and
always attach a full-sentence diagnosis. Reasons:

- It is a genuine per-example score, so `dspy.GEPA`'s and `gepa`'s Pareto
  frontier (built per validation instance) stays meaningful — some candidate
  criteria will be "best" on jokes near the bad/good boundary, others on
  jokes near good/great, and GEPA can combine those separately-learned
  lessons via `use_merge` or via the reflection minibatch.
- It matches the actual goal stated by the user — "Jev's probability
  predicts the label" is a ranking/ordering claim, not a claim about the
  exact numeric value 0.15 or 0.9.
- It is robust to a shift in Jev's overall probability scale (if Jev tends to
  return everything between 0.3 and 0.7, a fixed-target metric like option 1
  would score every example poorly and give GEPA no useful gradient; a
  ranking metric still works).
- It degrades gracefully with only 80-100 examples, because pairwise
  comparisons multiply the effective number of comparisons (with three
  labels, roughly 2/3 of all pairs are usable), whereas Spearman needs the
  whole batch at once and calibrated thresholds need enough examples per
  bucket to be stable.

A concrete per-example score:

```python
def pairwise_score(this_prob, this_label, other_probs_labels) -> float:
    wins, total = 0, 0
    for other_prob, other_label in other_probs_labels:
        if LABEL_ORDER[other_label] == LABEL_ORDER[this_label]:
            continue
        total += 1
        higher_label_wins = (
            (LABEL_ORDER[this_label] > LABEL_ORDER[other_label] and this_prob > other_prob)
            or (LABEL_ORDER[this_label] < LABEL_ORDER[other_label] and this_prob < other_prob)
        )
        wins += 1.0 if higher_label_wins else 0.5 if this_prob == other_prob else 0.0
    return wins / total if total else 1.0
```

Blend this with a small closeness term (weight roughly 0.7 pairwise / 0.3
closeness) if you also want the raw probability scale to end up roughly
sensible (bad near low, great near high) rather than merely correctly
ordered. Report both the pairwise score and the raw Jev probability in the
feedback text, so the reflection LM sees both the ranking failure and the
absolute number.

Also compute, once per full evaluation (not per example), the batch-level
Spearman correlation and log it as a side metric for your own monitoring —
it is a good single number to watch across GEPA iterations even though it is
not the per-example GEPA score itself.

---

## 5. Budget: how GEPA spends metric calls

- Every call to your adapter's `evaluate()` on one example counts as (at
  least) one metric call. A "metric call" in GEPA's accounting is one
  example scored under one candidate.
- GEPA's loop, per iteration, roughly: (1) pick a candidate from the Pareto
  frontier (or the current best), (2) run it with `capture_traces=True` on a
  reflection minibatch (`reflection_minibatch_size`, default small — 3 to 5)
  drawn from `trainset`, (3) call `reflection_lm` once to propose a new
  candidate from the resulting reflective dataset, (4) evaluate the new
  candidate on the **full** `valset` (or a validation policy's subset) to
  decide whether to keep it. Step (4) is the expensive one: with `valset`
  size `V`, each accepted-or-rejected candidate costs about `V` metric calls,
  plus the `reflection_minibatch_size` calls from step (2).
- With `trainset`/`valset` around 80-100 examples total (say, 70 train / 20
  val, held out from the same pool), and wanting on the order of 20-40 GEPA
  iterations, a reasonable **`max_metric_calls`** is in the
  **300-800** range: roughly `iterations × (valset_size + reflection_minibatch_size)`.
  Start with `auto="light"` or an explicit `max_metric_calls=300` for a first
  pass; move to 600-1000 once you see the score curve still improving near
  the budget ceiling.
- Jev calls are cheap and fast compared to LM calls, so the dominant cost in
  this task is **not** the evaluation budget — it is the number of times the
  reflection LM is called, and that scales with the number of accepted
  mutation steps, not with `max_metric_calls` directly. Expect on the order of
  a few dozen reflection LM calls for a 300-800 metric-call run. Use a strong
  but not extremely expensive `reflection_lm` (for example Claude Sonnet or
  GPT-4o), since this LM's call count stays modest even at a fairly generous
  metric-call budget.
- If Jev has its own rate limit or per-call cost, that is the real budget
  constraint here, not GEPA's LM cost. Set `num_threads` (in `dspy.GEPA`) or
  parallelize `evaluate()` internally (in the standalone `gepa` adapter, you
  control this yourself inside the loop) to match what Jev's API can sustain.

---

## 6. Pitfalls

- **Overfitting to a small train set.** With only 80-100 labeled jokes, a
  criteria string that reaches a very high score on `trainset` may just be
  memorizing quirks of those specific jokes (for example, always rewarding
  jokes that mention animals, if your "great" jokes happen to be mostly
  animal jokes). Always hold out a real `valset` that GEPA never uses for
  reflection, only for acceptance decisions, and watch the train/val gap.
- **Valset usage.** If you pass no `valset`, both `dspy.GEPA` and the
  standalone `optimize()` fall back to using `trainset` for both reflection
  and acceptance, which removes your only overfitting check. Always pass a
  separate `valset`, even if it means a 70/20/10 (train/val/held-out-test)
  split of a modest pool.
- **Pareto selection can favor niche candidates.** A candidate that is
  "best" on one unusual joke (and mediocre elsewhere) can sit on the Pareto
  frontier and keep getting sampled. This is usually fine because GEPA
  combines lessons from several frontier members over time, but if you see
  the best-scoring candidate on `valset` swing between very different styles
  of criteria run to run, consider `candidate_selection_strategy="current_best"`
  for a more conservative search, at the cost of exploring less.
  `use_merge=True` can help combine two good but different frontier
  candidates instead of picking one.
- **Seed prompt quality matters.** GEPA mutates from the seed candidate; a
  vague seed criteria string ("judge if this is funny") gives the reflection
  LM less to work with early on. Write a first-draft criteria string that
  already states a real judging question, even if it is not tuned yet.
- **Non-determinism.** Both Jev (if it has any sampling) and the reflection
  LM can be non-deterministic between runs. Set `seed=` for GEPA's own RNG,
  but understand this does not make Jev or the reflection LM deterministic.
  Run GEPA more than once, or with `track_best_outputs=True` and
  `track_stats=True`, and compare `val_aggregate_scores` across runs before
  trusting a single result.
- **Label imbalance.** If "good" dominates the label set, both the pairwise
  metric and any closeness metric will be dominated by good-vs-other pairs.
  Consider balancing the trainset or weighting pairs by label rarity.
- **Jev may not be sensitive to any phrasing.** If Jev's underlying model
  ignores most of the criteria text (for example, it only reacts to a few
  keywords), GEPA will plateau quickly regardless of reflection quality.
  Run a quick manual sensitivity check (change the criteria drastically by
  hand, see if Jev's output moves at all) before investing a large budget.

---

## 7. Comparison with MIPROv2 and with simple search

| Approach | What it optimizes | Needs an LM in the eval loop? | Fits the Jev task? |
| --- | --- | --- | --- |
| `dspy.GEPA` | Instructions of `dspy.Predict`/`ChainOfThought` predictors inside a DSPy program | Yes, always (the program itself is an LM call) | No — forces an artificial DSPy program wrapper around a non-LLM scorer |
| Standalone `gepa.optimize()` with a custom adapter | Any text component(s), scored by any evaluator you write | No — only the reflection step uses an LM | Yes — this is the direct fit |
| `MIPROv2` | Instructions and few-shot demonstrations of a DSPy program, via Bayesian search over LM-proposed instruction candidates | Yes, always | No, for the same reason as `dspy.GEPA`; also, MIPROv2 needs a DSPy program to attach demonstrations to, which does not exist here |
| Random search / hill-climbing over criteria text | Whatever text you manually mutate | No | Technically fits, but wastes the richest resource GEPA has: **textual feedback about why an example failed**. Random search treats every trial as a black box score, with no diagnosis and no direction; it needs far more evaluation calls to find comparable improvement, and gives you no explanation of *why* the winning criteria works. |

**When GEPA is worth it, in general:** you have (a) a text artifact that
drives behavior, (b) an evaluator that can be run repeatedly and cheaply
enough to afford a few hundred calls, and (c) a way to produce specific,
per-example textual feedback about failures — not just a bare score. All
three hold for the Jev task: the criteria is the artifact, Jev is cheap and
fast to call, and you can always describe *why* a probability mismatched a
label (see section 4). GEPA's reflective step turns that description into a
directed edit, which is exactly what a hand-written hill-climb or a random
search cannot do — those only see the number, never the reason.

**When GEPA is not worth it:** the evaluator is extremely expensive per call
(hundreds of dollars, or minutes, each), in which case even GEPA's modest
budget of a few hundred calls may be too much, and a much smaller, more
manual search is safer. It is also not worth it if you cannot produce any
meaningful per-example feedback text (for example, the evaluator returns only
a bare pass/fail with no other information) — GEPA still works then, but it
degrades toward the same blind search that random search performs, so the
extra complexity of a `gepa` adapter buys little.

---

## 8. Versions and install commands (checked September 2026)

```bash
# DSPy framework (only needed if you are also building an LLM-driven
# DSPy program elsewhere in the project; not required for the gepa
# standalone adapter path recommended above)
pip install dspy==3.3.1          # latest on PyPI as of 2026-08-21
# source: https://pypi.org/project/dspy/

# Standalone GEPA optimizer (the recommended path for the Jev task)
pip install gepa==0.1.4          # latest on PyPI as of 2026-07-15
# source: https://pypi.org/project/gepa/

# Optional integration extras for gepa (not needed for this task,
# since the adapter is custom and does not use LangChain, etc.)
# pip install "gepa[langchain]"
```

Re-check `pip index versions dspy` and `pip index versions gepa` before a
real run; both projects release often.

---

## Sources

- DSPy docs: https://dspy.ai
- GEPA overview (DSPy API): https://dspy.ai/api/optimizers/GEPA/overview/
- GEPA advanced (DSPy API): https://dspy.ai/api/optimizers/GEPA/GEPA_Advanced/
- GEPA getting started: https://dspy.ai/getting-started/gepa-optimization/
- GEPA in depth: https://dspy.ai/diving-deeper/gepa-in-depth/
- GEPA for AIME tutorial: https://dspy.ai/tutorials/gepa_aime/
- GEPA for code-backdoor classification tutorial: https://dspy.ai/tutorials/gepa_trusted_monitor/
- GEPA for structured extraction (Facility Support Analyzer) tutorial: https://dspy.ai/tutorials/gepa_facilitysupportanalyzer/
- Reflective Prompt Evolution with GEPA (AI program tutorial): https://dspy.ai/tutorials/gepa_ai_program/
- GEPA paper (arXiv 2507.19457): https://arxiv.org/abs/2507.19457
- gepa-ai/gepa GitHub: https://github.com/gepa-ai/gepa
- gepa on PyPI: https://pypi.org/project/gepa/
- dspy on PyPI: https://pypi.org/project/dspy/
- stanfordnlp/dspy GitHub (source of `dspy/teleprompt/gepa/gepa.py` details): https://github.com/stanfordnlp/dspy
- DeepWiki (stanfordnlp/dspy and gepa-ai/gepa), queried for constructor and adapter signatures: https://deepwiki.com
