# TypeSafe AI "Jev" System One Model: Research Report

## Purpose of this Report

This report gives facts about the Jev model and its API. The facts help an
engineer who wants to use Jev as a personal joke-preference scorer. The
scorer will use a `criteria` text. An outer loop will change this text over
time.

## Sources

1. The installed SDK, `typesafe_sdk` 0.7.0, at:
   `/Users/naveenreddy/experiments/jev/jev-testing/.venv/lib/python3.14/site-packages/typesafe_sdk/`
2. Public docs at `https://docs.typesafe.ai` (fetched 2026-09-19).
3. The local runner file:
   `/Users/naveenreddy/experiments/jev/jev-experiments/parsing-experiments/src/jev_eval/runners/jev.py`

The SDK source is the most exact source. Doc pages sometimes summarize the
same facts in plain words.

---

## 1. What a "System One" Model Is

TypeSafe defines a System One model as a model that makes fast, typed
decisions. It does not write free text.

- You send **state**: the text, JSON object, or JSON array to judge.
- You send one or more **questions**. Each question has a **type**
  (`noul`, `choice`, or `score`).
- The model answers all questions in one pass, in parallel. It returns a
  typed value and a probability for each question. It does not return
  prose.
- Docs source: `https://docs.typesafe.ai/concepts/system-one`

### How it differs from an LLM

| Trait | LLM | System One model (Jev) |
|---|---|---|
| Output | Free text | Typed value + probability, per question |
| Reasoning shown | Often yes (chain of thought) | No. It gives a judgment, not an explanation |
| Latency (per docs/press) | Often 1-10+ seconds | 70-500 ms for a full call |
| Cost driver | Input + output tokens | Input tokens only; output tokens are free |
| Multiple questions | One prompt, one answer, or costly multi-turn | Many named questions in one call, answered in parallel with no added latency |
| Determinism | Low, and varies by provider | Better than an LLM at temperature 0, but still not exact (see §7) |

Company background (from press, not the SDK): Jev is TypeSafe AI's first
System One model. TypeSafe left stealth on 2026-09-15 with a $40M seed round.
The founder is Diego Almeida, a former OpenAI RLHF researcher. Access is by
waitlist as of this report's date.
- [TypeSafe AI Blog: Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [MindStudio: Jev Explained](https://www.mindstudio.ai/blog/jev-system-one-model-launch)
- [Let's Data Science: TypeSafe AI Launches Jev](https://letsdatascience.com/news/typesafe-ai-launches-jev-decision-model-889a38c0)
- [DataCamp: Jev, TypeSafe's System One Model](https://www.datacamp.com/blog/system-one-models-jev)

---

## 2. The Primitives (Question Types)

The SDK exposes three question types. Each is a Pydantic model and also
accepts a plain `dict` with a `"type"` key. Source:
`typesafe_sdk/_core/question_types.py` and `typesafe_sdk/_schemas/models.py`.

### 2.1 `Noul` — yes/no question

```python
from typesafe_sdk import Noul

Noul(
    instructions: str | dict | list | None = None,
    criteria: NoulCriteria | None = None,   # {"true": ..., "false": ...}
    type: Literal["noul"] = "noul",         # set automatically
)
```

- `instructions`: the yes/no question or statement. Optional, but a
  question with no instructions and no criteria is not useful.
- `criteria`: optional dict with keys `true` and `false`. Each value is
  `str | dict | list | None`. It describes what counts as a "yes" and what
  counts as a "no."

Response object `NoulAnswer`:

```python
class NoulAnswer:
    type: Literal["noul"] = "noul"
    noul: float   # 0.0 - 1.0, probability of "yes"
```

There is **no separate confidence field on a Noul answer.** The probability
is the only signal. Docs: `https://docs.typesafe.ai/confidence` states,
"Noul answers do not include confidence metrics."

### 2.2 `Choice` — pick one label from a set

```python
from typesafe_sdk import Choice

Choice(
    instructions: str | dict | list | None = None,
    criteria: Mapping[str, str | dict | list | None],  # REQUIRED, at least one entry
    type: Literal["choice"] = "choice",
)
```

- `criteria` is required for `Choice`. It is a mapping from a label name
  to a description (or `None` for an undescribed label).

Response object `ChoiceAnswer`:

```python
class ChoiceAnswer:
    type: Literal["choice"] = "choice"
    choice: str                     # label with the highest probability
    confidence: float                # 0-1
    probabilities: dict[str, float]  # one entry per criteria label, sums to ~1
```

### 2.3 `Score` — rate against an ordered rubric

```python
from typesafe_sdk import Score

Score(
    instructions: str | dict | list | None = None,
    criteria: Sequence[str | dict | list],  # REQUIRED, non-empty, ORDERED list
    type: Literal["score"] = "score",
)
```

- `criteria` is a non-empty, ordered list. Item at index 0 is score level 0,
  item at index 1 is level 1, and so on. The SDK raises a client-side
  `TypeSafeError` if the list is empty (checked in
  `typesafe_sdk/_core/questions.py::_validate_score_criteria`, before any
  network call).

Response object `ScoreAnswer`:

```python
class ScoreAnswer:
    type: Literal["score"] = "score"
    score: float                        # probability-weighted average level; can be fractional
    confidence: float                   # 0-1
    legend: dict[int, str | dict | list]     # criteria, keyed by the integer level
    probabilities: dict[int, float]          # one per level, sums to ~1
```

There is no fourth primitive in this SDK version (0.7.0). `__init__.py`
exports exactly `Noul`, `Choice`, and `Score` (plus their `*Model` `TypedDict`
forms for raw-dict callers). No "classification," "ranking," or
"extraction" primitive exists in this SDK or in the docs' primitives list
(`https://docs.typesafe.ai/primitives`).

---

## 3. How `criteria` Works, Exactly

- **Parameter name**: `criteria`. It is a field on the question object, not
  a field on the call itself and not a field on `state`.
- **Where it goes**: **per question**, not per call and not per state. Each
  named question in the `questions` mapping carries its own `criteria`. A
  single `system_one` call can hold several questions, and each can have
  different criteria.
- **Type by primitive**:
  - `Noul.criteria`: `dict` with optional keys `"true"` and `"false"`. Each
    value is `str | dict | list | None`. Fully optional; a Noul question
    can have no criteria at all.
  - `Choice.criteria`: `dict[str, str | dict | list | None]`. **Required.**
    At least one label is needed (the wire schema also implies non-empty,
    since a choice with no labels cannot be answered).
  - `Score.criteria`: ordered `list[str | dict | list]`. **Required and
    non-empty.** The SDK checks this locally, before sending the request.
- **Type of each entry**: a criteria value is never restricted to a plain
  string. It can be a string, a JSON object, or a JSON array
  (`JSONContent` in `typesafe_sdk/_core/json_types.py`). The docs' advanced
  page confirms structured criteria are supported and useful when a
  question has more than one part: "putting them in the form of JSON helps
  with clarity because the keys are labeled"
  (`https://docs.typesafe.ai/primitives/advanced`).
- **Length limits**: **no documented hard limit** was found in the SDK or
  the docs for criteria string length or nesting depth. The advanced page
  gives only a soft rule of thumb for large taxonomies: "If a branch is
  too large, trim the value to its direct children and a sample of leaves."
  The real ceiling is the model's context window, stated by the docs as
  64,000 tokens per request (state + instructions + all criteria +
  questions, combined), from `https://docs.typesafe.ai/models`.
- **What criteria changes, per docs**: for `Noul`, criteria "clarify what
  each outcome means, which can be helpful when the question itself has
  more nuance to explain" (`https://docs.typesafe.ai/primitives/noul`).
  Criteria do not change the output shape (still one float `noul`); they
  change what the model treats as the boundary between "yes" and "no."
- **Example from the Noul docs page**:

```json
{
  "type": "noul",
  "instructions": "Is this message spam?",
  "criteria": {
    "true": "Unsolicited advertising",
    "false": "A legitimate conversation"
  }
}
```

- **Example from the SDK's own docstring** (`TypeSafeClient.system_one`):

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

with TypeSafeClient() as client:
    result = client.system_one(
        state="I was charged twice. Please help.",
        questions={
            "billing": Noul(instructions="Is this about billing?"),
            "tone": Choice(
                instructions="What is the tone?",
                criteria={"calm": None, "angry": None},
            ),
        },
    )
    assert 0 <= result.nouls["billing"].noul <= 1
    assert result.choices["tone"].choice in {"calm", "angry"}
```

For a joke-preference scorer, `criteria` is the natural place to put the
learned preference text: a `Noul` question such as "Would I laugh at this
joke?" with `criteria={"true": <optimized description of what the user
finds funny>, "false": <optimized description of what falls flat>}`.

---

## 4. Multiple Questions per Call

Yes, this is supported and is a first-class pattern.

- `questions` is a `Mapping[str, Question]`. Each key is a name you choose;
  it becomes the key in `response.answers`. The API answers every question
  in one parallel pass, and the docs state that "adding more questions to a
  call typically doesn't add any latency to the response"
  (`https://docs.typesafe.ai/patterns/fan-out`, the "speculative fan-out"
  pattern).
- **The current local harness sends a dict with one key** (`QUESTION_KEY =
  "parses"` in `jev.py`). This is a harness choice, not an SDK limit. The
  harness could send several named `Noul`/`Score`/`Choice` questions in one
  call, for example one per candidate punchline, or one per scoring
  dimension (see composite scoring, §8).
- Constraint to note: **all questions in one call share the same `state`.**
  You cannot put two different jokes in one call and ask one question about
  each — the state is one shared piece of content. To score two different
  jokes, you need two calls (or one call with an array state and questions
  phrased to compare entries in that array; the docs' array-state examples
  are for conversation turns, not independent items to score
  separately). For batching many independent jokes, the practical pattern
  is one call per joke, with several *questions* about that one joke (e.g.,
  a Noul for "funny," a Score for "how funny," a Choice for "which type of
  joke").
- SDK-level minimum: `normalize_questions` raises `TypeSafeError("At least
  one question is required.")` if the mapping is empty
  (`typesafe_sdk/_core/questions.py`).

---

## 5. Can `state` Be Structured (dict/JSON), or Must It Be a String?

`state` can be a string, a JSON object, or a JSON array. It does not have to
be a string.

- SDK type: `state: JSONContent` on `TypeSafeClient.system_one`, where
  `JSONContent = str | Mapping[str, JSONValue | None] | Sequence[JSONValue
  | None]` (`typesafe_sdk/_core/json_types.py`).
- Wire schema: `SystemOneRequest.state: str | dict[str, Any] | list[Any]`
  (`typesafe_sdk/_schemas/models.py`).
- Docs: `https://docs.typesafe.ai/concepts/state` — "State can be structured
  as a string, JSON object, or array of text values... Objects suit most
  requests by organizing related information with descriptive field names.
  Arrays handle sequences like message conversations."
- Constraint: "Jev accepts text only... Images, audio, and video are not
  supported (yet)." Non-English text is accepted "with lower precision."

For a joke-preference scorer, packing `{"setup": ..., "punchline": ...}` as
`state` is supported and is likely clearer to the model than a single
concatenated string.

---

## 6. Latency, Rate Limits, Pricing, Model Names

All figures below with a `(press)` tag come from third-party coverage of the
launch, not from the docs or SDK, and should be treated as approximate.

- **Model names** (`https://docs.typesafe.ai/models`):
  - `jev-latest` — alias for the most recent **stable** release. This is
    the SDK's built-in default (`typesafe_sdk/constants.py:
    DEFAULT_MODEL = "jev-latest"`).
  - `jev-preview` — alias for the most recent release, stable or not.
  - Both aliases currently resolve to a versioned model, reported by docs
    as `jev-1.13.0`.
  - `client.models.list()` calls `GET /v1/models` and returns
    `ModelMetadata(name, description, release_date)` for each available
    model (`typesafe_sdk/_core/client/sync/models.py`).
- **Latency**: docs/press report 70-500 ms per call for a full
  `system_one` request `(press)`. The local harness's `DEFAULT_MODEL =
  "jev-latest"` and a `timeout=60.0` default give generous headroom above
  this.
- **Rate limits** (docs, `https://docs.typesafe.ai/api` and press coverage):
  measured in tokens/second and requests/minute; exceeding either returns
  HTTP 429. Reported current limits are about 250,000 tokens/second and
  1,200 requests/minute `(press)`, and the docs state these limits "are
  adjusting dynamically... and can change without notice as GPU capacity
  becomes available." Higher limits: contact `sales@typesafe.ai`.
  - The SDK's `RetryPolicy` (`typesafe_sdk/_core/retry.py`) retries 429 and
    5xx by default (`max_retries=2`, exponential backoff from 0.5s up to
    5s, jitter 0.25, honoring `Retry-After` / `retry-after-ms` headers, 30s
    total retry budget per call).
- **Pricing**: input tokens are billed; **output tokens are free**
  (`typesafe_sdk/_schemas/models.py::Usage.output_tokens` docstring:
  "Output tokens are currently free of charge."; confirmed by docs and
  press). Reported rate: **$0.042 per million input tokens**
  (equivalently $42 per billion tokens) `(press,` cross-checked against
  `https://docs.typesafe.ai` search results).
- **Context window**: reported as 64k tokens per request `(press)`. This
  covers `state` + all `instructions` + all `criteria` + all question
  names, combined, for the whole call.
- **Error types** (`typesafe_sdk/_core/errors.py`, all subclass
  `TypeSafeError`):
  `TypeSafeBadRequestError` (400), `TypeSafeAuthenticationError` (401),
  `TypeSafePermissionDeniedError` (403), `TypeSafeNotFoundError` (404),
  `TypeSafeUnprocessableEntityError` (422), `TypeSafeRateLimitError` (429,
  carries `retry_after_ms`), `TypeSafeInternalServerError` (5xx),
  `TypeSafeAPIConnectionError` / `TypeSafeAPITimeoutError` (no HTTP
  response), and `TypeSafeAPIResponseValidationError` (response shape did
  not match, names the bad `field_path`).

---

## 7. Determinism: Does the Same Input Give the Same `noul`?

**No. The same input does not give an exactly repeated `noul` value on every
call.** It is more stable than a typical LLM, but it is not exact.

- Docs cookbook `https://docs.typesafe.ai/cookbooks/consistency_noul_cookbook`
  ran the same claim and rubric 15 times. Result: "TypeSafe's mean
  per-question probability standard deviation is `0.0102`," clearly lower
  than the LLM conditions tested, but not zero. On one hard question, the
  noul values in the docs' own test ranged from 0.43 to 0.53 — a spread
  that **crosses a 0.5 decision threshold**.
- Variance concentrates on judgment-heavy, subjective questions, not on
  clear-cut ones. A joke-preference question ("would I laugh at this?") is
  exactly the subjective kind of question the docs flag as higher-variance.
- Implication for the outer loop: a single call's `noul` is a noisy
  estimate, not a fixed truth. Averaging repeat calls, or treating scores
  within about ±0.05 as equal, is advisable before the outer loop
  concludes that one candidate `criteria` text beats another.

---

## 8. Calibration, Thresholds, and How to Treat the Noul Probability

- `noul` is described as a genuine probability, not a raw model score:
  "near 1 means a strong yes... near 0 means a strong no... 0.5 does not
  mean medium skill — the scale measures likelihood, not magnitude"
  (`https://docs.typesafe.ai/primitives/noul`).
- System One models are trained so that "their probabilities are optimized
  against outcomes to reflect uncertainty" (`https://docs.typesafe.ai/concepts/system-one`)
  — i.e., calibration against real outcomes is a stated training goal, not
  only a post-hoc figure of speech.
- **Noul answers carry no separate `confidence` field.** `Choice` and
  `Score` answers do carry `confidence`, derived from how concentrated the
  `probabilities` distribution is. The docs (`https://docs.typesafe.ai/confidence`)
  give a three-tier guide: high confidence → act automatically; medium →
  act cautiously (confirm, flag, gather more input); low confidence → do
  not act, escalate or use a fallback. They add: "your code encodes the
  risk tolerance" — TypeSafe deliberately leaves the threshold choice to
  the caller, and suggests a floor of 0.5 confidence with the exact cutoff
  set per risk level, from your own testing.
- **The docs recommend the `Score` primitive, not `Noul`, for anything on a
  spectrum**, explicitly stating that Noul is for a binary yes/no, and that
  a graded judgment belongs in `Score`. A joke-preference scorer is
  arguably more of a spectrum ("how funny, 0-4") than a strict binary
  ("funny: yes/no"), so `Score` may fit the underlying judgment better than
  `Noul`, even though the existing harness uses `Noul` for a different task
  (parses/does-not-parse, which is genuinely binary).

---

## 9. Known Weaknesses Stated in the Docs

- Text only. No images, audio, or video (`https://docs.typesafe.ai/concepts/state`).
- Non-English input is accepted but with lower accuracy than English.
- Ambiguous `criteria`/`instructions` wording degrades the probability's
  usefulness — the docs explicitly warn that unclear criteria "impair
  probability interpretation."
- `Noul` is the wrong tool for a graded judgment; the docs steer graded
  judgments toward `Score`.
- Output is not deterministic; see §7. The docs' own published numbers show
  spread across a decision threshold on a repeated identical call.
- Rate limits are explicitly unstable right now: "limits can change without
  notice as GPU capacity becomes available" — an early-access product
  behind a waitlist as of 2026-09-15 `(press)`.
- The model does not explain its answer. There is no reasoning trace to
  debug a wrong `noul` value against; only the probability and (for
  `Choice`/`Score`) a probability distribution.

---

## 10. Implications for a Criteria-Optimization Loop

**Design fits well.** `criteria` is per-question, structured (string, dict,
or list), and has no documented hard length cap short of the 64k-token
context window, so the outer loop can write fairly long, structured
preference descriptions into `NoulCriteria` (or a `Score` rubric) without
hitting an SDK-enforced limit.

**Calls per optimization step.** Each joke + punchline pairing is a
different `state`, and a call's `questions` all share one `state`. So the
practical unit of work is **one call per (joke, punchline) pair**, not one
call per whole batch. Multiple *questions* about that one pair can still
ride in the same call at no extra latency (e.g., a `Noul` "would I laugh,"
plus a `Score` "how funny, 0-4," plus a `Choice` "which humor style,"
all in one request) — this is the "speculative fan-out" pattern from
`https://docs.typesafe.ai/patterns/fan-out`.

**Cost estimate for ~120 jokes × ~3.5 punchlines × N iterations.**

- Items per iteration: 120 × 3.5 = 420 (joke, punchline) pairs.
- Calls per iteration: 420, one call per pair (assuming one composite call
  per pair, bundling several questions as above — bundling questions does
  not multiply calls or latency, only slightly raises input tokens per
  call).
- Input tokens per call: state (joke + punchline, tens of tokens) plus
  instructions and criteria for however many questions are bundled. A
  criteria text of a few hundred words is roughly 400-800 tokens. Budget
  ~600-1,200 input tokens per call as a rough planning figure.
- Cost per call: 1,000 input tokens × $0.042 / 1,000,000 ≈ **$0.000042**
  (a few thousandths of a cent). Output tokens are free.
- Cost per iteration (420 calls): ≈ **$0.018**, call it $0.02.
- Cost for N = 50 outer-loop iterations: ≈ **$1**. Even N = 500 stays under
  **$10**. Cost is not the binding constraint for this loop; call volume
  and latency are.
- Latency/throughput: at the press-reported 1,200 requests/minute cap (20
  req/s), 420 calls take a minimum of about 21 seconds per iteration if
  fully parallelized up to that cap (use `AsyncTypeSafeClient` /
  `asyncio.gather` with a semaphore near the rate limit, and let the
  built-in `RetryPolicy` absorb any 429s). Sequential, unbatched calls at
  70-500 ms each would instead take 30 seconds to 3.5 minutes for the same
  420 calls; concurrency is worth doing.

**Does batching several jokes into one call help?** No, not for
independent scoring. `state` is one object per call, so batching only helps
within a single item (bundling several *questions* about the same joke),
not across different jokes. To reduce call count across jokes, the only
lever is to reduce items scored per iteration (e.g., score a sampled subset
of the 420 pairs each outer-loop step rather than the full set), not to pack
multiple jokes into one `state`.

**Noise control.** Given the measured noul standard deviation of about
0.01, and spreads that can cross the 0.5 boundary on subjective questions
(§7), the outer loop should either (a) average 2-3 repeat calls per pair
before scoring a candidate `criteria`, or (b) use `Score` instead of `Noul`
so the finer-grained `score` value (a probability-weighted average over
levels) is less likely to flip across a hard boundary from noise alone.
Repeating calls multiplies cost and call count by the same factor (2-3x),
which the cost estimate above still keeps well under a few dollars total.

**Criteria length.** Nothing in the SDK enforces a specific character or
word limit on `criteria`. The real ceiling is the 64k-token context window,
shared with `state`, `instructions`, and all other questions in the same
call. A learned criteria text of even a few thousand words is very unlikely
to approach that limit on its own; it becomes a constraint only if the
outer loop also stuffs many few-shot examples or a large joke corpus into
the same call's `state`.

---

## Appendix: Exact SDK Surface Referenced

- Client: `typesafe_sdk.TypeSafeClient` (sync),
  `typesafe_sdk.AsyncTypeSafeClient` (async). Both expose
  `.system_one(state, questions, *, model=None, retry=None, timeout=None,
  extra_headers=None, extra_body=None, response_model=None)` and
  `.models.list(...)`.
- Question types: `typesafe_sdk.Noul`, `typesafe_sdk.Choice`,
  `typesafe_sdk.Score`, plus `NoulModel`/`ChoiceModel`/`ScoreModel`
  `TypedDict` forms for raw-dict callers, and `NoulCriteria`.
- Response types: `typesafe_sdk.SystemOneResponse` (with `.nouls`,
  `.choices`, `.scores` convenience dict properties keyed by question
  name), `NoulAnswer`, `ChoiceAnswer`, `ScoreAnswer`, `Usage`.
- Errors: all subclass `typesafe_sdk.TypeSafeError`; HTTP errors subclass
  `TypeSafeAPIError` and carry `.status`, `.body`, `.headers`,
  `.request_id`.
- Environment variables: `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`,
  `TYPESAFE_DEFAULT_MODEL`, `TYPESAFE_LOG_LEVEL`.
- Defaults: `DEFAULT_BASE_URL = "https://api.typesafe.ai"`,
  `DEFAULT_MODEL = "jev-latest"`, `DEFAULT_TIMEOUT = 10.0` seconds (the SDK
  default; the local harness overrides this to 60.0 seconds).

## Sources Cited

- `https://docs.typesafe.ai/primitives/noul`
- `https://docs.typesafe.ai/primitives/choice` (schema cross-checked against SDK)
- `https://docs.typesafe.ai/primitives/score` (schema cross-checked against SDK)
- `https://docs.typesafe.ai/primitives/advanced`
- `https://docs.typesafe.ai/primitives`
- `https://docs.typesafe.ai/concepts/system-one`
- `https://docs.typesafe.ai/concepts/state`
- `https://docs.typesafe.ai/confidence`
- `https://docs.typesafe.ai/patterns/fan-out`
- `https://docs.typesafe.ai/patterns/composite-scoring`
- `https://docs.typesafe.ai/cookbooks/consistency_noul_cookbook`
- `https://docs.typesafe.ai/introduction/quickstart`
- `https://docs.typesafe.ai/models`
- `https://docs.typesafe.ai/api`
- `https://docs.typesafe.ai/llms.txt`
- `https://typesafe.ai/blog/introducing-system-one-models-and-jev`
- `https://www.mindstudio.ai/blog/jev-system-one-model-launch`
- `https://letsdatascience.com/news/typesafe-ai-launches-jev-decision-model-889a38c0`
- `https://www.datacamp.com/blog/system-one-models-jev`
- Local SDK: `/Users/naveenreddy/experiments/jev/jev-testing/.venv/lib/python3.14/site-packages/typesafe_sdk/` (v0.7.0)
- Local usage: `/Users/naveenreddy/experiments/jev/jev-experiments/parsing-experiments/src/jev_eval/runners/jev.py`
