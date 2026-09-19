# Report: how the parsing experiments use Jev and DeepSeek

This report is for the reader who will build the joke-preference project. It
reads the `parsing-experiments` harness and gives a reuse list.

Source: `/Users/naveenreddy/experiments/jev/jev-experiments/parsing-experiments/`.

A near copy exists at `/Users/naveenreddy/experiments/jev/jev-testing/`. The
`parsing-experiments/README.md` file is byte-identical to
`jev-testing/README.md`. `jev-testing` is a flat, pre-rename working copy. It
adds `HANDOFF.md`, `STATUS.md`, and `.gitignore` at its top level, and it is
not a git repository. Use `jev-experiments/parsing-experiments` as the source
of truth. It is the file this report cites.

---

## 1. Architecture of the harness

The harness has five layers. Each layer is one file or one small package.

```text
parsing-experiments/
  src/jev_eval/
    tasks.py       # the locked task registry (task id, question, oracle, "yes means")
    schema.py       # the Item dataclass, JSONL load/validate, the model-output contract
    generators/     # builds oracle-labelled items from templates and faults
    oracles/        # runs the real parser/compiler tool per task and returns pass/fail
    runners/
      jev.py         # calls the Jev SDK
      llm.py         # calls Gemini and DeepSeek over HTTP
    metrics.py      # accuracy, invalid-rate, latency, Brier, ECE, reliability table
    cli.py          # generate / validate / run / score subcommands
  data/seed/, data/hard/   # committed JSONL item files
  results/<run-name>/      # predictions.jsonl, run_meta.json, metrics.json (gitignored)
  docs/*.md                # write-ups
  scripts/                 # counts.py, analyze_reasoning.py, analyze_variance.py, build_hard.py, make_subset.py
  tests/                   # unittest suite, one file per layer
```

Data flow:

1. `tasks.py` defines a fixed set of tasks. Each task has one locked question
   string, for example `"Does this parse as bash?"` (`src/jev_eval/tasks.py:32`).
2. `generators/` writes candidate artifacts, runs the real oracle tool on
   each one, and keeps only labels the oracle confirms. Output is a list of
   `Item` objects, written as JSONL.
3. `schema.py` defines `Item` (the labelled test case) and the model-output
   contract (`ModelOutput` / `InvalidOutput`). It also validates every JSONL
   row against a strict set of rules on load.
4. `runners/` (`jev.py`, `llm.py`) send one item at a time to a model and
   return a `CallResult`: parsed output, latency, raw text, error, usage,
   reasoning.
5. `cli.py` command `run` loops over items, calls a runner, writes one JSON
   row per item to `predictions.jsonl`, and writes `run_meta.json` and
   `metrics.json`.
6. `metrics.py` command `score` (or the `run` command at the end) turns a list
   of prediction rows into a metrics summary: accuracy, invalid rate, latency
   percentiles, Brier score, ECE, reliability table.

The CLI (`src/jev_eval/cli.py`) has four subcommands: `generate`, `validate`,
`run`, `score`. `run` needs `--provider` (one of `gemini-flash-lite`,
`deepseek-flash`, `jev`), `--data`, `--out`, and optional `--mock`, `--limit`,
`--difficulty`, `--timeout`.

---

## 2. Exactly how Jev is called

File: `src/jev_eval/runners/jev.py`.

### Client construction

```python
# src/jev_eval/runners/jev.py:71-89
def _build_client(self):
    try:
        import typesafe_sdk as sdk
    except ImportError as exc:
        raise FatalRunnerError(
            "typesafe_sdk is not installed. Install the Jev extra: "
            "pip install -e '.[jev]'"
        ) from exc

    env_name, key = required_api_key("jev")
    if not key:
        raise FatalRunnerError(
            f"No API key for jev (set {env_name}). Refusing to call the "
            "network. Re-run with --mock for an offline dry-run."
        )
    client = sdk.TypeSafeClient(api_key=key, timeout=self.timeout)
    return client, sdk
```

The SDK is imported lazily, inside the method, not at module top. This keeps
`typesafe_sdk` an optional dependency (`pyproject.toml` extra `jev`). Other
runners (LLM baselines) work with no SDK installed.

### Key handling

`required_api_key("jev")` (in `runners/llm.py:49-59`) reads `JEV_API_KEY` from
the environment. The harness passes the key to the SDK explicitly:

```python
client = sdk.TypeSafeClient(api_key=key, timeout=self.timeout)
```

It does **not** rely on the SDK's own default env var, `TYPESAFE_API_KEY`.
The harness "owns the key name," per the module docstring
(`runners/jev.py:1-27`) and the README ("Jev provider" section).

### The `system_one` call

```python
# src/jev_eval/runners/jev.py:96-104
question = get_task(item.task).question
response = self._client.system_one(
    item.artifact,
    {QUESTION_KEY: sdk.Noul(instructions=question)},
    model=self.model,
    timeout=self.timeout,
)
```

- First argument: `state`, the raw artifact text (source code). No wrapper,
  no rubric, no extra prompt text.
- Second argument: a dict of named questions. This harness sends exactly one,
  under the key `"parses"` (`QUESTION_KEY = "parses"`, `runners/jev.py:43`).
- `Noul(instructions=question)` — `instructions` is the locked task question,
  sent unchanged. `Noul` also accepts a `criteria` argument, confirmed by the
  test fake (`tests/test_runners.py:62-64`), but v1 **never sets it**. See
  section 6.
- `model="jev-latest"` by default (`DEFAULT_MODEL`, `runners/jev.py:46`),
  overridable by the `JEV_MODEL` env var (`MODEL_ENV`, `runners/jev.py:49`).
- `timeout=self.timeout`, default 60.0 seconds (`JevRunner.__init__`,
  `runners/jev.py:61`), passed through from the CLI `--timeout` flag.

### The response shape

A Noul answer is one field:

```json
{"type": "noul", "noul": 0.92}
```

There is no boolean and no separate confidence value. The harness maps it:

```python
# src/jev_eval/runners/jev.py:138-151
p_yes = float(answer.noul)
...
return CallResult(
    parsed=ModelOutput(answer=p_yes >= YES_THRESHOLD, p_yes=p_yes),
    latency_ms=latency_ms,
    raw=raw,
)
```

`YES_THRESHOLD = 0.5` (`runners/jev.py:40`) is called out explicitly as a
**harness policy choice, not a model output**. It can be changed in this one
constant, or predictions can be re-thresholded from the raw `p_yes` already
in `predictions.jsonl` without spending new calls (README, "Jev provider").

### Error classes and the abort/invalid policy

```python
# src/jev_eval/runners/jev.py:105-115
except (sdk.TypeSafeAuthenticationError, sdk.TypeSafePermissionDeniedError) as exc:
    raise FatalRunnerError(f"{type(exc).__name__}: {exc}") from exc
except sdk.TypeSafeError as exc:
    return CallResult(
        parsed=InvalidOutput(reason=f"{type(exc).__name__}: {exc}", raw=""),
        latency_ms=(time.perf_counter() - t0) * 1000.0,
        raw="",
        error=type(exc).__name__,
    )
```

Policy table (from `runners/jev.py` docstring and README):

| condition | result |
|---|---|
| `TypeSafeAuthenticationError` / `TypeSafePermissionDeniedError` | `FatalRunnerError` — the whole `run` aborts (exit 2); partial predictions already written are kept |
| any other `sdk.TypeSafeError` (timeout, rate limit, 5xx, validation error) | row marked `invalid`; exception class name goes in `error` |
| SDK response has no `noul` answer under the question key | row marked `invalid`, `error="missing_noul"` |
| `noul` value outside `[0, 1]` | row marked `invalid`, `error="noul_out_of_range"` |
| no `JEV_API_KEY` set | `FatalRunnerError` at construction time, before any network call |

The reasoning given in the docstring: an auth failure that is silently
retried per item would write hundreds of invalid rows that look like the
model scoring at chance. Aborting is safer than that.

### Threshold policy

`YES_THRESHOLD = 0.5`, a module constant, applied uniformly. `docs/evaluation.md`
shows re-thresholding the same 200 predictions at no extra cost: accuracy
rises from 0.785 at 0.5 to a peak of 0.860 at 0.75, but the doc warns this is
fitted on the same data it is measured on — "hold out a split before you
change the default."

### Latency observed

From `docs/evaluation.md` (200-item run, `jev-1.13.0` vs `deepseek-flash`):

| | Jev | DeepSeek |
|---|---|---|
| latency mean | 375 ms | 2662 ms |
| latency p50 | 342 ms | 1635 ms |
| latency p95 | 462 ms | 6361 ms |

Jev is about 7x faster on mean latency and about 14x faster at p95. One
sample prediction row: `latency_ms: 951.5` for one bash item
(`results/jev-full-200/predictions.jsonl` line 1) — individual calls vary,
but the aggregate mean across 200 items is 375 ms.

### `_mock` (offline dry-run)

```python
# src/jev_eval/runners/jev.py:153-169
def _mock(self, item: Item) -> CallResult:
    p_yes = 0.72 if hash(item.id) % 2 == 0 else 0.28
    ...
```

`--mock` exercises the noul-to-contract mapping with no network call and no
SDK import requirement.

---

## 3. Exactly how DeepSeek is called

File: `src/jev_eval/runners/llm.py`. DeepSeek is one of two LLM baselines
(the other is Gemini Flash Lite); both share one code path.

### Endpoint and model

```python
DEEPSEEK_DEFAULT_MODEL = "deepseek-flash"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
```
(`runners/llm.py:23-24`)

Both are overridable by environment variables: `DEEPSEEK_MODEL` and
`DEEPSEEK_BASE_URL` (`runners/llm.py:199-200`).

### Key handling

```python
# src/jev_eval/runners/llm.py:57-58
if provider == "deepseek-flash":
    return "DEEPSEEK_API_KEY", os.environ.get("DEEPSEEK_API_KEY")
```

### The call

```python
# src/jev_eval/runners/llm.py:195-237
def _call_deepseek(self, system: str, user: str):
    ...
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }
    body = _http_json(
        base,
        payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        timeout=self.timeout,
    )
    choices = body.get("choices") or []
    ...
    message = choices[0].get("message") or {}
    usage = body.get("usage")
    reasoning = message.get("reasoning_content")
    content = message.get("content")
    ...
    return text, usage, reasoning
```

`_http_json` (`runners/llm.py:240-261`) is a small wrapper over
`urllib.request` — stdlib only, no HTTP client dependency. It raises
`RuntimeError` on a non-2xx status or a non-JSON / non-object body.

### Prompt template

Both LLM baselines share one system rubric:

```python
# src/jev_eval/runners/llm.py:29-39
SYSTEM_RUBRIC = """You are scoring a single atomic yes/no item.
Answer only the exact question below. Do not execute, compile, or interpret beyond the question.

Question: {question}

Return a single JSON object with exactly these fields:
- "answer": boolean — true if the answer is yes, false if no
- "p_yes": number in [0, 1] — your probability that the correct answer is yes

No markdown, no extra keys, no commentary. Example: {{"answer": true, "p_yes": 0.87}}
"""
```

`build_prompt` (`runners/llm.py:78-82`) fills `{question}` with the task's
locked question and builds the user message as `f"Artifact:\n\n{item.artifact}\n"`.

### JSON contract

`{"answer": bool, "p_yes": float in [0, 1]}`. DeepSeek is called with
`response_format: {"type": "json_object"}`, which is DeepSeek's structured
output mode (Gemini instead uses `responseMimeType` + `responseSchema`, a
different mechanism per README). The returned text is passed to
`parse_model_output` (`schema.py:211-244`), which:
- accepts raw JSON, JSON fenced in a ```` ```json ```` block, or a JSON object
  found inside surrounding prose (via `_extract_balanced_object`);
- rejects (marks invalid) missing fields, wrong types (e.g. a string
  `"true"`, or `p_yes` as a JSON boolean), and out-of-range `p_yes`;
- never imputes a missing field.

### Temperature history

No temperature parameter is sent by the current code (`_call_deepseek` above
has no `temperature` key). `docs/reasoning-tokens.md` and `docs/run-variance.md`
record that this changed over time:

- Early runs sent `temperature: 0` (pinned).
- The harness was later changed to send **no** sampling parameter at all —
  it now "takes each provider's default, so it measures the model as
  shipped." (`runners/llm.py:167-169` comment, repeated for both providers;
  `docs/reasoning-tokens.md` "Provenance" note.)
- This makes the earlier temperature-0 numbers **not reproducible** with the
  current code, and the docs flag which numbers came from which regime.

### Reasoning token capture

```python
# src/jev_eval/runners/llm.py:226-227
reasoning = message.get("reasoning_content")
reasoning = reasoning if isinstance(reasoning, str) else None
```

DeepSeek Flash is a reasoning model. Its response separates the chain of
thought (`reasoning_content`) from the visible answer (`content`). The
harness keeps both: `reasoning` goes into `CallResult.reasoning`
(`runners/llm.py:97`, with the comment "kept so failures can be read, not
only counted").

### Usage telemetry

```python
# src/jev_eval/cli.py:80-97
def _telemetry(call) -> dict:
    usage = call.usage or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "usage": usage or None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "reasoning_chars": len(call.reasoning) if call.reasoning else None,
        "reasoning": call.reasoning,
    }
```

`reasoning_tokens` comes straight from DeepSeek's own
`usage.completion_tokens_details.reasoning_tokens` field — the provider's own
count, not independently checked (`docs/reasoning-tokens.md`, "Limits").
`reasoning_chars` sizes the reasoning text when present. This dict is spread
into every prediction row (`cli.py:116`, `cli.py:132`), so it lands in
`predictions.jsonl` even for non-reasoning providers (where it is mostly
`None`).

---

## 4. The metrics module

File: `src/jev_eval/metrics.py`. Import-safe, pure functions on plain data
(lists and dicts) — no dependency on `Item` or the runners.

```python
def expected_calibration_error(
    golds: list[bool],
    p_yes: list[float],
    *,
    bins: int = 10,
) -> tuple[float | None, list[dict[str, Any]]]:
    ...

def brier_score(golds: list[bool], p_yes: list[float]) -> float | None:
    ...

def score_rows(rows: Iterable[dict[str, Any]], *, include_by_task: bool = True) -> dict[str, Any]:
    ...
```

(Signatures at `metrics.py:30-35`, `metrics.py:76`, `metrics.py:88`.)

- `expected_calibration_error` bins `p_yes` into `bins` equal-width buckets
  (default 10, bucket `i` covers `[i/bins, (i+1)/bins]`, last bucket includes
  1.0), and returns `(ece, reliability_table)`. Each reliability-table row is
  `{"bin", "lo", "hi", "count", "avg_p_yes", "empirical_yes_rate"}`.
- `brier_score` is the mean squared error between `p_yes` and the 0/1 gold
  label.
- `confidence(p_yes)` lives in `schema.py:247-248`:
  `max(p_yes, 1.0 - p_yes)`.
- `score_rows` takes a list of prediction-row dicts (the same shape written
  to `predictions.jsonl`) and returns:

```python
{
  "n": ..., "n_valid": ..., "n_invalid": ..., "invalid_rate": ...,
  "accuracy": ...,               # computed on valid rows only; None if no valid rows
  "accuracy_note": "accuracy is computed on valid outputs only; invalids are not imputed",
  "latency_ms": {"mean": ..., "p50": ..., "p95": ...},
  "calibration": {
    "n_with_p_yes": ..., "brier": ..., "ece": ..., "mean_confidence": ...,
    "confidence_note": "confidence = max(p_yes, 1 - p_yes)",
    "reliability": [...],
  },
  # optional, when rows span more than one task:
  "by_task": {"<task_id>": <same shape, include_by_task=False>, ...},
}
```

`score_rows` recurses into itself per task when `include_by_task=True` (the
default) and more than one distinct `task` value is present in `rows`. It
reads a row's `answer`, `gold`, `invalid`, `p_yes`, `latency_ms`, `task` keys
— exactly the keys `cli.py`'s `_prediction_row` writes (section 5). This
module can be imported and used directly, with no CLI, no file I/O, and no
knowledge of Jev or any provider — it is a plain function over rows shaped
like `{"answer": bool|None, "gold": bool, "invalid": bool, "p_yes": float|None, "latency_ms": float, "task": str}`.

---

## 5. The prediction row format and run_meta.json

### `predictions.jsonl` — one JSON object per line

Built by `_prediction_row` (`cli.py:100-133`). Two shapes, selected by
`isinstance(call.parsed, ModelOutput)` vs `InvalidOutput`:

Valid row (from `results/jev-full-200/predictions.jsonl`, line 1):

```json
{"id":"bash.parse.00001","task":"bash.parse","gold":true,"provider":"jev",
 "invalid":false,"invalid_reason":null,"answer":true,"p_yes":0.8,
 "confidence":0.8,"latency_ms":951.5077921096236,
 "raw":"{\"model\":\"jev-1.13.0\",\"answers\":{\"parses\":{\"type\":\"noul\",\"noul\":0.8}},\"usage\":{\"input_tokens\":288,\"output_tokens\":21}}",
 "error":null,"usage":null,"prompt_tokens":null,"completion_tokens":null,
 "reasoning_tokens":null,"reasoning_chars":null,"reasoning":null}
```

DeepSeek row (from `results/deepseek-hard-new/predictions.jsonl`, line 1):

```json
{"id":"bash.parse.h2.00001","task":"bash.parse","gold":false,
 "provider":"deepseek-flash","invalid":false,"invalid_reason":null,
 "answer":true,"p_yes":1.0,"confidence":1.0,"latency_ms":2822.895...,
 "raw":"{\"answer\": true, \"p_yes\": 1}","error":null, ...telemetry keys...}
```

Full key list, in write order: `id`, `task`, `gold`, `provider`, `invalid`,
`invalid_reason`, `answer`, `p_yes`, `confidence`, `latency_ms`, `raw`,
`error`, then the six telemetry keys (`usage`, `prompt_tokens`,
`completion_tokens`, `reasoning_tokens`, `reasoning_chars`, `reasoning`).
`gold` is the oracle label from the source `Item`; `answer`/`p_yes` are
`None` on an invalid row; `raw` holds the raw model text (LLM) or the raw
JSON dump of the SDK response (Jev), so a failure can be re-read without a
new call.

### `run_meta.json`

Written once per run (`cli.py:189-198`):

```json
{
  "provider": "jev",
  "mock": false,
  "n": 200,
  "difficulty": "all",
  "data": [
    "data/seed/bash.parse.jsonl",
    "data/seed/c.compile.jsonl",
    "data/seed/latex.parse.jsonl",
    "data/seed/python.parse.jsonl"
  ],
  "created_at": "2026-09-18T15:50:41.698935+00:00",
  "version": "0.1.0"
}
```

Fields: `provider`, `mock` (bool), `n` (row count), `difficulty` (list or the
string `"all"`), `data` (the resolved input file paths), `created_at` (UTC
ISO 8601), `version` (the `jev_eval` package version).

`metrics.json` (also written by `run`, and by `score`) holds exactly the
`score_rows` output shown in section 4.

---

## 6. Key findings from the docs and article

Run counts, so numbers below can be weighed:

- `docs/evaluation.md`: 1 run each, Jev and DeepSeek, 200 items (the full
  seed set).
- `docs/hard-set-v2.md`: 1 run each, 160 new "hard, long" items, plus reuse
  of the 74 hard items already inside the 200-item run.
- `docs/reasoning-tokens.md`: 1 run, DeepSeek only, 360 items (200 + 160)
  — a token/latency analysis of the same responses, not new model calls.
- `docs/run-variance.md`: 3 repeat runs, DeepSeek only, on the same 160
  hard items, default sampling (no temperature pin).

### Jev's calibration — the direct answer

**Is Jev's noul well calibrated? No, not on hard items, and it degrades as
items get harder.**

| set | Jev Brier | Jev ECE | Jev mean confidence | Jev accuracy |
|---|---|---|---|---|
| full 200-item seed set | 0.151 | 0.146 | 0.824 | 0.785 |
| combined 234 hard items (v1 hard + new hard) | 0.259 | 0.254 | — | 0.607 |

(`docs/evaluation.md` top table; `docs/hard-set-v2.md` "Calibration on the
combined set" line.)

For comparison, DeepSeek's Brier/ECE stay low even on the hard sets: Brier
0.061, ECE 0.055 on the same 234 hard items; Brier 0.034, ECE 0.036 on the
200-item seed set.

**Distribution of noul values.** Jev does not become uncertain (drift toward
0.5) as items get harder. It stays confidently high and wrong. From the
article: on the 160 new "hard, long" items, Jev's mean probability on
*broken* items is 0.712 — confidently wrong, not uncertain
(`docs/hard-set-v2.md`, "Jev does not detect faults in long artifacts"). On
the split "Jev misses every time" fault types, mean `p_yes` runs 0.67–0.92;
on the "Jev always catches" fault types, mean `p_yes` runs 0.05–0.40 — the
two groups do not overlap (`docs/evaluation.md`, "Jev's blind spot is
non-local structure").

Across all 360 items in the article's combined run, Jev said "yes" 278
times, although only 180 of 360 are actually valid — a strong yes-bias. On
the LaTeX slice of the new hard set (40 items, 20 valid / 20 broken), Jev
said yes to all 40; its 0.500 score there is exactly chance and "carries no
information" (`docs/hard-set-v2.md`).

**Threshold sensitivity.** Because the noul itself carries some signal even
when the 0.5-thresholded boolean does not, re-thresholding the 200-item run
recovers some accuracy: 0.785 at 0.5, up to 0.860 at 0.75, dropping again at
0.9 (0.730) (`docs/evaluation.md`, "Threshold"). This is explicitly flagged
as fitted to the same data it is scored on — not a validated setting.

**Determinism.** Jev was never repeated in this whole project — "one run
per item" everywhere Jev is mentioned. The docs are explicit that this makes
Jev's numbers the *less* well-established of the two model columns, not the
more, because DeepSeek's hard-set number is a mean of three runs
(0.9187 / 0.9313 / 0.9437) while Jev's is a single call per item
(`docs/hard-set-v2.md` callout box; `README.md` caveats). No claim is made
about whether Jev's `system_one` call is itself deterministic — it is simply
never tested for that here.

**The `criteria` field was never sent.** Confirmed in three independent
places:
1. Code: `runners/jev.py` calls `sdk.Noul(instructions=question)` with no
   `criteria=` argument (`runners/jev.py:101`).
2. Test: `test_sends_locked_question_and_raw_artifact` asserts
   `self.assertIsNone(question.criteria)` (`tests/test_runners.py:175`).
3. Docs: the module docstring says explicitly, "No `criteria` are attached:
   the v1 seed labels are reproducible against a fixed oracle and a fixed
   question, so changing the question shape would change what is measured.
   Criteria belong in a separate comparison run." (`runners/jev.py:20-23`,
   repeated in the README "Jev provider" section, the article's caveats
   section, and listed as unfinished "next steps" in both `evaluation.md`
   and `hard-set-v2.md`: "Test whether `criteria` with `true` and `false`
   descriptions reduce Jev's false positives... Keep it as a separate run.")

This matters directly for the joke project: it confirms `criteria` is a real,
documented, but unused field on `Noul`, and it is the exact field the joke
project's GEPA optimizer will tune.

### Other findings worth carrying over

- **Jev's failure mode is structural, not random.** It catches local,
  token-level faults (missing semicolon, unclosed string) and misses
  non-local, block-structure faults (missing `then`, unclosed brace 40 lines
  away) (`docs/evaluation.md`).
- **Jev is ~7x faster than DeepSeek on mean latency, ~14x at p95**, and
  produced **zero invalid outputs** across every run in this project — the
  SDK's typed response never fails the harness's own JSON contract, unlike
  free-text LLM output which sometimes needs the fenced/embedded-JSON
  fallback parsing in `schema.py`.
- **DeepSeek can self-contradict** (`{"answer": false, "p_yes": 0.95}`); a
  Noul answer structurally cannot, because it is one field
  (`docs/hard-set-v2.md`, "DeepSeek contradicts itself; Jev cannot"). This is
  a real advantage of Jev's typed output, independent of accuracy.
- **DeepSeek's reasoning length predicts instability, not error** (r = +0.004
  with error, but a 4.6x token gap between stable and unstable items) —
  relevant only if the joke project ever adds a reasoning-model baseline
  alongside Jev.

---

## 7. Test strategy — a fake SDK namespace for Jev

File: `tests/test_runners.py:54-219`. This is the piece to copy directly for
any new project that also calls `typesafe_sdk`.

The tests **never import `typesafe_sdk`** and never touch the network. They
build a fake namespace object with the same attribute names the runner code
looks up (`sdk.Noul`, `sdk.TypeSafeError`, `sdk.TypeSafeAuthenticationError`,
etc.), and a fake client whose `system_one` method returns a canned response
or raises a canned exception.

```python
# tests/test_runners.py:61-112 (trimmed)
class _FakeNoul:
    def __init__(self, instructions=None, criteria=None):
        self.instructions = instructions
        self.criteria = criteria

class _FakeNoulAnswer:
    type = "noul"
    def __init__(self, noul: float):
        self.noul = noul
    def model_dump(self) -> dict:
        return {"type": "noul", "noul": self.noul}

class _FakeUsage:
    def model_dump(self) -> dict:
        return {"input_tokens": 10, "output_tokens": 2}

class _FakeResponse:
    def __init__(self, answers: dict):
        self.model = "jev-1.13.0"
        self.answers = answers
        self.usage = _FakeUsage()

class _FakeTypeSafeError(Exception): pass
class _FakeAuthError(_FakeTypeSafeError): pass
class _FakePermissionError(_FakeTypeSafeError): pass
class _FakeRateLimitError(_FakeTypeSafeError): pass

def _fake_sdk() -> SimpleNamespace:
    return SimpleNamespace(
        Noul=_FakeNoul,
        TypeSafeError=_FakeTypeSafeError,
        TypeSafeAuthenticationError=_FakeAuthError,
        TypeSafePermissionDeniedError=_FakePermissionError,
        TypeSafeRateLimitError=_FakeRateLimitError,
    )

class _FakeClient:
    def __init__(self, noul=None, raises=None):
        self.noul = noul
        self.raises = raises
        self.calls: list[dict] = []
    def system_one(self, state, questions, *, model=None, timeout=None):
        self.calls.append({"state": state, "questions": questions, "model": model, "timeout": timeout})
        if self.raises is not None:
            raise self.raises
        return _FakeResponse({QUESTION_KEY: _FakeNoulAnswer(self.noul)})
```

Wiring helper, used by every Jev test:

```python
# tests/test_runners.py:132-139
def _wired_runner(noul=None, raises=None) -> tuple[JevRunner, _FakeClient]:
    """A live-path JevRunner with the SDK and client swapped for fakes."""
    runner = JevRunner(mock=True)     # mock=True skips the real client build
    client = _FakeClient(noul=noul, raises=raises)
    runner.mock = False                # then flip it back to the live code path
    runner._client = client
    runner._sdk = _fake_sdk()
    return runner, client
```

The trick: `JevRunner(mock=True)` skips `_build_client` (no import, no env
lookup, no exception if the SDK is absent). The test then sets `mock = False`
and injects `_client`/`_sdk` directly, so `runner.complete(item)` runs the
*real* `complete()` code path (`system_one` call, error handling, mapping)
against the fake objects.

Tests built this way cover: noul-to-`p_yes` mapping with no inversion
(`test_noul_maps_straight_to_p_yes`), the threshold boundary being inclusive
(`test_threshold_boundary_is_inclusive`), that the state sent is the raw
artifact with no wrapper and the question sent is the exact locked task
question with `criteria=None` (`test_sends_locked_question_and_raw_artifact`),
timeout propagation (`test_cli_timeout_reaches_the_call`), rate-limit →
invalid not fatal (`test_rate_limit_is_invalid_not_fatal`), auth/permission
errors → `FatalRunnerError` (`test_auth_error_aborts_the_run`,
`test_permission_error_aborts_the_run`), missing-answer → invalid
(`test_missing_answer_is_invalid`), out-of-range noul → invalid
(`test_out_of_range_noul_is_invalid`), and a missing API key raising
`FatalRunnerError` for the right reason, pinned by patching the environment
so the test cannot pass merely because the SDK is uninstalled
(`test_missing_key_is_fatal`).

This whole pattern needs zero real network access, zero real API key, and
zero real `typesafe_sdk` install to run in CI. It is directly reusable: copy
the fake classes and `_wired_runner` helper, replace `QUESTION_KEY` and the
item builder with the joke project's own, and write the same test shapes
against the joke runner.

---

## 8. Repo conventions

- **`pyproject.toml`** (`parsing-experiments/pyproject.toml`): PEP 621
  metadata, `setuptools` build backend, `requires-python = ">=3.12"`,
  `dependencies = []` (stdlib-only runtime), one optional extra:
  `jev = ["typesafe_sdk>=0.7"]`. `packages.find` scoped to `where = ["src"]`
  (a `src/` layout). One console script: `jev-eval = "jev_eval.cli:main"`.
- **Python version:** 3.12+ is required, not just supported — the Python
  oracle uses `ast.parse(..., feature_version=(3, 12))` and the README asks
  for `python3.12 -m venv .venv` explicitly.
- **Venv location:** `.venv/` at the project root
  (`parsing-experiments/.venv/`, not committed — see `.gitignore` below).
  Each experiment directory gets its own venv; there is no shared repo-root
  venv.
- **Env var names:** `JEV_API_KEY` (Jev, harness-owned, not the SDK's own
  `TYPESAFE_API_KEY`), `JEV_MODEL` (override, default `jev-latest`),
  `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL` (default `deepseek-flash`),
  `DEEPSEEK_BASE_URL` (default `https://api.deepseek.com/chat/completions`),
  `GEMINI_API_KEY` or `GOOGLE_API_KEY`, `GEMINI_MODEL` (default
  `gemini-2.5-flash-lite`).
- **`.gitignore`** (seen in the `jev-testing` copy, same rules apply to
  `parsing-experiments`): standard Python ignores
  (`__pycache__/`, `*.py[cod]`, `*.egg-info/`, `.eggs/`, `dist/`, `build/`,
  `.venv/`, `venv/`, caches) plus **`results/`** — every run directory
  (`predictions.jsonl`, `run_meta.json`, `metrics.json`, DeepSeek reasoning
  traces) is local-only and never committed. Note: the *actual*
  `parsing-experiments/results/` directory in this repo currently contains
  real run output on disk (verified directly — `jev-full-200`,
  `deepseek-full-200`, `ds-def-r1..r3`, etc.), consistent with "gitignored,
  not deleted."
- **Commit style** (`git log --oneline` inside `jev-experiments`, 3 commits
  total):
  ```
  1e4c60e docs(articles): rewrite the write-up for a general audience
  43bd903 refactor: rename jev-testing to parsing-experiments
  769aa0b feat: jev-experiments monorepo with the System One evaluation
  ```
  Conventional-commit style: `<type>(<scope>): <description>`, lower case,
  imperative, no trailing period — matching the global `CLAUDE.md` git-prose
  rule (`<type>(<scope>): <description>`, 72-character header limit).
- **Tests:** stdlib `unittest`, no pytest. Run with
  `PYTHONPATH=src python -m unittest discover -s tests -v`. One test file per
  layer: `test_seed_schema.py`, `test_schema_metrics.py`, `test_oracles.py`,
  `test_runners.py`, `test_model_output.py`, `test_cli.py`.

---

## 9. Reuse list for the joke-preference project

### Import or copy near-verbatim

| module / function | where | why it is reusable as-is |
|---|---|---|
| `metrics.score_rows`, `metrics.brier_score`, `metrics.expected_calibration_error` | `src/jev_eval/metrics.py` | Pure functions over plain row dicts (`answer`, `gold`, `invalid`, `p_yes`, `latency_ms`, `task`). No dependency on `Item`, tasks, or providers. Import directly, or copy the file — either is safe, since GEPA scoring needs exactly this kind of calibration read-out on "will Naveen like this joke?" scores. |
| `schema.confidence` | `src/jev_eval/schema.py:247-248` | One-line helper (`max(p_yes, 1-p_yes)`), used by `score_rows`. |
| The Jev client-construction pattern | `runners/jev.py:71-89` (`_build_client`) | Lazy SDK import, explicit `TYPESAFE_API_KEY`-avoidance via an app-owned env var, `FatalRunnerError` on missing key/missing SDK. Copy the shape; rename the env var (e.g. `JOKE_JEV_API_KEY` or reuse `JEV_API_KEY`). |
| The `system_one` call + error-handling shape | `runners/jev.py:91-151` | The whole try/except ladder (`FatalRunnerError` on auth/permission, `InvalidOutput` row on other `TypeSafeError`, invalid on missing/out-of-range `noul`) is provider-agnostic. Only the question dict and the mapping from `noul` to the joke project's own output type need to change. |
| The fake-SDK test harness | `tests/test_runners.py:54-219` (`_FakeNoul`, `_FakeNoulAnswer`, `_FakeUsage`, `_FakeResponse`, `_FakeTypeSafeError` + subclasses, `_FakeClient`, `_fake_sdk`, `_wired_runner`) | Copy wholesale. It needs no changes beyond swapping the item builder and `QUESTION_KEY`. This is the fastest way to get Jev-call tests running with zero network access and zero SDK install requirement. |
| `_http_json` | `runners/llm.py:240-261` | Stdlib-only POST-JSON helper with clean error surfacing. Reusable if the joke project ever calls DeepSeek or another HTTP JSON API directly (e.g. as a GEPA-side judge or baseline). |
| Repo conventions | `pyproject.toml`, `.gitignore`, commit style | Copy the `pyproject.toml` shape (src layout, optional `jev` extra, `requires-python = ">=3.12"`), the `.gitignore` (add `results/`), and the commit header convention. |

### Must be built new

- **`Item`/schema equivalent for jokes.** `schema.item_from_dict` enforces a
  parsing-specific contract (`question == spec.question`, `fault_tags` rules
  tied to a boolean oracle answer). A joke item has no oracle-truth boolean —
  "will Naveen like this joke?" is a preference label from Naveen himself
  (or a proxy), not a compiler exit code. A new schema is needed: probably
  `{id, joke_text, liked: bool, rating?: float, criteria_version, ...}` with
  its own validation rules. The `answer/fault_tags` coupling in
  `schema.item_from_dict` (`schema.py:99-105`) does not transfer.
- **A new "task"/rubric concept.** `tasks.py`'s locked, never-changing
  question is the opposite of what the joke project needs: the whole point
  of the new project is that the `criteria` text sent to Jev's `Noul` is the
  thing GEPA optimizes, i.e. it *must* vary run to run. The parsing project
  deliberately locks the question and never sends `criteria`
  (`runners/jev.py:20-23`); the joke project deliberately does the reverse.
  `QUESTION_KEY`/`instructions` can stay fixed (e.g.
  `"Will Naveen like this joke?"`), but a new mechanism is needed to pass a
  *variable* `criteria` string into `Noul(instructions=..., criteria=...)`
  per GEPA candidate — the parsing runner has no such parameter threaded
  through at all.
- **No oracle layer.** `oracles/` runs real parsers/compilers for
  ground truth. There is no equivalent for joke preference; ground truth is
  Naveen's own judgment, collected some other way (a rating log, a small
  labelled set, etc.). This needs a new, human-sourced dataset instead of a
  generator+oracle pipeline.
- **GEPA integration itself.** Nothing in this harness touches DSPy or GEPA.
  The joke project needs a new module that (a) defines a DSPy signature/
  metric wrapping a Jev call (reusing the `runners/jev.py` pattern) as the
  scored program, and (b) wires GEPA to mutate the `criteria` text as the
  thing being optimized. `metrics.score_rows`'s accuracy/Brier/ECE machinery
  is a good scoring signal to reuse for that, once liked/not-liked labels
  exist, but GEPA's optimizer loop, mutation strategy, and pareto tracking
  are new work with no analog here.
- **CLI subcommands `generate`/`validate`.** These are parsing-experiment
  specific (template-based artifact generation, oracle-checked labels). The
  joke project's CLI (if it has one) needs different verbs — collecting
  ratings, running a GEPA optimization pass, scoring a `criteria` candidate
  against held-out labels — built from scratch, though `run`/`score`'s
  overall shape (loop over items, write `predictions.jsonl` +
  `run_meta.json`, then call `score_rows`) is a good template to imitate.

---

## Summary of file:line references used in this report

- `src/jev_eval/tasks.py:23-58` — task registry.
- `src/jev_eval/schema.py:24-64,67-129,211-248` — `Item`, `item_from_dict`,
  `parse_model_output`, `confidence`.
- `src/jev_eval/runners/jev.py:1-27,40-69,71-151,153-169` — Jev runner:
  docstring/policy, constants, client build, `complete`, `_mock`.
- `src/jev_eval/runners/llm.py:20-39,49-59,78-97,100-149,151-261` — providers,
  keys, prompt template, `CallResult`, `LlmRunner`, DeepSeek/Gemini calls,
  `_http_json`.
- `src/jev_eval/metrics.py:11-140` — `_mean`, `_quantile`,
  `expected_calibration_error`, `brier_score`, `score_rows`.
- `src/jev_eval/cli.py:80-133,136-204` — `_telemetry`, `_prediction_row`,
  `cmd_run`.
- `src/jev_eval/runners/__init__.py:1-15` — `PROVIDERS`, `make_runner`.
- `tests/test_runners.py:1-219` — fake SDK, `_wired_runner`, all Jev tests.
- `pyproject.toml` — build config, extras, `requires-python`.
- `docs/evaluation.md`, `docs/hard-set-v2.md`, `docs/reasoning-tokens.md`,
  `docs/run-variance.md`, `README.md`, `articles/system-one-vs-reasoning.md`
  — findings and numbers cited in sections 6 and elsewhere.
