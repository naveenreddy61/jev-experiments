# jev-eval

Atomic yes/no capability-frontier eval suite for TypeSafe AI’s **Jev** model.

v1 is a harness first: generators, a committed seed set (~50 items × 4 tasks),
oracles, and runners. Iron out the loop on small LLMs (Gemini Flash Lite,
DeepSeek Flash), then compare Jev on the same items, contract and metrics.

Parse and run are never mixed. `python.exec` / `will-run` are reserved future
task ids only.

## Locked v1 tasks

| task_id | input | exact question | oracle | yes means |
|---|---|---|---|---|
| `latex.parse` | LaTeX fragment | Does this parse as LaTeX? | `pdflatex` dry-run (article wrapper) | exit 0 |
| `bash.parse` | Shell snippet | Does this parse as bash? | `bash -n` | exit 0 |
| `python.parse` | Python source | Does this parse as Python 3.12? | `ast.parse` only (never `exec`) | no `SyntaxError` |
| `c.compile` | Single `.c` translation unit | Does this compile as C11 (no link)? | `gcc -std=c11 -fsyntax-only` | exit 0 |

Answers are **oracle labels only**. Never hand-label.

## Install

Requires **Python 3.12+**. Runtime extras are stdlib-only.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or run without installing:

```bash
export PYTHONPATH=src
python3 -m jev_eval --help
```

### System tools (oracles)

| tool | used by | install (Debian/Ubuntu) |
|---|---|---|
| Python 3.12 | `python.parse` (`ast.parse`, `feature_version=(3, 12)`) | `python3.12` |
| GNU bash | `bash.parse` (`bash -n`) | `bash` |
| gcc | `c.compile` (`gcc -std=c11 -fsyntax-only`) | `build-essential` / `gcc` |
| pdflatex | `latex.parse` | `sudo apt-get install -y --no-install-recommends texlive-latex-base` |

LaTeX fragments are wrapped as:

```latex
\documentclass{article}
\begin{document}
<fragment>
\end{document}
```

and checked with:

```text
pdflatex -interaction=nonstopmode -halt-on-error -draftmode \
  -output-directory <tmp> fragment.tex
```

This suite pins **pdflatex** (TeX Live) so seed labels stay reproducible.
tectonic is a reasonable headless alternative on other machines, but swapping
the engine can flip labels — keep the oracle fixed when comparing models.

## Workflow

```bash
# 1. Generate oracle-validated items (scale n later)
python -m jev_eval generate --task all --n 50 --out data/seed

# 2. Re-check committed / generated items against oracles
python -m jev_eval validate data/seed

# 3. Query a baseline (mock = offline JSON-contract dry-run)
python -m jev_eval run --provider gemini-flash-lite --data data/seed --out results/gemini --mock
python -m jev_eval run --provider deepseek-flash --data data/seed --out results/deepseek

# 4. Score predictions (accuracy, invalid-rate, latency, p_yes calibration)
python -m jev_eval score results/gemini
```

`--provider` is one of: `gemini-flash-lite` | `deepseek-flash` | `jev`.

If an API key is missing, `run` is a no-op (exit 2) with a clear message.
Use `--mock` to exercise output parsing without network access.

## Item schema (jsonl)

Each line:

```json
{
  "id": "python.parse.00042",
  "task": "python.parse",
  "difficulty": "easy",
  "artifact": "...source...",
  "question": "Does this parse as Python 3.12?",
  "answer": true,
  "fault_tags": [],
  "oracle_meta": {"tool": "ast.parse", "version_note": "...", "exit_code": 0}
}
```

- `answer` is a bool from the oracle only.
- `fault_tags` is empty when `answer` is true; mutation ids when false.
- Seed target: ~50 items/task, ~50/50 yes/no, easy/medium/hard by construction.
- Generators add unique comment noise and prefer with/without-fault pairs.

Committed seed files live in [`data/seed/`](data/seed/).

## Model output contract

Every model (LLM and Jev) must produce:

```json
{"answer": true, "p_yes": 0.87}
```

- `answer`: bool
- `p_yes`: float in `[0, 1]` — the model’s P(yes)
- missing or garbage → mark `invalid`, **do not impute**
- metrics derive `confidence = max(p_yes, 1 - p_yes)` when useful

The LLM prompt is a fixed task rubric plus the artifact, and requests this
JSON schema. Gemini is called with `responseMimeType=application/json` and a
response schema; DeepSeek uses `response_format=json_object`.

### API keys (env only — never commit secrets)

| provider | env | default model |
|---|---|---|
| `gemini-flash-lite` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `gemini-2.5-flash-lite` (`GEMINI_MODEL`) |
| `deepseek-flash` | `DEEPSEEK_API_KEY` | `deepseek-flash` (`DEEPSEEK_MODEL`) |
| `jev` | `JEV_API_KEY` | `jev-latest` (`JEV_MODEL`) |

Optional: `DEEPSEEK_BASE_URL` (default `https://api.deepseek.com/chat/completions`).

## Jev provider

Jev is a TypeSafe **System One** model, not a text LLM. It returns a typed
answer, so the Jev path does no text parsing and never uses the JSON rubric
that the LLM baselines need.

Install the extra and set the key:

```bash
pip install -e '.[jev]'     # adds typesafe_sdk
export JEV_API_KEY=...      # this harness owns the key name and passes it
                            # to the SDK explicitly (the SDK's own default
                            # env var is TYPESAFE_API_KEY)
python -m jev_eval run --provider jev --data data/seed --out results/jev
```

`src/jev_eval/runners/jev.py` asks one [Noul](https://docs.typesafe.ai/primitives/noul)
question per item. The `instructions` are the locked task question, sent
unchanged:

```python
from typesafe_sdk import TypeSafeClient, Noul

client = TypeSafeClient(api_key=key, timeout=timeout)
response = client.system_one(
    item.artifact,                                   # state
    {"parses": Noul(instructions=task.question)},
    model="jev-latest",
)
p_yes = response.answers["parses"].noul              # float in [0, 1]
```

A Noul answer is a single probability:

```json
{"type": "noul", "noul": 0.92}
```

There is **no** boolean and **no** separate confidence field, so the mapping
onto the harness contract is direct:

```text
p_yes  = noul
answer = p_yes >= YES_THRESHOLD      # YES_THRESHOLD = 0.5
```

`YES_THRESHOLD` is a policy choice of this harness, not a model output. Change
it in `runners/jev.py`, or re-threshold the raw `p_yes` in `predictions.jsonl`
without spending new calls.

v1 sends **no** `criteria`. The seed labels are reproducible only against a
fixed oracle *and* a fixed question, so adding criteria changes what is
measured. Treat a criteria variant as a second run to compare, not an edit.

### Error handling

| condition | result |
|---|---|
| authentication / permission denied | run aborts (exit 2); partial predictions are kept |
| timeout, rate limit, 5xx, validation error | row marked `invalid`; exception class in `error` |
| answer missing or `noul` outside [0, 1] | row marked `invalid` |

An auth failure aborts instead of writing hundreds of invalid rows, which
would otherwise read as the model scoring at chance.

## Results

- [Evaluation, full 200 items](docs/evaluation.md) — Jev 0.785, `deepseek-flash`
  0.970, both with zero invalid output. Every error from either model is a false
  positive, and DeepSeek's errors are a strict subset of Jev's. Jev is ~7x
  faster. Both failed every `bad_double_bracket` item in that run; a later
  DeepSeek run caught 1 of 5, so the tag is a near-miss, not an absolute wall.
- [DeepSeek reasoning tokens](docs/reasoning-tokens.md) — reasoning length
  tracks difficulty, not artifact size, and does not predict errors. 98% of
  output tokens are reasoning, and roughly a third of the budget goes to the
  top 5% of items.
- [Run-to-run variance](docs/run-variance.md) — three identical DeepSeek runs
  on the hard set score 0.9187, 0.9313 and 0.9437. All the spread comes from
  11 items, and those items cost 4.6x the reasoning tokens of the stable ones.
  Reasoning length predicts **instability**, which error rate alone does not
  show.
- [Extended hard set](docs/hard-set-v2.md) — 160 longer items added. On hard
  items only, Jev falls to 0.607 and DeepSeek holds about 0.93 (single-run
  figure 0.940; see the variance note above). Jev's recall on
  broken artifacts drops to 0.150: it detects local faults and misses
  block-structure faults across distance.

## Metrics

`metrics.py` reports:

- **accuracy** on valid outputs only
- **invalid-rate** (schema/transport failures; not imputed as answers)
- **latency** mean / p50 / p95
- **calibration** when `p_yes` is present: Brier score, expected calibration
  error (ECE), reliability table, mean confidence

## Layout

```text
README.md
pyproject.toml
src/jev_eval/
  tasks.py
  schema.py
  generators/
  oracles/
  runners/
    llm.py
    jev.py
  metrics.py
  cli.py
data/seed/
scripts/generate_seed.sh
tests/
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Oracle tests use fixtures under `tests/fixtures/` and require the system
tools above (pdflatex, gcc, bash, Python 3.12).

## Regenerating the seed set

```bash
./scripts/generate_seed.sh 50
```

Labels are whatever the local oracles emit. Re-run `validate` after changing
generator templates or the LaTeX engine.
