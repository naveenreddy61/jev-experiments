"""Jev runner: TypeSafe System One adapter.

Jev is not a text LLM. It returns a typed Noul answer, so this runner never
parses text and never uses the JSON rubric that the LLM baselines need. The
SDK gives a single float:

    NoulAnswer(type="noul", noul=0.92)

`noul` is P(yes) for the question in `instructions`. There is no boolean and
no separate confidence field; see https://docs.typesafe.ai/primitives/noul.
The mapping to the harness contract is therefore:

    p_yes  = answers[QUESTION_KEY].noul
    answer = p_yes >= YES_THRESHOLD

YES_THRESHOLD is a policy choice of this harness, not a model output. 0.5
keeps the boolean comparable with the LLM baselines, which self-report a
boolean around the same midpoint.

The question text is the locked task rubric from `jev_eval.tasks`, sent
unchanged. No `criteria` are attached: the v1 seed labels are reproducible
against a fixed oracle and a fixed question, so changing the question shape
would change what is measured. Criteria belong in a separate comparison run.

The SDK is an optional dependency (`pip install -e '.[jev]'`). It is imported
lazily so the LLM providers keep working without it.
"""

from __future__ import annotations

import json
import os
import time

from jev_eval.schema import InvalidOutput, Item, ModelOutput
from jev_eval.tasks import get_task
from jev_eval.runners.llm import CallResult, required_api_key

#: Harness policy: the noul probability at or above which the boolean is yes.
YES_THRESHOLD = 0.5

#: Question key sent to the API. Internal to the request; never shown to the model.
QUESTION_KEY = "parses"

#: Model id. `jev-latest` tracks the current release.
DEFAULT_MODEL = "jev-latest"

#: Environment override for the model id, for pinning a version in a comparison.
MODEL_ENV = "JEV_MODEL"


class FatalRunnerError(RuntimeError):
    """An error that no per-item retry can fix, such as a bad API key.

    The run aborts instead of writing hundreds of invalid rows, which would
    otherwise look like the model scoring at chance.
    """


class JevRunner:
    def __init__(self, *, mock: bool = False, timeout: float = 60.0):
        self.provider = "jev"
        self.mock = mock
        self.timeout = timeout
        self.model = os.environ.get(MODEL_ENV) or DEFAULT_MODEL
        self._client = None
        self._sdk = None
        if not mock:
            self._client, self._sdk = self._build_client()

    def _build_client(self):
        try:
            import typesafe_sdk as sdk
        except ImportError as exc:  # pragma: no cover - depends on install extras
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
        # The SDK reads TYPESAFE_API_KEY by default. This harness owns the key
        # name, so pass it explicitly rather than mutating the environment.
        client = sdk.TypeSafeClient(api_key=key, timeout=self.timeout)
        return client, sdk

    def complete(self, item: Item) -> CallResult:
        if self.mock:
            return self._mock(item)

        sdk = self._sdk
        question = get_task(item.task).question
        t0 = time.perf_counter()
        try:
            response = self._client.system_one(
                item.artifact,
                {QUESTION_KEY: sdk.Noul(instructions=question)},
                model=self.model,
                timeout=self.timeout,
            )
        except (sdk.TypeSafeAuthenticationError, sdk.TypeSafePermissionDeniedError) as exc:
            raise FatalRunnerError(f"{type(exc).__name__}: {exc}") from exc
        except sdk.TypeSafeError as exc:
            # Timeouts, rate limits, 5xx and validation errors are per-item
            # failures. They stay invalid and are never imputed.
            return CallResult(
                parsed=InvalidOutput(reason=f"{type(exc).__name__}: {exc}", raw=""),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                raw="",
                error=type(exc).__name__,
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        answer = response.answers.get(QUESTION_KEY)
        raw = json.dumps(
            {
                "model": response.model,
                "answers": {k: v.model_dump() for k, v in response.answers.items()},
                "usage": response.usage.model_dump(),
            },
            separators=(",", ":"),
        )
        if answer is None or getattr(answer, "type", None) != "noul":
            return CallResult(
                parsed=InvalidOutput(
                    reason=f"response has no noul answer under {QUESTION_KEY!r}",
                    raw=raw,
                ),
                latency_ms=latency_ms,
                raw=raw,
                error="missing_noul",
            )

        p_yes = float(answer.noul)
        if not 0.0 <= p_yes <= 1.0:
            return CallResult(
                parsed=InvalidOutput(reason=f"noul {p_yes} out of [0, 1]", raw=raw),
                latency_ms=latency_ms,
                raw=raw,
                error="noul_out_of_range",
            )

        return CallResult(
            parsed=ModelOutput(answer=p_yes >= YES_THRESHOLD, p_yes=p_yes),
            latency_ms=latency_ms,
            raw=raw,
        )

    def _mock(self, item: Item) -> CallResult:
        """Offline dry-run over the same mapping the live path uses."""
        p_yes = 0.72 if hash(item.id) % 2 == 0 else 0.28
        raw = json.dumps(
            {
                "model": self.model,
                "answers": {QUESTION_KEY: {"type": "noul", "noul": p_yes}},
                "usage": {"input_tokens": None, "output_tokens": None},
            },
            separators=(",", ":"),
        )
        t0 = time.perf_counter()
        return CallResult(
            parsed=ModelOutput(answer=p_yes >= YES_THRESHOLD, p_yes=p_yes),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            raw=raw,
        )
