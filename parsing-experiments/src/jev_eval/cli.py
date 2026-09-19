"""CLI: generate, validate, run, score."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from jev_eval import __version__
from jev_eval.generators import generate, write_jsonl
from jev_eval.metrics import score_rows
from jev_eval.oracles import OracleToolMissing, check
from jev_eval.runners import PROVIDERS, make_runner
from jev_eval.runners.jev import FatalRunnerError
from jev_eval.runners.llm import missing_key_message, required_api_key
from jev_eval.schema import InvalidOutput, ModelOutput, SchemaError, load_items
from jev_eval.tasks import TASK_IDS, get_task


def _iter_jsonl_paths(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        paths = sorted(target.glob("*.jsonl"))
        if not paths:
            raise SystemExit(f"no .jsonl files in {target}")
        return paths
    raise SystemExit(f"not a file or directory: {target}")


def cmd_generate(args: argparse.Namespace) -> int:
    tasks = list(TASK_IDS) if args.task == "all" else [args.task]
    for task_id in tasks:
        get_task(task_id)
        print(f"generating {args.n} items for {task_id} (seed={args.seed})...", flush=True)
        items = generate(task_id, args.n, args.seed)
        out = Path(args.out) / f"{task_id}.jsonl"
        write_jsonl(items, out)
        n_yes = sum(1 for it in items if it.answer)
        print(
            f"  wrote {len(items)} items -> {out} "
            f"(yes={n_yes} no={len(items) - n_yes})"
        )
        if len(items) < args.n:
            print(f"  warning: requested {args.n}, produced {len(items)}", file=sys.stderr)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    paths = _iter_jsonl_paths(Path(args.data))
    try:
        items = load_items(paths)
    except SchemaError as exc:
        print(f"schema error: {exc}", file=sys.stderr)
        return 1

    failures = 0
    for item in items:
        try:
            result = check(item.task, item.artifact)
        except OracleToolMissing as exc:
            print(f"oracle unavailable: {exc}", file=sys.stderr)
            return 2
        if result.ok != item.answer:
            failures += 1
            print(
                f"FAIL {item.id}: label={item.answer} oracle={result.ok} "
                f"exit={result.exit_code} {result.detail[:180]}"
            )
    n_yes = sum(1 for it in items if it.answer)
    print(
        f"validated {len(items)} items across {len(paths)} file(s); "
        f"yes={n_yes} no={len(items) - n_yes}; mismatches={failures}"
    )
    return 1 if failures else 0


def _telemetry(call) -> dict:
    """Provider token accounting, flattened for analysis.

    `reasoning_tokens` is what a reasoning model spent before answering.
    Latency only approximates it, because latency also carries queueing and
    network time. `reasoning_chars` sizes the chain when the API returns the
    text; the text itself is kept so failures can be read, not only counted.
    """
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


def _prediction_row(item, call, provider: str) -> dict:
    parsed = call.parsed
    if isinstance(parsed, ModelOutput):
        return {
            "id": item.id,
            "task": item.task,
            "gold": item.answer,
            "provider": provider,
            "invalid": False,
            "invalid_reason": None,
            "answer": parsed.answer,
            "p_yes": parsed.p_yes,
            "confidence": max(parsed.p_yes, 1.0 - parsed.p_yes),
            "latency_ms": call.latency_ms,
            "raw": call.raw,
            "error": call.error,
            **_telemetry(call),
        }
    assert isinstance(parsed, InvalidOutput)
    return {
        "id": item.id,
        "task": item.task,
        "gold": item.answer,
        "provider": provider,
        "invalid": True,
        "invalid_reason": parsed.reason,
        "answer": None,
        "p_yes": None,
        "confidence": None,
        "latency_ms": call.latency_ms,
        "raw": parsed.raw or call.raw,
        "error": call.error,
        **_telemetry(call),
    }


def cmd_run(args: argparse.Namespace) -> int:
    provider = args.provider
    _, key = required_api_key(provider)
    if not args.mock and not key:
        print(missing_key_message(provider), file=sys.stderr)
        return 2

    paths = _iter_jsonl_paths(Path(args.data))
    try:
        items = load_items(paths)
    except SchemaError as exc:
        print(f"schema error: {exc}", file=sys.stderr)
        return 1
    if args.difficulty:
        wanted = set(args.difficulty)
        items = [it for it in items if it.difficulty in wanted]
        if not items:
            print(
                f"no items with difficulty {sorted(wanted)} in {args.data}",
                file=sys.stderr,
            )
            return 1
    if args.limit is not None:
        items = items[: args.limit]

    try:
        runner = make_runner(provider, mock=args.mock, timeout=args.timeout)
    except FatalRunnerError as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    rows = []
    with pred_path.open("w", encoding="utf-8") as fh:
        for item in items:
            try:
                call = runner.complete(item)
            except FatalRunnerError as exc:
                print(f"fatal: {exc}", file=sys.stderr)
                print(
                    f"aborted after {len(rows)} item(s); partial predictions "
                    f"kept at {pred_path}",
                    file=sys.stderr,
                )
                return 2
            row = _prediction_row(item, call, provider)
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")
            status = "invalid" if row["invalid"] else ("yes" if row["answer"] else "no")
            print(f"{item.id}\t{status}\t{row['latency_ms']:.1f}ms")

    meta = {
        "provider": provider,
        "mock": bool(args.mock),
        "n": len(rows),
        "difficulty": sorted(set(args.difficulty)) if args.difficulty else "all",
        "data": [str(p) for p in paths],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": __version__,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    metrics = score_rows(rows)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {pred_path}")
    print(json.dumps({k: metrics[k] for k in ("n", "n_valid", "n_invalid", "invalid_rate", "accuracy")}, indent=2))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    target = Path(args.results)
    if target.is_dir():
        pred = target / "predictions.jsonl"
        if not pred.is_file():
            raise SystemExit(f"no predictions.jsonl in {target}")
        target = pred
    rows = []
    with target.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{target}:{lineno}: {exc}") from exc
    metrics = score_rows(rows)
    text = json.dumps(metrics, indent=2) + "\n"
    print(text, end="")
    if args.out:
        out = Path(args.out)
        if out.suffix != ".json" and (out.exists() and out.is_dir() or out.suffix == ""):
            out.mkdir(parents=True, exist_ok=True)
            out = out / "metrics.json"
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    elif target.parent.is_dir():
        (target.parent / "metrics.json").write_text(text, encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jev-eval",
        description="Atomic yes/no capability-frontier eval suite for TypeSafe Jev.",
    )
    parser.add_argument("--version", action="version", version=f"jev-eval {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate oracle-validated items")
    g.add_argument("--task", default="all", choices=["all", *TASK_IDS])
    g.add_argument("--n", type=int, default=50, help="items per task")
    g.add_argument("--out", default="data/seed", help="output directory")
    g.add_argument("--seed", type=int, default=1)
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("validate", help="re-check items against oracles")
    v.add_argument("data", help="jsonl file or directory")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("run", help="query a model over a dataset")
    r.add_argument("--provider", required=True, choices=PROVIDERS)
    r.add_argument("--data", default="data/seed")
    r.add_argument("--out", default="results/latest")
    r.add_argument("--mock", action="store_true", help="offline dry-run of output parsing")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument(
        "--difficulty",
        action="append",
        choices=["easy", "medium", "hard"],
        help="keep only these difficulties; repeatable (default: all)",
    )
    r.add_argument("--timeout", type=float, default=60.0)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("score", help="compute accuracy / calibration metrics")
    s.add_argument("results", help="predictions.jsonl or a run directory")
    s.add_argument("--out", default=None, help="optional metrics.json path or directory")
    s.set_defaults(func=cmd_score)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except OracleToolMissing as exc:
        print(f"oracle unavailable: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0
