"""Recover token usage from eval runs recorded before the loop counted it.

Runs finished before token accounting existed still carry everything needed:
vLLM attempts stored ``prompt_ids`` and ``output_ids`` per turn, and API
attempts stored the provider's usage payload. This reads those back, writes a
per-attempt breakdown to ``<results-dir>/token_usage.json``, and adds the run
total to ``summary.json`` under the same ``token_usage`` key that new runs
write, so downstream code reads one field either way.

What cannot be recovered is cache accounting: prefix-cache hits were not
recorded per call, so a backfilled attempt reports ``cached_input_tokens: 0``
with ``source: "token_ids"``. That means "not measured", not "no cache hits" —
the billed input for those runs was lower than ``input_tokens`` suggests.

Usage: backfill_token_usage.py <results-dir> [<results-dir> ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from officeqa_rl.token_usage import add_usage, episode_usage, per_task_means  # noqa: E402
from officeqa_rl.types import GeneratedTurn  # noqa: E402


def _count_ids(text: str, key: str) -> tuple[int, int]:
    """Count entries across every flat integer array stored under ``key``.

    Scanning the raw text instead of parsing avoids materializing millions of
    Python ints per attempt; the arrays hold plain integers, so the first ``]``
    after the opening bracket closes them.
    """
    marker = f'"{key}":'
    total = arrays = position = 0
    while True:
        start = text.find(marker, position)
        if start < 0:
            return total, arrays
        cursor = start + len(marker)
        while cursor < len(text) and text[cursor] in " \t\r\n":
            cursor += 1
        if cursor < len(text) and text[cursor] == "[":
            end = text.find("]", cursor)
            if end < 0:
                return total, arrays
            body = text[cursor + 1 : end]
            total += body.count(",") + 1 if body.strip() else 0
            arrays += 1
            position = end
        else:
            position = cursor


def attempt_usage(path: Path) -> dict[str, Any] | None:
    """Token usage for one attempt file, or None when nothing was recorded."""
    text = path.read_text(encoding="utf-8", errors="replace")
    prompt_tokens, turns = _count_ids(text, "prompt_ids")
    output_tokens, _ = _count_ids(text, "output_ids")
    if turns and prompt_tokens:
        return {
            "prompt_tokens": prompt_tokens,
            "cached_input_tokens": 0,
            "input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": 0,
            "model_calls": turns,
            "final_prompt_tokens": 0,
            "source": "token_ids",
        }

    # No token ids: an API backend may still have stored per-turn usage.
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    recorded = payload.get("token_usage")
    if isinstance(recorded, dict):
        return recorded
    raw_turns = (payload.get("metadata") or {}).get("generated_turns") or []
    turn_objects = [
        GeneratedTurn(
            text="",
            prompt_ids=turn.get("prompt_ids"),
            output_ids=turn.get("output_ids"),
            usage=turn.get("usage"),
        )
        for turn in raw_turns
    ]
    usage = episode_usage(turn_objects)
    return usage if usage["model_calls"] and usage["source"] != "none" else None


def backend_stats_total(path: Path) -> dict[str, Any] | None:
    """Run-level usage from a shared backend's cumulative stats snapshot.

    The Gemini and Kimi backends accumulate usage on the backend object and
    every attempt payload embeds a snapshot of it, so the largest snapshot in
    the directory is the run total. It cannot be split per attempt.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except ValueError:
        return None
    stats = ((payload.get("metadata") or {}).get("generation_backend") or {}).get("stats")
    if not isinstance(stats, dict):
        return None
    totals = stats.get("usage_totals")
    if not isinstance(totals, dict):
        return None
    prompt = int(totals.get("prompt_token_count") or totals.get("prompt_tokens") or 0)
    cached = int(
        totals.get("cached_content_token_count") or totals.get("cached_tokens") or 0
    )
    output = int(
        totals.get("candidates_token_count") or totals.get("completion_tokens") or 0
    )
    if not prompt and not output:
        return None
    return {
        "prompt_tokens": prompt,
        "cached_input_tokens": min(cached, prompt),
        "input_tokens": prompt - min(cached, prompt),
        "output_tokens": output,
        "reasoning_tokens": int(totals.get("thoughts_token_count") or 0),
        "model_calls": int(stats.get("n_calls") or 0),
        "source": "backend_stats_snapshot",
    }


def attempt_files(directory: Path) -> list[Path]:
    """Attempt payloads, matching how the evaluator selects them.

    A pass@k run writes ``<uid>.s<i>.json``; when those exist, bare
    ``<uid>.json`` files are leftovers from an earlier single-attempt run and
    are not part of the reported results, so they are excluded from its cost.
    """
    paths = sorted(
        path
        for path in directory.glob("*.json")
        if path.name not in {"summary.json", "token_usage.json"}
    )
    sampled = [path for path in paths if re.search(r"\.s\d+$", path.stem)]
    return sampled or paths


def backfill(directory: Path, workers: int) -> dict[str, Any]:
    paths = attempt_files(directory)
    per_attempt: dict[str, Any] = {}
    total: dict[str, Any] | None = None
    sources: set[str] = set()
    if paths:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for path, usage in zip(paths, pool.map(attempt_usage, paths, chunksize=4)):
                if usage is None:
                    continue
                per_attempt[path.stem] = usage
                sources.add(str(usage.get("source", "unknown")))
                total = add_usage(total, usage)
    if total is not None:
        total["source"] = sources.pop() if len(sources) == 1 else "mixed"

    if total is None and paths:
        # Backends that only kept cumulative stats: take the largest snapshot.
        best = None
        for path in sorted(paths, key=lambda item: item.stat().st_size, reverse=True)[:64]:
            candidate = backend_stats_total(path)
            if candidate and candidate["prompt_tokens"] > (
                best["prompt_tokens"] if best else 0
            ):
                best = candidate
        if best is not None:
            total = dict(best)
            total["episodes"] = len(paths)

    if total is None:
        return {"directory": str(directory), "attempts": 0, "token_usage": None}

    tasks = None
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        try:
            tasks = len(json.loads(summary_path.read_text(encoding="utf-8"))["results"])
        except (ValueError, KeyError):
            tasks = None
    tasks = tasks or len({name.split(".s")[0] for name in per_attempt}) or None
    if tasks:
        total.update(per_task_means(total, tasks))
    return {
        "directory": str(directory),
        "attempts": len(per_attempt) or len(paths),
        "tasks": tasks,
        "token_usage": total,
        "per_attempt": per_attempt,
    }


def write_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="+", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the totals without writing token_usage.json or summary.json",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args(argv)

    for directory in args.results_dir:
        if not directory.is_dir():
            print(f"{directory}: not a directory", file=sys.stderr)
            continue
        report = backfill(directory, args.workers)
        usage = report["token_usage"]
        if usage is None:
            print(f"{directory.name}: no token usage recoverable")
            continue
        print(
            "%-52s attempts=%-5d prompt=%13d input=%13d cached=%11d output=%10d "
            "calls=%6d source=%s"
            % (
                directory.name,
                report["attempts"],
                usage["prompt_tokens"],
                usage["input_tokens"],
                usage["cached_input_tokens"],
                usage["output_tokens"],
                usage.get("model_calls", 0),
                usage.get("source", "?"),
            )
        )
        if report.get("tasks"):
            print(
                "%-52s per task: prompt=%12.1f output=%10.1f calls=%6.1f  (%d tasks)"
                % (
                    "",
                    usage.get("prompt_tokens_per_task", 0.0),
                    usage.get("output_tokens_per_task", 0.0),
                    usage.get("model_calls_per_task", 0.0),
                    report["tasks"],
                )
            )
        if args.dry_run:
            continue
        try:
            _write_report(directory, report, usage)
        except OSError as error:
            print(f"{directory.name}: could not write ({error}); numbers above still stand")
    return 0


def _write_report(directory: Path, report: dict[str, Any], usage: dict[str, Any]) -> None:
    write_atomic(
        directory / "token_usage.json",
        {key: value for key, value in report.items() if key != "directory"},
    )
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except ValueError:
        return
    summary["token_usage"] = usage
    write_atomic(summary_path, summary)


if __name__ == "__main__":
    raise SystemExit(main())
