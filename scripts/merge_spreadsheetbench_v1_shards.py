#!/usr/bin/env python3
"""Combine completed SSB1 task shards and rebuild one canonical summary.

The shard directories remain untouched. Attempt JSONs are rewritten only in
the destination so absolute artifact paths point into the canonical tree;
large output artifacts are hard-linked when possible and copied otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import errno
import filecmp
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Sequence

from officeqa_rl.spreadsheetbench_v1 import load_spreadsheetbench_v1_tasks
from run_spreadsheetbench_v1_eval import parse_args as parse_eval_args
from run_spreadsheetbench_v1_eval import run_full_eval


ATTEMPT_RE = re.compile(r"^(?P<uid>.+)\.s(?P<sample>\d+)\.json$")
ARTIFACT_DIRS = ("outputs", "solutions", "workspaces")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _attempts(directory: Path) -> dict[tuple[str, int], Path]:
    attempts: dict[tuple[str, int], Path] = {}
    for path in directory.glob("*.s*.json"):
        match = ATTEMPT_RE.fullmatch(path.name)
        if match is None:
            continue
        key = (match.group("uid"), int(match.group("sample")))
        if key in attempts:
            raise RuntimeError(f"Duplicate attempt key in {directory}: {key}")
        attempts[key] = path
    return attempts


def _replace_prefix(value: Any, source: str, destination: str) -> Any:
    if isinstance(value, str):
        if value == source:
            return destination
        if value.startswith(source + os.sep):
            return destination + value[len(source) :]
        return value
    if isinstance(value, list):
        return [_replace_prefix(item, source, destination) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_prefix(item, source, destination)
            for key, item in value.items()
        }
    return value


def _write_attempt(source: Path, source_root: Path, destination: Path, output: Path) -> None:
    payload = _replace_prefix(
        _load_json(source), str(source_root.resolve()), str(output.resolve())
    )
    encoded = json.dumps(payload, indent=2) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"Canonical attempt collision: {destination}")
        return
    destination.write_text(encoded, encoding="utf-8")


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not filecmp.cmp(source, destination, shallow=False):
            raise RuntimeError(f"Canonical artifact collision: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno not in {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.EMLINK,
            errno.ENOTSUP,
        }:
            raise
        shutil.copy2(source, destination)


def _copy_artifacts(shards: list[Path], output: Path) -> None:
    claimed: dict[tuple[str, Path], Path] = {}
    for shard in shards:
        for dirname in ARTIFACT_DIRS:
            root = shard / dirname
            if not root.is_dir():
                continue
            for source in root.rglob("*"):
                if not source.is_file():
                    continue
                relative = source.relative_to(root)
                key = (dirname, relative)
                previous = claimed.get(key)
                if previous is not None and previous != source:
                    raise RuntimeError(
                        f"Artifact key is present in two shards: {dirname}/{relative}"
                    )
                claimed[key] = source
                _link_or_copy(source, output / dirname / relative)


def _validate_summaries(
    shards: list[Path], samples_per_task: int
) -> list[dict[str, Any]]:
    summaries = [_load_json(shard / "summary.json") for shard in shards]
    shard_indices: set[int] = set()
    for shard, summary in zip(shards, summaries, strict=True):
        if summary.get("benchmark") != "spreadsheetbench-v1":
            raise RuntimeError(f"Not an SSB1 summary: {shard}")
        if summary.get("samples_per_task") != samples_per_task:
            raise RuntimeError(f"Unexpected samples_per_task in {shard}")
        if summary.get("failed"):
            raise RuntimeError(f"Shard finished with failed attempts: {shard}")
        if summary.get("completed") != summary.get("total_instructions"):
            raise RuntimeError(f"Shard is incomplete: {shard}")
        shard_indices.add(int(summary.get("shard_index", -1)))
    if shard_indices != set(range(len(shards))):
        raise RuntimeError(f"Unexpected shard indices: {sorted(shard_indices)}")
    return summaries


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    shards = [path.resolve() for path in args.shard_dir]
    output = args.output_dir.resolve()
    if output in shards:
        raise RuntimeError("The canonical output directory cannot be a shard directory")
    summaries = _validate_summaries(shards, args.samples_per_task)

    tasks = load_spreadsheetbench_v1_tasks(args.data_root)
    expected = {
        (task.uid, sample)
        for task in tasks
        for sample in range(args.samples_per_task)
    }
    combined: dict[tuple[str, int], tuple[Path, Path]] = {}
    for shard in shards:
        for key, path in _attempts(shard).items():
            if key in combined:
                raise RuntimeError(f"Attempt key is present in two shards: {key}")
            combined[key] = (shard, path)
    actual = set(combined)
    if actual != expected:
        missing = sorted(expected - actual)[:10]
        extra = sorted(actual - expected)[:10]
        raise RuntimeError(
            f"Expected {len(expected)} attempts, found {len(actual)}; "
            f"missing={missing}, extra={extra}"
        )

    output.mkdir(parents=True, exist_ok=True)
    for (_uid, _sample), (shard, source) in combined.items():
        _write_attempt(source, shard, output / source.name, output)
    _copy_artifacts(shards, output)

    # Every expected attempt was validated above, so this is a fully-resumed
    # evaluation: run_full_eval only reconstructs metrics and never connects
    # to the placeholder Vertex endpoint supplied for argument validation.
    eval_args = parse_eval_args(
        [
            "--data-root",
            str(args.data_root),
            "--generation-backend",
            "vertex",
            "--model",
            args.model,
            "--output-dir",
            str(output),
            "--samples-per-task",
            str(args.samples_per_task),
            "--temperature",
            str(args.temperature),
            "--failure-retries",
            "0",
            "--vertex-url",
            "http://unused.invalid/generate",
        ]
    )
    summary = asyncio.run(run_full_eval(eval_args))
    if (
        summary["total_instructions"] != len(tasks)
        or summary["completed"] != len(tasks)
        or summary["scored_attempts"] != len(expected)
        or summary["failed"]
    ):
        raise RuntimeError("Canonical summary failed post-merge validation")
    summary["combined_from"] = [str(path) for path in shards]
    summary["source_shards"] = [
        {
            "shard_index": item["shard_index"],
            "total_instructions": item["total_instructions"],
            "scored_attempts": item["scored_attempts"],
        }
        for item in summaries
    ]
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Merged {len(tasks)} tasks and {len(expected)} attempts into {output}; "
        f"hard={summary['hard_restriction']:.4f}, "
        f"soft={summary['soft_restriction']:.4f}, "
        f"cells={summary['cell_accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
