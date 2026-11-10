#!/usr/bin/env python3
"""Download JobBench and build the slime train/eval JSONL splits.

Split policy (see ``src/officeqa_rl/jobbench/dataset.py`` for the reasoning):
the 63-task ``easy`` split trains, the 65-task ``main`` split — the one the
public leaderboard reports — evaluates. The two splits hold unrelated
scenarios; they overlap only in occupation labels, so no task content crosses
the boundary.

Rows follow the slime data-source format the OfficeQA and SpreadsheetBench
splits use::

    {"prompt": "...", "label": "...", "metadata": {"uid": ..., ...}}

``prompt`` is the task instruction text the agent sees. ``label`` is the uid,
not the answer: JobBench's answers live in ``RUBRICS.json`` beside each task
and must never reach the agent, so the rollout hook reloads them from
``metadata.data_root`` at grading time. Under Docker-outside-of-Docker that
directory must exist at the same absolute path on the host and in the training
container, exactly like the SpreadsheetBench release directory.

Outputs (gitignored, under --output-dir):
    jobbench_train.jsonl   63 easy-split rows
    jobbench_eval.jsonl    65 main-split rows
    jobbench_manifest.json checksums, counts, and provenance

The task-level split record is written to
``<data-root>/../split_manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from officeqa_rl.jobbench.dataset import (
    EVAL_SPLIT,
    SPLIT_DIRS,
    TRAIN_SPLIT,
    JobBenchTask,
    Split,
    load_jobbench_tasks,
)

HF_REPO_ID = "JobBench/job-bench"
_HTTP_HEADERS = {"User-Agent": "officeqa-rl/jobbench-builder"}
#: Task counts published with the release; a mismatch means a changed revision.
EXPECTED_COUNTS: dict[Split, int] = {"main": 65, "easy": 63}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/jobbench/hf"),
        help="Release directory holding dataset/ and dataset_easy/",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/slime"))
    # Which split trains and which evaluates. The defaults are the published
    # arrangement (easy trains, main evaluates); swapping them is a real
    # experiment, not a relabel -- see the note in the module docstring.
    parser.add_argument(
        "--train-split", choices=("main", "easy"), default=TRAIN_SPLIT,
        help=f"Split to train on (default {TRAIN_SPLIT})",
    )
    parser.add_argument(
        "--eval-split", choices=("main", "easy"), default=EVAL_SPLIT,
        help=f"Split to evaluate on (default {EVAL_SPLIT})",
    )
    parser.add_argument(
        "--output-prefix", default="jobbench",
        help=("Basename prefix for the written files. Use a distinct prefix "
              "when swapping splits so the default dataset is not overwritten."),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use an already downloaded release instead of fetching from the Hub",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face dataset revision to pin",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Continue when a split's task count differs from the published one",
    )
    return parser.parse_args(argv)


def _repo_files(revision: str) -> list[tuple[str, int]]:
    """List the release's files and sizes from the Hub."""
    from huggingface_hub import HfApi

    entries = HfApi().list_repo_tree(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=revision,
        recursive=True,
    )
    return [
        (entry.path, int(getattr(entry, "size", 0) or 0))
        for entry in entries
        if getattr(entry, "size", None) is not None
    ]


def _is_complete(path: Path, size: int) -> bool:
    return path.is_file() and (size <= 0 or path.stat().st_size == size)


def _download_one(url: str, destination: Path, *, attempts: int = 4) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers=_HTTP_HEADERS)
            with urllib.request.urlopen(request, timeout=60) as response:
                with partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle, 1 << 20)
            partial.replace(destination)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)


def download_release(data_root: Path, revision: str, *, max_workers: int = 8) -> dict[str, int]:
    """Fetch the ~620 MB release file by file, skipping what is already there.

    ``snapshot_download`` is deliberately not used: without an ``HF_TOKEN`` its
    workers stall indefinitely partway through this repo (observed at 380 of
    985 files, zero-byte ``.incomplete`` placeholders, no progress and no
    timeout), while plain resolve-URL reads of the same files return in under a
    second. Files already present at their expected size are skipped, so a
    re-run resumes.
    """
    from huggingface_hub import hf_hub_url

    data_root.mkdir(parents=True, exist_ok=True)
    listing = _repo_files(revision)
    pending = [
        (path, size) for path, size in listing if not _is_complete(data_root / path, size)
    ]
    if not pending:
        return {"files": len(listing), "downloaded": 0}

    print(
        f"downloading {len(pending)} of {len(listing)} files "
        f"({sum(size for _, size in pending) / 1e6:.0f} MB) from {HF_REPO_ID}",
        file=sys.stderr,
    )
    failures: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _download_one,
                hf_hub_url(
                    repo_id=HF_REPO_ID,
                    filename=path,
                    repo_type="dataset",
                    revision=revision,
                ),
                data_root / path,
            ): path
            for path, _ in pending
        }
        for done in as_completed(futures):
            try:
                done.result()
            except Exception as error:  # collect, so one bad file is reported
                failures.append((futures[done], error))
    if failures:
        detail = "\n".join(f"  {path}: {error}" for path, error in failures[:10])
        raise SystemExit(f"{len(failures)} JobBench files failed to download:\n{detail}")
    return {"files": len(listing), "downloaded": len(pending)}


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slime_row(task: JobBenchTask, data_root: Path) -> dict[str, Any]:
    """One training/eval row. Rubrics stay on disk, reached via data_root."""
    return {
        "prompt": task.instructions,
        "label": task.uid,
        "metadata": {
            "uid": task.uid,
            "benchmark": "jobbench",
            "split": task.split,
            "occupation": task.occupation,
            "task_id": task.task_id,
            "data_root": str(data_root.resolve()),
            "n_rubrics": len(task.rubrics),
            "max_score": task.max_score,
            "n_local_reference_files": len(task.local_reference_files),
            "n_search_files": len(task.search_files),
        },
    }


def task_record(task: JobBenchTask) -> dict[str, Any]:
    return {
        "uid": task.uid,
        "split": task.split,
        "occupation": task.occupation,
        "task_id": task.task_id,
        "n_rubrics": len(task.rubrics),
        "max_score": task.max_score,
        "rubric_weights": [rubric.weight for rubric in task.rubrics],
        "n_criteria": sum(len(rubric.criteria) for rubric in task.rubrics),
        "local_reference_files": list(task.local_reference_files),
        "search_files": list(task.search_files),
        "instructions_sha256": sha256_of_text(task.instructions),
        "rubrics_sha256": sha256_of_file(task.rubrics_path),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"path": str(path), "rows": len(rows), "sha256": sha256_of_file(path)}


def summarize(tasks: list[JobBenchTask]) -> dict[str, Any]:
    rubric_counts = [len(task.rubrics) for task in tasks]
    return {
        "tasks": len(tasks),
        "occupations": len({task.occupation for task in tasks}),
        "rubric_items": sum(rubric_counts),
        "rubric_items_per_task_mean": (
            round(sum(rubric_counts) / len(rubric_counts), 2) if rubric_counts else 0.0
        ),
        "rubric_items_per_task_max": max(rubric_counts, default=0),
        "tasks_with_search_files": sum(1 for task in tasks if task.search_files),
        "reference_files_per_task_mean": (
            round(
                sum(len(task.local_reference_files) for task in tasks) / len(tasks), 2
            )
            if tasks
            else 0.0
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    data_root: Path = args.data_root
    if not args.no_download:
        download_release(data_root, args.revision)

    train_split = args.train_split
    eval_split = args.eval_split
    if train_split == eval_split:
        raise SystemExit(
            f"--train-split and --eval-split are both {train_split!r}; the two "
            "splits must differ or the eval measures the training set"
        )
    splits: dict[str, list[JobBenchTask]] = {}
    for split in (train_split, eval_split):
        tasks = load_jobbench_tasks(data_root, split)
        expected = EXPECTED_COUNTS[split]
        if len(tasks) != expected:
            message = (
                f"JobBench split {split!r} has {len(tasks)} tasks, expected {expected}. "
                "The release revision may have changed."
            )
            if not args.allow_count_mismatch:
                raise SystemExit(message + " Re-run with --allow-count-mismatch to continue.")
            print(f"WARNING: {message}", file=sys.stderr)
        splits[split] = tasks

    train_tasks = splits[train_split]
    eval_tasks = splits[eval_split]

    empty = [task.uid for task in train_tasks + eval_tasks if not task.rubrics]
    if empty:
        raise SystemExit(f"Tasks without rubrics cannot be graded: {empty}")

    train_file = write_jsonl(
        args.output_dir / f"{args.output_prefix}_train.jsonl",
        [slime_row(task, data_root) for task in train_tasks],
    )
    eval_file = write_jsonl(
        args.output_dir / f"{args.output_prefix}_eval.jsonl",
        [slime_row(task, data_root) for task in eval_tasks],
    )

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "repo_id": HF_REPO_ID,
            "revision": args.revision,
            "data_root": str(data_root.resolve()),
            "split_dirs": SPLIT_DIRS,
        },
        "policy": {
            "train_split": train_split,
            "eval_split": eval_split,
            "rationale": (
                "easy trains because every referenced file is inside task_folder/, "
                "so an episode is solvable without network access; main evaluates "
                "because it is the split the public leaderboard reports."
                if (train_split, eval_split) == ("easy", "main")
                else (
                    f"non-default arrangement: {train_split} trains, {eval_split} "
                    "evaluates. Training on main needs search_file_policy=stage -- "
                    "57 of its 65 tasks reference files_required_to_search/ that a "
                    "network-less sandbox cannot fetch, so under 'omit' those rubric "
                    "items are unreachable and the episode is optimised against a "
                    "reward it cannot earn."
                )
            ),
        },
        "train": {**train_file, **summarize(train_tasks)},
        "eval": {**eval_file, **summarize(eval_tasks)},
        "occupation_overlap": sorted(
            {task.occupation for task in train_tasks}
            & {task.occupation for task in eval_tasks}
        ),
    }
    write_json(args.output_dir / f"{args.output_prefix}_manifest.json", manifest)

    split_manifest = {
        "built_at": manifest["built_at"],
        "source": manifest["source"],
        "policy": manifest["policy"],
        "train": [task_record(task) for task in train_tasks],
        "eval": [task_record(task) for task in eval_tasks],
    }
    write_json(data_root.parent / "split_manifest.json", split_manifest)
    return manifest


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build(args)
    for name in ("train", "eval"):
        section = manifest[name]
        print(
            f"{name:5s} {section['tasks']:3d} tasks  "
            f"{section['occupations']:2d} occupations  "
            f"{section['rubric_items']:4d} rubric items "
            f"(mean {section['rubric_items_per_task_mean']}/task)  -> {section['path']}"
        )
    overlap = manifest["occupation_overlap"]
    print(f"occupations in both splits: {len(overlap)} (task content does not overlap)")


if __name__ == "__main__":
    main()
