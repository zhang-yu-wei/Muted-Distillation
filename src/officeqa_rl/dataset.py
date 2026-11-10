"""OfficeQA dataset access and slime JSONL row conversion.

Single owner of the row -> task mapping. The inference CLI, the slime dataset
builder, and the training rollout hook all construct ``OfficeQATask`` through
these functions so the task representation cannot drift between entry points.

Slime rows keep the ground truth only under ``label``; ``prompt`` and
``metadata`` are the agent-visible surface and never carry the answer.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from officeqa_rl.types import OfficeQATask

CSV_REQUIRED_COLUMNS = ("uid", "question", "answer")


def _clean(value: str | None) -> str:
    # The benchmark CSV quotes multi-line fields with \r\n breaks. The original
    # OfficeQA runner read them through universal-newline mode, so \n is the
    # canonical form for prompt parity with the published harness.
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def task_from_csv_row(row: dict[str, str]) -> OfficeQATask:
    """Build a task from one native benchmark CSV row."""
    uid = _clean(row.get("uid"))
    question = _clean(row.get("question"))
    if not uid:
        raise ValueError("CSV row has no uid")
    if not question:
        raise ValueError(f"CSV row {uid} has no question")
    return OfficeQATask(
        uid=uid,
        question=question,
        answer=_clean(row.get("answer")) or None,
        source_files=_clean(row.get("source_files")),
        difficulty=_clean(row.get("difficulty")) or None,
    )


def task_from_slime_fields(
    prompt: str,
    label: str | None,
    metadata: dict[str, Any] | None,
) -> OfficeQATask:
    """Build a task from slime sample fields (``Sample.prompt``/``label``/``metadata``)."""
    metadata = metadata or {}
    uid = _clean(metadata.get("uid"))
    question = _clean(prompt)
    if not uid:
        raise ValueError("Slime sample metadata has no uid")
    if not question:
        raise ValueError(f"Slime sample {uid} has an empty prompt")
    return OfficeQATask(
        uid=uid,
        question=question,
        answer=_clean(label) or None,
        source_files=_clean(metadata.get("source_files")),
        difficulty=_clean(metadata.get("difficulty")) or None,
    )


def task_from_slime_row(row: dict[str, Any]) -> OfficeQATask:
    """Build a task from one slime JSONL row (``{"prompt", "label", "metadata"}``)."""
    return task_from_slime_fields(
        prompt=row.get("prompt", ""),
        label=row.get("label"),
        metadata=row.get("metadata"),
    )


def slime_row_from_csv_row(row: dict[str, str]) -> dict[str, Any]:
    """Convert one benchmark CSV row into the slime JSONL row format.

    The builder rejects rows without an answer: a row that cannot be graded
    cannot produce a training reward or an evaluation score.
    """
    task = task_from_csv_row(row)
    if task.answer is None:
        raise ValueError(f"CSV row {task.uid} has no answer; it cannot be graded")
    return {
        "prompt": task.question,
        "label": task.answer,
        "metadata": {
            "uid": task.uid,
            "source_files": task.source_files,
            "difficulty": task.difficulty,
        },
    }


def read_csv_rows(dataset_csv: Path) -> list[dict[str, str]]:
    """Read all benchmark rows, validating the header once."""
    with dataset_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        header = reader.fieldnames or []
        missing = [column for column in CSV_REQUIRED_COLUMNS if column not in header]
        if missing:
            raise ValueError(f"{dataset_csv} is missing columns: {', '.join(missing)}")
        return list(reader)


def load_task(dataset_csv: Path, uid: str) -> OfficeQATask:
    """Load exactly one task without exposing other benchmark rows."""
    matches = [row for row in read_csv_rows(dataset_csv) if _clean(row.get("uid")) == uid]
    if not matches:
        raise KeyError(f"OfficeQA UID not found: {uid}")
    if len(matches) > 1:
        raise ValueError(f"OfficeQA UID is not unique: {uid}")
    return task_from_csv_row(matches[0])


__all__ = [
    "CSV_REQUIRED_COLUMNS",
    "load_task",
    "read_csv_rows",
    "slime_row_from_csv_row",
    "task_from_csv_row",
    "task_from_slime_fields",
    "task_from_slime_row",
]
