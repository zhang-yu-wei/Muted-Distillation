"""JobBench task loading and the train/eval split policy.

JobBench ships two splits of the same 35 occupations. They are *not* two
difficulty settings of the same scenarios: corresponding task slots hold
unrelated work (``biostatisticians/task1`` is a Framingham sample-size
validation on ``main`` and a leprosy trial design memo on ``easy``). ``easy``
means shorter prompts and no externally-sourced reference files.

That difference decides the split policy here:

``easy`` (63 tasks) is the **train** split
    Every file a task references is inside its ``task_folder/``, so an episode
    is solvable in a network-less sandbox.

``main`` (65 tasks) is the **eval** split
    This is the split the public leaderboard reports. Its instructions also
    name documents the agent is expected to obtain from the web; those live in
    ``files_required_to_search/`` and the official runner does not stage them
    (see ``SearchFilePolicy``).

On-disk layout of a downloaded release::

    <data_root>/
      dataset/<occupation>/task<N>/       # the main split
        RUBRICS.json                      # grading only, never staged
        task_card.md                      # human brief, never staged
        task_folder/TASK_INSTRUCTIONS.txt # the agent's prompt
        task_folder/<reference files>
        files_required_to_search/<...>    # main split only
      dataset_easy/<occupation>/task<N>/  # the easy split
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Literal

Split = Literal["main", "easy"]

#: Subdirectory of a release holding each split.
SPLIT_DIRS: dict[Split, str] = {"main": "dataset", "easy": "dataset_easy"}

#: Split used for RL training and the one held out for evaluation.
TRAIN_SPLIT: Split = "easy"
EVAL_SPLIT: Split = "main"

INSTRUCTIONS_NAME = "TASK_INSTRUCTIONS.txt"
RUBRICS_NAME = "RUBRICS.json"
TASK_CARD_NAME = "task_card.md"
TASK_FOLDER_NAME = "task_folder"
SEARCH_FILES_NAME = "files_required_to_search"

_TASK_DIR_RE = re.compile(r"^task(\d+)$")


class SearchFilePolicy(str, Enum):
    """What to do with a main-split task's ``files_required_to_search/``.

    ``omit``
        Faithful to the official runner, which stages only ``task_folder/``.
        Rubric items that depend on an externally-sourced document are then
        unreachable for an agent without web access, which costs score on the
        affected tasks but keeps numbers comparable with the leaderboard.
    ``stage``
        Copy the files into the workspace next to the local references. Makes
        every rubric item reachable offline, and makes the result **not**
        comparable with published JobBench scores. Record the choice in any
        result you report.
    """

    OMIT = "omit"
    STAGE = "stage"


@dataclass(frozen=True)
class JobBenchRubric:
    """One weighted pass/fail rubric item and its sub-criteria."""

    index: int
    rubric: str
    weight: int
    criteria: tuple[str, ...]

    def to_judge_dict(self) -> dict[str, Any]:
        """Return the upstream ``RUBRICS.json`` item shape the judge expects."""
        return {
            "rubric": self.rubric,
            "weight": self.weight,
            "criterion": list(self.criteria),
        }


@dataclass(frozen=True)
class JobBenchTask:
    """One JobBench task, with grading material kept out of the workspace."""

    uid: str
    split: Split
    occupation: str
    task_id: str
    task_dir: Path
    instructions: str
    rubrics: tuple[JobBenchRubric, ...]
    local_reference_files: tuple[str, ...]
    search_files: tuple[str, ...]

    @property
    def task_folder(self) -> Path:
        return self.task_dir / TASK_FOLDER_NAME

    @property
    def search_files_dir(self) -> Path:
        return self.task_dir / SEARCH_FILES_NAME

    @property
    def rubrics_path(self) -> Path:
        return self.task_dir / RUBRICS_NAME

    @property
    def max_score(self) -> int:
        return sum(rubric.weight for rubric in self.rubrics)

    def rubric_dicts(self) -> list[dict[str, Any]]:
        return [rubric.to_judge_dict() for rubric in self.rubrics]


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "unknown"


def make_uid(split: Split, occupation: str, task_id: str) -> str:
    """Return the stable task identifier used by results, splits, and dumps."""
    return f"JB__{split}__{_safe_component(occupation)}__{_safe_component(task_id)}"


def _parse_rubrics(path: Path) -> tuple[JobBenchRubric, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload["rubrics"] if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError(f"Expected a list of rubrics in {path}")

    rubrics: list[JobBenchRubric] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Rubric {index} in {path} is not an object")
        criterion = item.get("criterion", [])
        if isinstance(criterion, str):
            criteria: tuple[str, ...] = (criterion,)
        else:
            criteria = tuple(str(entry) for entry in criterion)
        rubrics.append(
            JobBenchRubric(
                index=index,
                rubric=str(item.get("rubric", "")),
                weight=int(item.get("weight", 0)),
                criteria=criteria,
            )
        )
    return tuple(rubrics)


def _listed_files(directory: Path) -> tuple[str, ...]:
    if not directory.is_dir():
        return ()
    return tuple(
        sorted(
            entry.relative_to(directory).as_posix()
            for entry in directory.rglob("*")
            if entry.is_file()
        )
    )


def load_task(task_dir: Path, split: Split) -> JobBenchTask:
    """Load one ``<occupation>/task<N>`` directory."""
    task_folder = task_dir / TASK_FOLDER_NAME
    instructions_path = task_folder / INSTRUCTIONS_NAME
    if not instructions_path.is_file():
        raise FileNotFoundError(f"Missing {INSTRUCTIONS_NAME} in {task_folder}")
    rubrics_path = task_dir / RUBRICS_NAME
    if not rubrics_path.is_file():
        raise FileNotFoundError(f"Missing {RUBRICS_NAME} in {task_dir}")

    occupation = task_dir.parent.name
    task_id = task_dir.name
    reference_files = tuple(
        name for name in _listed_files(task_folder) if name != INSTRUCTIONS_NAME
    )
    return JobBenchTask(
        uid=make_uid(split, occupation, task_id),
        split=split,
        occupation=occupation,
        task_id=task_id,
        task_dir=task_dir,
        instructions=instructions_path.read_text(encoding="utf-8").strip(),
        rubrics=_parse_rubrics(rubrics_path),
        local_reference_files=reference_files,
        search_files=_listed_files(task_dir / SEARCH_FILES_NAME),
    )


def iter_task_dirs(data_root: Path, split: Split) -> Iterable[Path]:
    """Yield ``<occupation>/task<N>`` directories of one split, sorted."""
    split_root = data_root / SPLIT_DIRS[split]
    if not split_root.is_dir():
        raise FileNotFoundError(
            f"Missing JobBench split directory {split_root}. Download the "
            "release first: python3 scripts/build_jobbench_dataset.py"
        )
    for occupation_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        task_dirs = [p for p in occupation_dir.iterdir() if p.is_dir()]
        for task_dir in sorted(
            task_dirs, key=lambda p: (_task_number(p.name), p.name)
        ):
            if (task_dir / TASK_FOLDER_NAME).is_dir():
                yield task_dir


def _task_number(name: str) -> int:
    match = _TASK_DIR_RE.match(name)
    return int(match.group(1)) if match else 1 << 30


def load_jobbench_tasks(
    data_root: Path, split: Split = EVAL_SPLIT
) -> list[JobBenchTask]:
    """Load every task of one split in a deterministic order."""
    tasks = [load_task(task_dir, split) for task_dir in iter_task_dirs(data_root, split)]
    uids = {task.uid for task in tasks}
    if len(uids) != len(tasks):
        raise ValueError(f"Duplicate JobBench uids in {data_root}/{SPLIT_DIRS[split]}")
    return tasks


def load_task_by_uid(data_root: Path, uid: str) -> JobBenchTask:
    """Reload a single task by uid, as the rollout hook does per episode."""
    try:
        _, split, occupation, task_id = uid.split("__", 3)
    except ValueError as error:
        raise ValueError(f"Malformed JobBench uid: {uid!r}") from error
    if split not in SPLIT_DIRS:
        raise ValueError(f"Unknown JobBench split in uid {uid!r}: {split!r}")
    task_dir = data_root / SPLIT_DIRS[split] / occupation / task_id  # type: ignore[index]
    if not task_dir.is_dir():
        raise FileNotFoundError(f"No JobBench task directory for uid {uid}: {task_dir}")
    return load_task(task_dir, split)  # type: ignore[arg-type]


__all__ = [
    "EVAL_SPLIT",
    "INSTRUCTIONS_NAME",
    "JobBenchRubric",
    "JobBenchTask",
    "RUBRICS_NAME",
    "SEARCH_FILES_NAME",
    "SPLIT_DIRS",
    "SearchFilePolicy",
    "Split",
    "TASK_CARD_NAME",
    "TASK_FOLDER_NAME",
    "TRAIN_SPLIT",
    "iter_task_dirs",
    "load_jobbench_tasks",
    "load_task",
    "load_task_by_uid",
    "make_uid",
]
