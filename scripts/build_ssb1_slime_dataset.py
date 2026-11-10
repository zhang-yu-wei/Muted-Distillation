"""Download SpreadsheetBench v1 and build the slime train/eval JSONL splits.

Split policy: the 400-task Verified release is the untouched evaluation set
(all existing SSB1 baseline numbers were measured on it); training uses the
unverified remainder of the full 912-task release, minus tasks that can never
grade positive or that carry no instruction text (see EXCLUDED_TASKS). Both rows
follow the slime data-source format used by the OfficeQA splits::

    {"prompt": "...", "label": "...", "metadata": {"uid": ..., ...}}

``prompt`` is the raw instruction and ``label`` the merged answer position;
both are agent-visible for this benchmark. Grading needs the golden workbooks,
so ``metadata.data_root`` records the absolute release directory and the
rollout hook reloads the full task by ``metadata.task_id``. Under DooD that
directory must exist at the same absolute path on the Docker host, like the
OfficeQA corpus dir.

Outputs (gitignored, under --output-dir):
    ssb1_train.jsonl          502 train rows (full minus Verified minus excluded)
    ssb1_verified_eval.jsonl  400 Verified eval rows
    ssb1_manifest.json        checksums, counts, and provenance for both files

The id-level split record (train/eval id lists, exclusion reasons, archive
checksum) is rewritten deterministically to
``<data-dir>/train_split_manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from officeqa_rl.spreadsheetbench_v1 import (
    SpreadsheetBenchV1Task,
    load_spreadsheetbench_v1_tasks,
)

RELEASES = {
    "full": {
        "url": (
            "https://raw.githubusercontent.com/RUCKBReasoning/SpreadsheetBench/"
            "main/data/spreadsheetbench_912_v0.1.tar.gz"
        ),
        "sha256": "9cf7228b54f1edcdd4b372eb736774adf29cb4f804c9920229bac6c154833399",
        "archive_root": "all_data_912_v0.1",
        "tasks": 912,
    },
    "verified": {
        "url": (
            "https://raw.githubusercontent.com/RUCKBReasoning/SpreadsheetBench/"
            "main/data/spreadsheetbench_verified_400.tar.gz"
        ),
        "sha256": "10ef893dd29cb13ab97143ea787e68cdc9574a13873ab9a54e50b31dc03fc949",
        "archive_root": "spreadsheetbench_verified_400",
        "tasks": 400,
    },
}

# Directory names under --data-dir. "verified_400" matches the default
# SSB1_DATA_ROOT of scripts/eval_spreadsheetbench_v1.sh.
FULL_DIR = "all_data_912_v0.1"
VERIFIED_DIR = "verified_400"

# Train-split tasks dropped during the 2026-08-04/05 split audit.
#
# Ungradeable: the annotation references cells that cannot be read from the
# golden workbooks, so compare_v1_output() scores 0 for every possible output.
# Permanent zero-reward tasks are pure waste: GRPO advantages vanish and DAPO
# dynamic sampling would resample them forever. Found by sweeping every train
# golden workbook through the grading range parser.
#
# Empty instruction: dataset.json holds only a non-breaking space and the task
# directory has no other instruction source, so the agent would receive an
# empty task and its reward would be noise.
EXCLUDED_TASKS = {
    "532-3": "answer_position is doubly quoted ('Received'!'Received!A1:G16'); range never parses",
    "53-14": "answer_sheet 'Sheet1' absent from golden workbooks (only 'Interest')",
    "118-8": "answer_position sheet 'Invoice List‘' has a curly quote; golden sheet is 'Invoice List'",
    "147-7": "answer_sheet 'step1' vs golden sheet 'Step1'; sheet lookup is case-sensitive",
    "CF_25745": "answer_sheet 'Sheet1' absent from golden workbooks (only 'Ark1')",
    "52763": "answer_position names 'Sheet 2'; golden sheet is 'Sheet2'",
    "59654": "first sheet name '     DATA    ' has surrounding spaces; range parser strips to 'DATA'",
    "55224": "instruction is a single non-breaking space; no task text exists",
    "55457": "instruction is a single non-breaking space; no task text exists",
    "55877": "instruction is a single non-breaking space; no task text exists",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/spreadsheetbench_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/slime"))
    parser.add_argument("--prefix", default="ssb1")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail if a release directory is missing instead of downloading it",
    )
    return parser.parse_args(argv)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_release(data_dir: Path, name: str, target_dir: str, *, download: bool) -> Path:
    """Return the release directory, downloading and extracting it if absent."""
    release = RELEASES[name]
    destination = data_dir / target_dir
    if (destination / "dataset.json").is_file():
        return destination
    if not download:
        raise FileNotFoundError(
            f"{destination} is missing and --no-download was given; "
            f"fetch {release['url']} manually"
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / Path(release["url"]).name
    if not archive.is_file() or sha256_of_file(archive) != release["sha256"]:
        print(f"downloading {release['url']}")
        with tempfile.NamedTemporaryFile(dir=data_dir, delete=False) as handle:
            partial = Path(handle.name)
        try:
            urllib.request.urlretrieve(release["url"], partial)
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    digest = sha256_of_file(archive)
    if digest != release["sha256"]:
        raise ValueError(
            f"{archive} sha256 mismatch: expected {release['sha256']}, got {digest}"
        )

    with tarfile.open(archive) as tar:
        tar.extractall(data_dir, filter="data")
    extracted = data_dir / release["archive_root"]
    if extracted != destination:
        extracted.rename(destination)
    if not (destination / "dataset.json").is_file():
        raise FileNotFoundError(f"{archive} did not contain {target_dir}/dataset.json")
    return destination


def slime_row(task: SpreadsheetBenchV1Task, data_root: Path) -> dict[str, Any]:
    if not task.instruction or not task.answer_position:
        raise ValueError(
            f"task {task.task_id} has an empty instruction or answer_position; "
            "a row that cannot pose or grade the task must be excluded"
        )
    return {
        "prompt": task.instruction,
        "label": task.answer_position,
        "metadata": {
            "uid": task.uid,
            "benchmark": "ssb1",
            "task_id": task.task_id,
            "instruction_type": task.instruction_type,
            "answer_position": task.answer_position,
            "case_count": len(task.cases),
            "data_root": str(data_root),
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")
    return {"file": path.name, "rows": len(rows), "sha256": sha256_of_file(path)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    full_root = ensure_release(
        args.data_dir, "full", FULL_DIR, download=not args.no_download
    ).resolve()
    verified_root = ensure_release(
        args.data_dir, "verified", VERIFIED_DIR, download=not args.no_download
    ).resolve()

    # The repo loader owns uid construction, answer_sheet merging, and workbook
    # pairing, so rows here cannot drift from what rollout and eval will load.
    full_tasks = load_spreadsheetbench_v1_tasks(full_root)
    eval_tasks = load_spreadsheetbench_v1_tasks(verified_root)
    if len(full_tasks) != RELEASES["full"]["tasks"]:
        raise ValueError(f"expected 912 full-release tasks, got {len(full_tasks)}")
    if len(eval_tasks) != RELEASES["verified"]["tasks"]:
        raise ValueError(f"expected 400 Verified tasks, got {len(eval_tasks)}")

    full_ids = {task.task_id for task in full_tasks}
    eval_ids = {task.task_id for task in eval_tasks}
    if not eval_ids <= full_ids:
        raise ValueError(
            f"Verified ids missing from the full release: {sorted(eval_ids - full_ids)[:5]}"
        )
    if not set(EXCLUDED_TASKS) <= full_ids - eval_ids:
        raise ValueError("every excluded id must be a non-Verified full-release task")

    train_tasks = sorted(
        (
            task
            for task in full_tasks
            if task.task_id not in eval_ids and task.task_id not in EXCLUDED_TASKS
        ),
        key=lambda task: task.task_id,
    )
    train_uids = {task.uid for task in train_tasks}
    if train_uids & {task.uid for task in eval_tasks}:
        raise ValueError("train and eval uid sets overlap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": write_jsonl(
            args.output_dir / f"{args.prefix}_train.jsonl",
            [slime_row(task, full_root) for task in train_tasks],
        ),
        "verified_eval": write_jsonl(
            args.output_dir / f"{args.prefix}_verified_eval.jsonl",
            [slime_row(task, verified_root) for task in eval_tasks],
        ),
    }

    archive = args.data_dir / Path(RELEASES["full"]["url"]).name
    split_manifest = {
        "description": (
            "SpreadsheetBench v1 RL train split: full 912 release minus the 400 "
            "Verified ids (held out as the eval set) minus tasks whose annotations "
            "can never grade positive."
        ),
        "source": {
            "full_release": str(args.data_dir / FULL_DIR),
            "full_release_archive": archive.name,
            "full_release_sha256": RELEASES["full"]["sha256"],
            "download_url": RELEASES["full"]["url"],
            "eval_release": str(args.data_dir / VERIFIED_DIR),
        },
        "counts": {
            "full": len(full_tasks),
            "verified_eval": len(eval_tasks),
            "excluded_tasks": len(EXCLUDED_TASKS),
            "train": len(train_tasks),
            "train_workbook_cases": sum(len(task.cases) for task in train_tasks),
            "train_instruction_types": {
                kind: sum(1 for task in train_tasks if task.instruction_type == kind)
                for kind in sorted({task.instruction_type for task in train_tasks})
            },
        },
        "excluded_tasks": EXCLUDED_TASKS,
        "eval_ids": sorted(eval_ids),
        "train_ids": [task.task_id for task in train_tasks],
    }
    (args.data_dir / "train_split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "split_manifest": str((args.data_dir / "train_split_manifest.json").resolve()),
        "data_roots": {"train": str(full_root), "verified_eval": str(verified_root)},
        "counts": split_manifest["counts"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "outputs": outputs,
    }
    manifest_path = args.output_dir / f"{args.prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    manifest = build(parse_args(argv))
    for name, output in manifest["outputs"].items():
        print(f"{name}: {output['rows']} rows -> {output['file']}")
    print(f"manifest: {manifest['outputs']['train']['rows']} train tasks, split at {manifest['split_manifest']}")


if __name__ == "__main__":
    sys.exit(main())
