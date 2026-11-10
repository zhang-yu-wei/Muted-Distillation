"""Official-compatible scoring and solution replay for SpreadsheetBench v1."""

from __future__ import annotations

import asyncio
import datetime
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Sequence

from openpyxl.utils import get_column_letter
from openpyxl.utils.cell import range_boundaries

from officeqa_rl.spreadsheetbench_v1 import SpreadsheetBenchV1Case


DEFAULT_SANDBOX_IMAGE = "spreadsheetbench-v1-sandbox"
DEFAULT_CONTAINER_MEMORY_LIMIT = "8g"


async def _remove_container(name: str) -> None:
    """Best-effort force removal, including timeout and cancellation paths."""
    try:
        process = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "-f",
            name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=60)
        except TimeoutError:
            process.kill()
            await process.wait()
    except Exception:
        # Replay diagnostics should describe the submitted solution, not turn
        # a best-effort cleanup failure into a second evaluation exception.
        pass


def _transform_value(value: Any) -> Any:
    """Match SpreadsheetBench v1 evaluation/evaluation.py exactly."""
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    if isinstance(value, datetime.time):
        return str(value)[:-3]
    if isinstance(value, datetime.datetime):
        excel_start_date = datetime.datetime(1899, 12, 30)
        delta = value - excel_start_date
        return round(delta.days + delta.seconds / 86400.0, 0)
    if isinstance(value, str):
        try:
            return round(float(value), 2)
        except ValueError:
            return value
    return value


def _values_equal(left: Any, right: Any) -> bool:
    left = _transform_value(left)
    right = _transform_value(right)
    if (left == "" and right is None) or (left is None and right == ""):
        return True
    if (left == "" and right == "") or (left is None and right is None):
        return True
    return type(left) is type(right) and left == right


def _answer_ranges(answer_position: str, first_sheet: str) -> list[tuple[str, str]]:
    ranges: list[tuple[str, str]] = []
    for raw_range in answer_position.split(","):
        raw_range = raw_range.strip()
        if not raw_range:
            continue
        if "!" in raw_range:
            sheet_name, cell_range = raw_range.split("!", 1)
        else:
            sheet_name, cell_range = first_sheet, raw_range
        ranges.append(
            (sheet_name.strip().strip("'"), cell_range.strip().strip("'"))
        )
    return ranges


def _cell_names(cell_range: str) -> list[str]:
    min_col, min_row, max_col, max_row = range_boundaries(cell_range)
    # The official evaluator iterates columns first.
    return [
        f"{get_column_letter(column)}{row}"
        for column in range(min_col, max_col + 1)
        for row in range(min_row, max_row + 1)
    ]


def compare_v1_output(
    golden_file: Path,
    output_file: Path,
    answer_position: str,
    *,
    max_cell_errors: int = 50,
) -> dict[str, Any]:
    """Compare designated cells with the official v1 value semantics.

    ``accuracy`` is the official per-workbook binary result. ``cell_accuracy``
    is an additional diagnostic and does not alter the official score.
    """
    import openpyxl

    golden_file = Path(golden_file)
    output_file = Path(output_file)
    empty = {
        "accuracy": 0.0,
        "cell_accuracy": 0.0,
        "cells": {"correct": 0, "total": 0},
        "cell_errors": [],
        "cell_error_count": 0,
        "error": "",
    }
    try:
        golden = openpyxl.load_workbook(golden_file, data_only=True)
    except Exception as error:
        return {**empty, "error": f"golden workbook load failed: {error}"}

    try:
        answer_ranges = _answer_ranges(answer_position, golden.sheetnames[0])
        expected_cells = sum(
            len(_cell_names(cell_range)) for _, cell_range in answer_ranges
        )
    except Exception as error:
        golden.close()
        return {**empty, "error": f"invalid answer_position: {error}"}

    if not output_file.is_file():
        golden.close()
        return {
            **empty,
            "cells": {"correct": 0, "total": expected_cells},
            "cell_error_count": expected_cells,
            "error": "output file not exist",
        }
    try:
        output = openpyxl.load_workbook(output_file, data_only=True)
    except Exception as error:
        golden.close()
        return {
            **empty,
            "cells": {"correct": 0, "total": expected_cells},
            "cell_error_count": expected_cells,
            "error": f"output workbook load failed: {error}",
        }

    correct = 0
    total = 0
    error_count = 0
    cell_errors: list[str] = []
    for sheet_name, cell_range in answer_ranges:
        cell_names = _cell_names(cell_range)
        total += len(cell_names)
        if sheet_name not in golden:
            error_count += len(cell_names)
            if len(cell_errors) < max_cell_errors:
                cell_errors.append(f"Golden worksheet not found: {sheet_name}")
            continue
        if sheet_name not in output:
            error_count += len(cell_names)
            if len(cell_errors) < max_cell_errors:
                cell_errors.append(f"Output worksheet not found: {sheet_name}")
            continue
        golden_sheet = golden[sheet_name]
        output_sheet = output[sheet_name]
        for cell_name in cell_names:
            expected = golden_sheet[cell_name].value
            actual = output_sheet[cell_name].value
            if _values_equal(expected, actual):
                correct += 1
            else:
                error_count += 1
                if len(cell_errors) < max_cell_errors:
                    cell_errors.append(
                        f"Value difference at {sheet_name}!{cell_name}: "
                        f"expected {expected!r}, got {actual!r}"
                    )

    golden.close()
    output.close()
    cell_accuracy = correct / total if total else 0.0
    return {
        "accuracy": 1.0 if total > 0 and correct == total else 0.0,
        "cell_accuracy": cell_accuracy,
        "cells": {"correct": correct, "total": total},
        "cell_errors": cell_errors,
        "cell_error_count": error_count,
        "error": "" if total else "answer_position selects no cells",
    }


def aggregate_v1_metrics(case_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute official soft/hard restriction scores across workbook cases."""
    results = [int(metrics.get("accuracy", 0.0) == 1.0) for metrics in case_metrics]
    correct_cells = sum(
        metrics.get("cells", {}).get("correct", 0) for metrics in case_metrics
    )
    total_cells = sum(
        metrics.get("cells", {}).get("total", 0) for metrics in case_metrics
    )
    soft = sum(results) / len(results) if results else 0.0
    hard = 1.0 if results and all(results) else 0.0
    return {
        "accuracy": hard,
        "hard_restriction": hard,
        "soft_restriction": soft,
        "cell_accuracy": correct_cells / total_cells if total_cells else 0.0,
        "cells": {"correct": correct_cells, "total": total_cells},
        "test_case_results": results,
        "case_metrics": list(case_metrics),
    }


def decode_subprocess_output(output: str | bytes | None) -> str:
    """Return subprocess output as JSON-safe text.

    Async subprocess pipes return bytes. Replay diagnostics are persisted in
    evaluation JSON, so normalize them before returning.
    """
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output or ""


async def replay_solution(
    solution_path: Path,
    case: SpreadsheetBenchV1Case,
    output_path: Path,
    *,
    visible_input_name: str,
    visible_output_name: str,
    sandbox_image: str = DEFAULT_SANDBOX_IMAGE,
    timeout_sec: float = 600.0,
    memory_limit: str = DEFAULT_CONTAINER_MEMORY_LIMIT,
) -> dict[str, Any]:
    """Run a generated solution on one case in a fresh, label-safe sandbox."""
    solution_path = solution_path.resolve()
    output_path = output_path.resolve()
    if not solution_path.is_file():
        return {
            "exit_code": None,
            "timed_out": False,
            "output_produced": False,
            "stdout": "",
            "stderr": "solution.py does not exist",
        }

    with tempfile.TemporaryDirectory(prefix="ssb1-replay-") as temporary:
        workspace = Path(temporary)
        shutil.copy2(solution_path, workspace / "solution.py")
        # Always expose the same aliases used for visible case 1. This supports
        # both the required environment-variable contract and official-style
        # programs that contain the visible filenames as literals.
        shutil.copy2(case.input_path, workspace / visible_input_name)
        container_name = f"ssb1-replay-{uuid.uuid4().hex[:12]}"
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--memory",
            memory_limit,
            "--memory-swap",
            memory_limit,
            "--pids-limit",
            "256",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            "HOME=/tmp",
            "-e",
            f"SSB_INPUT=/workspace/{visible_input_name}",
            "-e",
            f"SSB_OUTPUT=/workspace/{visible_output_name}",
            "-v",
            f"{workspace}:/workspace",
            "-w",
            "/workspace",
            sandbox_image,
            "python3",
            "/workspace/solution.py",
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                raw_stdout, raw_stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_sec
                )
                timed_out = False
                exit_code: int | None = process.returncode
            except TimeoutError:
                timed_out = True
                exit_code = None
                process.kill()
                # Killing the docker CLI does not stop the named container.
                # Remove it before draining the CLI pipes: a runaway child can
                # otherwise keep those pipes open forever and wedge the whole
                # evaluation after the nominal replay timeout.
                await _remove_container(container_name)
                try:
                    raw_stdout, raw_stderr = await asyncio.wait_for(
                        process.communicate(), timeout=60
                    )
                except TimeoutError:
                    raw_stdout = b""
                    raw_stderr = b"docker CLI did not exit after replay timeout"
            stdout = decode_subprocess_output(raw_stdout)
            stderr = decode_subprocess_output(raw_stderr)
        finally:
            # Killing the docker CLI process does not kill its container; the
            # named container must be removed explicitly or a runaway
            # solution.py can survive and consume the entire host.
            await _remove_container(container_name)

        produced = workspace / visible_output_name
        output_produced = produced.is_file()
        if output_produced:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.unlink(missing_ok=True)
            shutil.copy2(produced, output_path)

    return {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "output_produced": output_produced,
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
    }


__all__ = [
    "DEFAULT_SANDBOX_IMAGE",
    "aggregate_v1_metrics",
    "compare_v1_output",
    "decode_subprocess_output",
    "replay_solution",
]
