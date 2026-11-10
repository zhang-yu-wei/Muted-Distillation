"""SpreadsheetBench 2 loader, prompt, and grading wiring (no docker needed)."""

import json
from pathlib import Path

import openpyxl
import pytest

from officeqa_rl.spreadsheetbench import (
    SpreadsheetBenchEnvironment,
    build_spreadsheet_system_prompt,
    load_spreadsheetbench_tasks,
)
from officeqa_rl.spreadsheetbench_scoring import compare_output, comparison_flags
from officeqa_rl.types import EnvironmentConfig


def _write_workbook(path: Path, values: dict[str, object]) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    for cell, value in values.items():
        sheet[cell] = value
    workbook.save(path)


def _make_data_root(tmp_path: Path) -> Path:
    """One Template task whose golden differs from the input in A2 and B1."""
    data_root = tmp_path / "ssb"
    task_dir = data_root / "Template" / "spreadsheet" / "01_demo"
    task_dir.mkdir(parents=True)
    _write_workbook(task_dir / "01_01_input.xlsx", {"A1": 10, "A2": None, "B1": None})
    _write_workbook(task_dir / "01_01_golden.xlsx", {"A1": 10, "A2": 32, "B1": 7.5})
    (data_root / "Template" / "dataset.json").write_text(
        json.dumps(
            [
                {
                    "id": "01_01",
                    "instruction": "Fill in the schedule.",
                    "spreadsheet_path": "spreadsheet/01_demo/01_01_input.xlsx",
                    "golden_response_path": "spreadsheet/01_demo/01_01_golden.xlsx",
                    "answer_position": "'Sheet1'!A1:B2",
                }
            ]
        ),
        encoding="utf-8",
    )
    return data_root


def test_load_tasks_maps_fields_and_uids(tmp_path):
    tasks = load_spreadsheetbench_tasks(_make_data_root(tmp_path), ["Template"])

    assert len(tasks) == 1
    task = tasks[0]
    assert task.uid == "Template__01_01"
    assert task.category == "Template"
    assert task.instruction == "Fill in the schedule."
    assert task.input_path.is_file()
    assert task.golden_path.is_file()
    assert task.output_name == "01_01_output.xlsx"


def test_load_tasks_underscores_spaced_ids(tmp_path):
    data_root = _make_data_root(tmp_path)
    rows = json.loads((data_root / "Template" / "dataset.json").read_text())
    rows[0]["id"] = "Task 96"
    (data_root / "Template" / "dataset.json").write_text(json.dumps(rows))

    task = load_spreadsheetbench_tasks(data_root, ["Template"])[0]

    assert task.uid == "Template__Task_96"
    assert task.output_name == "Task_96_output.xlsx"
    assert task.task_id == "Task 96"


def test_load_tasks_rejects_missing_files(tmp_path):
    data_root = _make_data_root(tmp_path)
    (data_root / "Template" / "spreadsheet" / "01_demo" / "01_01_golden.xlsx").unlink()

    with pytest.raises(FileNotFoundError):
        load_spreadsheetbench_tasks(data_root, ["Template"])


def test_load_tasks_rejects_unknown_category(tmp_path):
    with pytest.raises(ValueError):
        load_spreadsheetbench_tasks(_make_data_root(tmp_path), ["NotACategory"])


def test_comparison_flags_follow_official_debugging_rules():
    assert comparison_flags("Debugging", "x/Inconsistent Color Coding_input.xlsx") == (
        True,
        False,
    )
    assert comparison_flags("Debugging", "x/Embedded Hardcode_input.xlsx") == (
        False,
        True,
    )
    assert comparison_flags("Debugging", "x/Errors_input.xlsx") == (False, False)
    # Only Debugging inputs carry the filename-based mode markers.
    assert comparison_flags("Template", "x/Embedded Hardcode_input.xlsx") == (
        False,
        False,
    )


def test_compare_output_scores_perfect_and_unmodified(tmp_path):
    data_root = _make_data_root(tmp_path)
    task = load_spreadsheetbench_tasks(data_root, ["Template"])[0]

    perfect = tmp_path / "perfect_output.xlsx"
    _write_workbook(perfect, {"A1": 10, "A2": 32, "B1": 7.5})
    result = compare_output(
        task.input_path, task.golden_path, perfect, task.answer_position
    )
    assert result["accuracy"] == 1.0
    assert result["modification_accuracy"] == 1.0
    assert result["regression_accuracy"] == 1.0
    assert result["cell_errors"] == []

    untouched = tmp_path / "untouched_output.xlsx"
    _write_workbook(untouched, {"A1": 10, "A2": None, "B1": None})
    result = compare_output(
        task.input_path, task.golden_path, untouched, task.answer_position
    )
    assert result["accuracy"] == 0.0
    assert result["modification_accuracy"] == 0.0
    assert result["regression_accuracy"] == 1.0
    assert result["cell_error_count"] == 2


def test_compare_output_tolerates_one_percent(tmp_path):
    data_root = _make_data_root(tmp_path)
    task = load_spreadsheetbench_tasks(data_root, ["Template"])[0]

    near = tmp_path / "near_output.xlsx"
    _write_workbook(near, {"A1": 10, "A2": 32.1, "B1": 7.5})

    result = compare_output(
        task.input_path, task.golden_path, near, task.answer_position
    )

    assert result["accuracy"] == 1.0  # 32.1 vs 32 is within the official 1%


def test_compare_output_missing_file_reports_error(tmp_path):
    data_root = _make_data_root(tmp_path)
    task = load_spreadsheetbench_tasks(data_root, ["Template"])[0]

    result = compare_output(
        task.input_path,
        task.golden_path,
        tmp_path / "does_not_exist.xlsx",
        task.answer_position,
    )

    assert result["accuracy"] == 0.0
    assert result["error"] == "output file not exist"


def test_system_prompt_keeps_single_fenced_block_protocol():
    prompt = build_spreadsheet_system_prompt("bash")

    assert "exactly one fenced bash block" in prompt
    assert "`submit`" in prompt
    assert "view_xlsx" in prompt


def test_environment_requires_docker_runtime(tmp_path):
    data_root = _make_data_root(tmp_path)
    task = load_spreadsheetbench_tasks(data_root, ["Template"])[0]
    config = EnvironmentConfig(
        corpus_dir=data_root,
        pdf_dir=None,
        workspace_root=tmp_path / "ws",
        sandbox_image=None,
        command_timeout_sec=30,
        max_observation_chars=1000,
        sandbox_runtime="local",
    )

    with pytest.raises(ValueError, match="docker"):
        SpreadsheetBenchEnvironment(config, task, run_id="run-1")
