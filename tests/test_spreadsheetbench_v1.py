"""SpreadsheetBench v1 loader, prompt, and official-compatible scoring."""

import asyncio
import json
import os
import subprocess
from pathlib import Path

import openpyxl
import pytest

from officeqa_rl.spreadsheetbench_v1 import (
    SpreadsheetBenchV1Case,
    SpreadsheetBenchV1Environment,
    build_spreadsheet_v1_system_prompt,
    load_spreadsheetbench_v1_tasks,
)
from officeqa_rl.spreadsheetbench_v1_scoring import (
    aggregate_v1_metrics,
    compare_v1_output,
    replay_solution,
)
from officeqa_rl.types import EnvironmentConfig


ROOT = Path(__file__).resolve().parents[1]


def _write_workbook(
    path: Path,
    values: dict[str, object],
    *,
    sheet_name: str = "Sheet1",
) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = sheet_name
    for cell, value in values.items():
        sheet[cell] = value
    workbook.save(path)


def _make_data_root(tmp_path: Path, case_count: int = 3) -> Path:
    data_root = tmp_path / "verified_400"
    task_dir = data_root / "spreadsheet" / "demo_7"
    task_dir.mkdir(parents=True)
    for index in range(1, case_count + 1):
        _write_workbook(
            task_dir / f"{index}_demo_7_input.xlsx", {"A1": index, "B1": None}
        )
        _write_workbook(
            task_dir / f"{index}_demo_7_answer.xlsx", {"A1": index, "B1": index * 2}
        )
    (data_root / "dataset.json").write_text(
        json.dumps(
            [
                {
                    "id": "demo_7",
                    "instruction": "Double A1 into B1.",
                    "spreadsheet_path": "spreadsheet/demo_7",
                    "instruction_type": "Cell-Level Manipulation",
                    "answer_position": "'Sheet1'!B1",
                }
            ]
        ),
        encoding="utf-8",
    )
    return data_root


def test_load_v1_task_groups_all_workbook_cases(tmp_path):
    task = load_spreadsheetbench_v1_tasks(_make_data_root(tmp_path))[0]

    assert task.uid == "SSB1__demo_7"
    assert task.task_id == "demo_7"
    assert task.instruction_type == "Cell-Level Manipulation"
    assert [case.index for case in task.cases] == [1, 2, 3]
    assert task.input_name == "1_demo_7_input.xlsx"
    assert task.output_name == "1_demo_7_output.xlsx"
    assert all(case.golden_path.is_file() for case in task.cases)


def test_load_v1_task_accepts_default_spreadsheet_layout(tmp_path):
    data_root = _make_data_root(tmp_path)
    rows = json.loads((data_root / "dataset.json").read_text())
    rows[0].pop("spreadsheet_path")
    (data_root / "dataset.json").write_text(json.dumps(rows))

    task = load_spreadsheetbench_v1_tasks(data_root)[0]

    assert len(task.cases) == 3


def test_load_v1_task_accepts_spreadsheet_path_to_first_file(tmp_path):
    data_root = _make_data_root(tmp_path)
    rows = json.loads((data_root / "dataset.json").read_text())
    rows[0]["spreadsheet_path"] = "spreadsheet/demo_7/1_demo_7_input.xlsx"
    (data_root / "dataset.json").write_text(json.dumps(rows))

    task = load_spreadsheetbench_v1_tasks(data_root)[0]

    assert [case.index for case in task.cases] == [1, 2, 3]


def test_load_verified_layout_and_qualifies_answer_sheet(tmp_path):
    data_root = tmp_path / "verified"
    task_dir = data_root / "spreadsheet" / "13-1"
    task_dir.mkdir(parents=True)
    _write_workbook(
        task_dir / "1_13-1_init.xlsx", {"A1": "input"}, sheet_name="LISTS"
    )
    _write_workbook(
        task_dir / "1_13-1_golden.xlsx", {"A1": "answer"}, sheet_name="LISTS"
    )
    (data_root / "dataset.json").write_text(
        json.dumps(
            [
                {
                    "id": "13-1",
                    "instruction": "Populate LISTS.",
                    "spreadsheet_path": "spreadsheet/13-1",
                    "instruction_type": "Sheet-Level Manipulation",
                    "answer_sheet": "LISTS",
                    "answer_position": "A1:A2,C1",
                }
            ]
        ),
        encoding="utf-8",
    )

    task = load_spreadsheetbench_v1_tasks(data_root)[0]

    assert len(task.cases) == 1
    assert task.input_name == "1_13-1_init.xlsx"
    assert task.output_name == "1_13-1_output.xlsx"
    assert task.answer_position == "'LISTS'!A1:A2,'LISTS'!C1"


def test_load_verified_layout_accepts_unnumbered_exception(tmp_path):
    data_root = tmp_path / "verified"
    task_dir = data_root / "spreadsheet" / "13284"
    task_dir.mkdir(parents=True)
    _write_workbook(task_dir / "initial.xlsx", {"A1": "input"})
    _write_workbook(task_dir / "golden.xlsx", {"A1": "answer"})
    (data_root / "dataset.json").write_text(
        json.dumps(
            [
                {
                    "id": "13284",
                    "instruction": "Populate the result.",
                    "spreadsheet_path": "spreadsheet/13284",
                    "instruction_type": "Cell-Level Manipulation",
                    "answer_position": "A1",
                }
            ]
        ),
        encoding="utf-8",
    )

    task = load_spreadsheetbench_v1_tasks(data_root)[0]

    assert task.input_name == "initial.xlsx"
    assert task.output_name == "output.xlsx"
    assert task.cases[0].golden_path.name == "golden.xlsx"


def test_load_verified_layout_accepts_lone_misnamed_golden(tmp_path):
    data_root = tmp_path / "verified"
    task_dir = data_root / "spreadsheet" / "42930"
    task_dir.mkdir(parents=True)
    _write_workbook(task_dir / "1_42930_init.xlsx", {"A1": "input"})
    _write_workbook(task_dir / "1_43930_golden.xlsx", {"A1": "answer"})
    (data_root / "dataset.json").write_text(
        json.dumps(
            [
                {
                    "id": "42930",
                    "instruction": "Assign unique numbers.",
                    "spreadsheet_path": "spreadsheet/42930",
                    "instruction_type": "Cell-Level Manipulation",
                    "answer_position": "A1",
                }
            ]
        ),
        encoding="utf-8",
    )

    task = load_spreadsheetbench_v1_tasks(data_root)[0]

    assert task.cases[0].golden_path.name == "1_43930_golden.xlsx"


def test_load_v1_task_rejects_missing_answer_case(tmp_path):
    data_root = _make_data_root(tmp_path)
    (data_root / "spreadsheet" / "demo_7" / "2_demo_7_answer.xlsx").unlink()

    with pytest.raises(FileNotFoundError, match="Missing answer workbook"):
        load_spreadsheetbench_v1_tasks(data_root)


def test_compare_v1_output_uses_official_rounding_and_blank_rules(tmp_path):
    golden = tmp_path / "golden.xlsx"
    output = tmp_path / "output.xlsx"
    _write_workbook(golden, {"A1": 1.234, "A2": None, "A3": "fixed"})
    _write_workbook(output, {"A1": "1.23", "A2": "", "A3": "fixed"})

    metrics = compare_v1_output(golden, output, "A1:A3")

    assert metrics["accuracy"] == 1.0
    assert metrics["cell_accuracy"] == 1.0
    assert metrics["cells"] == {"correct": 3, "total": 3}


def test_compare_v1_output_reports_dense_diagnostics(tmp_path):
    golden = tmp_path / "golden.xlsx"
    output = tmp_path / "output.xlsx"
    _write_workbook(golden, {"A1": 1, "A2": 2, "B1": 3})
    _write_workbook(output, {"A1": 1, "A2": 9, "B1": 3})

    metrics = compare_v1_output(golden, output, "'Sheet1'!A1:A2, B1")

    assert metrics["accuracy"] == 0.0
    assert metrics["cell_accuracy"] == pytest.approx(2 / 3)
    assert metrics["cells"] == {"correct": 2, "total": 3}
    assert metrics["cell_error_count"] == 1
    assert "Sheet1!A2" in metrics["cell_errors"][0]


def test_compare_v1_output_counts_missing_output_cells(tmp_path):
    golden = tmp_path / "golden.xlsx"
    _write_workbook(golden, {"A1": 1, "A2": 2})

    metrics = compare_v1_output(golden, tmp_path / "missing.xlsx", "A1:A2")

    assert metrics["accuracy"] == 0.0
    assert metrics["cells"] == {"correct": 0, "total": 2}
    assert metrics["error"] == "output file not exist"


def test_aggregate_v1_metrics_matches_hard_and_soft_restrictions():
    metrics = aggregate_v1_metrics(
        [
            {"accuracy": 1.0, "cells": {"correct": 2, "total": 2}},
            {"accuracy": 0.0, "cells": {"correct": 1, "total": 2}},
            {"accuracy": 1.0, "cells": {"correct": 2, "total": 2}},
        ]
    )

    assert metrics["test_case_results"] == [1, 0, 1]
    assert metrics["hard_restriction"] == 0.0
    assert metrics["soft_restriction"] == pytest.approx(2 / 3)
    assert metrics["cell_accuracy"] == pytest.approx(5 / 6)


def test_replay_solution_decodes_timeout_bytes(tmp_path, monkeypatch):
    solution = tmp_path / "solution.py"
    input_path = tmp_path / "input.xlsx"
    golden_path = tmp_path / "golden.xlsx"
    output_path = tmp_path / "output.xlsx"
    solution.write_text("pass\n")
    input_path.write_bytes(b"input")
    golden_path.write_bytes(b"golden")
    case = SpreadsheetBenchV1Case(1, input_path, golden_path)

    class TimedOutProcess:
        returncode = None

        def __init__(self):
            self.killed = False

        async def communicate(self):
            if not self.killed:
                await asyncio.sleep(60)
            return b"partial \xff", b"late"

        def kill(self):
            self.killed = True

    async def start_process(*args, **kwargs):
        return TimedOutProcess()

    async def remove_container(_name):
        return None

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start_process)
    monkeypatch.setattr(
        "officeqa_rl.spreadsheetbench_v1_scoring._remove_container",
        remove_container,
    )
    replay = asyncio.run(
        replay_solution(
            solution,
            case,
            output_path,
            visible_input_name="input.xlsx",
            visible_output_name="output.xlsx",
            timeout_sec=0.01,
        )
    )

    assert replay["timed_out"] is True
    assert replay["stdout"] == "partial \ufffd"
    assert replay["stderr"] == "late"
    json.dumps(replay)


def test_v1_prompt_requires_reusable_environment_variable_program():
    prompt = build_spreadsheet_v1_system_prompt("bash")

    assert "exactly one fenced bash block" in prompt
    assert "/workspace/solution.py" in prompt
    assert 'os.environ["SSB_INPUT"]' in prompt
    assert "additional workbook variants" in prompt
    assert "Instruction type and Answer position" in prompt


def test_v1_task_message_exposes_official_prompt_fields(tmp_path):
    data_root = _make_data_root(tmp_path)
    task = load_spreadsheetbench_v1_tasks(data_root)[0]
    config = EnvironmentConfig(
        corpus_dir=data_root,
        pdf_dir=None,
        workspace_root=tmp_path / "ws",
        sandbox_image="ssb1:test",
        command_timeout_sec=30,
        max_observation_chars=1000,
        sandbox_runtime="docker",
    )
    environment = SpreadsheetBenchV1Environment(config, task, run_id="run-1")
    environment._active = True

    message = environment.get_task_message()

    assert "Instruction type: Cell-Level Manipulation" in message["content"]
    assert "Answer position: 'Sheet1'!B1" in message["content"]
    assert message["extra"]["task"]["instruction_type"] == task.instruction_type
    assert message["extra"]["task"]["answer_position"] == task.answer_position


def test_v1_environment_requires_docker_runtime(tmp_path):
    data_root = _make_data_root(tmp_path)
    task = load_spreadsheetbench_v1_tasks(data_root)[0]
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
        SpreadsheetBenchV1Environment(config, task, run_id="run-1")


def test_v1_interactive_sandbox_has_memory_and_process_limits(tmp_path, monkeypatch):
    data_root = _make_data_root(tmp_path)
    task = load_spreadsheetbench_v1_tasks(data_root)[0]
    config = EnvironmentConfig(
        corpus_dir=data_root,
        pdf_dir=None,
        workspace_root=tmp_path / "ws",
        sandbox_image="ssb1:test",
        command_timeout_sec=30,
        max_observation_chars=1000,
        sandbox_runtime="docker",
    )
    environment = SpreadsheetBenchV1Environment(config, task, run_id="limited")
    environment._prepare_workspace()
    commands = []

    async def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return 0, "test-container", False

    monkeypatch.setattr(environment, "_run_process", fake_run)
    asyncio.run(environment._start_command_container())
    command = commands[0]
    for option, value in (("--memory", "8g"), ("--memory-swap", "8g"),
                          ("--pids-limit", "256"), ("--network", "none")):
        assert command[command.index(option) + 1] == value
    assert command.count("-v") == 1
    assert command[command.index("-v") + 1] == f"{environment.workspace}:/workspace"
    manifest = json.loads((environment.workspace / "manifest.json").read_text())
    assert manifest["container_memory_limit"] == "8g"
    assert environment.get_info()["container_pids_limit"] == 256


@pytest.mark.parametrize("error", [
    "EngineDeadError: engine unavailable",
    RuntimeError("EngineCore encountered an issue. See stack trace above."),
    "Engine core process died",
])
def test_v1_evaluator_fails_fast_if_local_engine_dies(error, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from run_spreadsheetbench_v1_eval import _raise_if_engine_failed

    with pytest.raises(RuntimeError, match="Saved attempts are retained"):
        _raise_if_engine_failed("vllm", error)
    _raise_if_engine_failed("openai-server", error)
    _raise_if_engine_failed("vllm", "Command timed out")


def test_v1_launcher_wires_data_backend_uid_and_results(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "dataset.json").write_text("[]", encoding="utf-8")
    args_log = tmp_path / "args.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/bin/bash\nprintf '%s\\n' \"$*\" > \"$EVAL_ARGS_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_docker = fake_bin / "docker"
    fake_docker.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_docker.chmod(0o755)
    results_dir = tmp_path / "results"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "EVAL_ARGS_LOG": str(args_log),
        "PYTHON_BIN": str(fake_python),
        "SSB1_DATA_ROOT": str(data_root),
        "SSB1_BACKEND": "openai-server",
        "SSB1_MODEL": "test-model",
        "SSB1_OPENAI_BASE_URL": "http://127.0.0.1:9000/v1",
        "SSB1_SANDBOX_IMAGE": "ssb1:test",
        "SSB1_OUTPUT_DIR": str(results_dir),
        "SSB1_UID": "SSB1__demo_7",
        "SSB1_CONCURRENCY": "2",
    }

    subprocess.run(
        ["bash", str(ROOT / "scripts" / "eval_spreadsheetbench_v1.sh")],
        check=True,
        env=env,
    )

    args = args_log.read_text(encoding="utf-8")
    assert "--generation-backend openai-server" in args
    assert "--model test-model" in args
    assert f"--data-root {data_root}" in args
    assert "--sandbox-image ssb1:test" in args
    assert "--uid SSB1__demo_7" in args
    assert "--temperature 1.0" in args
    assert "--samples-per-task 1" in args
    assert "--concurrency 2" in args
    assert "--openai-base-url http://127.0.0.1:9000/v1" in args
    assert list(results_dir.glob("eval-*.log"))


def test_v1_vllm_launcher_uses_standard_evaluation_recipe(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "dataset.json").write_text("[]", encoding="utf-8")
    args_log = tmp_path / "args.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/bin/bash\nprintf '%s\\n' \"$*\" > \"$EVAL_ARGS_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_docker = fake_bin / "docker"
    fake_docker.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_docker.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "EVAL_ARGS_LOG": str(args_log),
        "PYTHON_BIN": str(fake_python),
        "SSB1_DATA_ROOT": str(data_root),
        "SSB1_BACKEND": "vllm",
        "SSB1_MODEL": "Qwen3.5-4B",
        "SSB1_SANDBOX_IMAGE": "ssb1:test",
        "SSB1_OUTPUT_DIR": str(tmp_path / "results"),
    }

    subprocess.run(
        ["bash", str(ROOT / "scripts" / "eval_spreadsheetbench_v1.sh")],
        check=True,
        env=env,
    )

    args = args_log.read_text(encoding="utf-8")
    assert "--temperature 1.0" in args
    assert "--samples-per-task 8" in args
    assert "--concurrency 48" in args
    assert "--data-parallel-size 8" in args
    assert "--tensor-parallel-size 1" in args
    assert "--pipeline-parallel-size 1" in args
    assert "--max-num-seqs 16" in args
    assert "--enable-prefix-caching" in args
