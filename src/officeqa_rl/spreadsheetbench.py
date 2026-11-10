"""SpreadsheetBench 2 tasks, prompts, and sandbox environment.

The benchmark (data under ``data/spreadsheetbench_v2/spreadsheetbench-v2``,
official code under ``third_party/SpreadsheetBench-2``) gives the agent an
input .xlsx workbook and a natural-language instruction; the deliverable is an
output workbook graded cell-by-cell against a golden workbook. This module
reuses the OfficeQA agent loop and sandbox plumbing; only the workspace
contents, the task message, and the ``submit`` semantics differ.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from officeqa_rl.full_corpus_env import FullCorpusEnvironment
from officeqa_rl.types import EnvironmentConfig, StepOutput, Submitted

ALL_CATEGORIES = ("Debugging", "Financial_Model", "Template", "Visualization")
# Visualization is excluded by default: its official grading needs chart-image
# export through the Excel/WPS COM interface (Windows) plus a VLM judge.
CELL_GRADED_CATEGORIES = ("Debugging", "Financial_Model", "Template")


@dataclass
class SpreadsheetBenchTask:
    """One benchmark task; golden_path is for grading only, never mounted."""

    uid: str
    category: str
    task_id: str
    instruction: str
    input_path: Path
    golden_path: Path
    answer_position: str
    criteria: list[dict[str, Any]] = field(default_factory=list)

    @property
    def input_name(self) -> str:
        return self.input_path.name

    @property
    def output_name(self) -> str:
        # The official evaluator looks for "<id>_output.xlsx"; spaces in ids
        # (Visualization uses "Task 96") are underscored for shell friendliness.
        safe_id = re.sub(r"\s+", "_", self.task_id)
        return f"{safe_id}_output.xlsx"


def load_spreadsheetbench_tasks(
    data_root: Path,
    categories: Sequence[str] = CELL_GRADED_CATEGORIES,
) -> list[SpreadsheetBenchTask]:
    """Load dataset.json rows for the requested categories, validating files."""
    data_root = data_root.expanduser().resolve()
    tasks: list[SpreadsheetBenchTask] = []
    for category in categories:
        if category not in ALL_CATEGORIES:
            raise ValueError(f"Unknown SpreadsheetBench category: {category}")
        category_dir = data_root / category
        dataset_json = category_dir / "dataset.json"
        if not dataset_json.is_file():
            raise FileNotFoundError(f"Missing dataset file: {dataset_json}")
        for row in json.loads(dataset_json.read_text(encoding="utf-8")):
            task_id = str(row["id"])
            input_path = category_dir / row["spreadsheet_path"]
            golden_path = category_dir / row["golden_response_path"]
            for path in (input_path, golden_path):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"{category}/{task_id} references a missing file: {path}"
                    )
            safe_id = re.sub(r"\s+", "_", task_id)
            tasks.append(
                SpreadsheetBenchTask(
                    uid=f"{category}__{safe_id}",
                    category=category,
                    task_id=task_id,
                    instruction=str(row["instruction"]).strip(),
                    input_path=input_path,
                    golden_path=golden_path,
                    answer_position=str(row.get("answer_position", "")),
                    criteria=list(row.get("criteria", [])),
                )
            )
    return tasks


SPREADSHEET_SYSTEM_PROMPT = """You are an autonomous spreadsheet automation engineer working in a Linux sandbox.

Your job: follow the task instruction to transform the input .xlsx workbook and save the completed workbook at the given output path.

You can issue exactly one command per turn. Put the command in exactly one fenced {action_block_language} block and do not write anything after the block.

Available actions:
- Run shell commands. python3 has openpyxl, pandas, numpy, xlsxwriter, and matplotlib installed; LibreOffice (`soffice`) is available for recalculating formulas.
- `view_xlsx <file> list`: list the sheets of a workbook.
- `view_xlsx <file> content "<sheet>" [start_row [end_row]]`: print sheet contents, including formulas and cached values.
- `submit`: finish the task. Only run it after the output workbook exists and you have verified it.

Recommended workflow:
1. Inspect the input workbook with `view_xlsx` to understand its structure before editing.
2. Plan the edits. Modify only the cells the instruction requires; preserve every other cell's value, formula, and formatting.
3. Implement the edits with a python3 + openpyxl script (write it with a heredoc such as `cat > /tmp/edit.py << 'EOF' ... EOF`, then run `python3 /tmp/edit.py`). Keep formulas as formulas where the surrounding model uses formulas.
4. Verify the output file: reopen it and check the edited cells and that other sheets are intact.
5. Run `submit`.

Example:
```{action_block_language}
view_xlsx "$SSB_INPUT" list
```

Example:
```{action_block_language}
submit
```

The environment variables $SSB_INPUT and $SSB_OUTPUT hold the input and output paths. Grading compares computed cell values, so formulas you write must actually evaluate to the right numbers. Never ask follow-up questions."""


class SpreadsheetBenchEnvironment(FullCorpusEnvironment):
    """Writable workspace holding one input workbook.

    Differences from the OfficeQA corpus environment:
    - the workspace holds a copy of the input workbook; the benchmark data
      directory is never mounted because it also contains the golden files;
    - the container runs with ``--network none``, matching the official
      harness's no-network sandbox;
    - ``submit`` takes no answer text and succeeds only once the output
      workbook exists; the recorded submission is the output file name.
    """

    def __init__(
        self,
        config: EnvironmentConfig,
        task: SpreadsheetBenchTask,
        *,
        run_id: str,
    ) -> None:
        if config.sandbox_runtime != "docker":
            raise ValueError(
                "SpreadsheetBenchEnvironment requires the docker sandbox "
                "runtime; the local host lacks the pinned LibreOffice/openpyxl "
                "toolchain"
            )
        super().__init__(config, task, run_id=run_id)  # type: ignore[arg-type]
        self.ssb_task = task

    @property
    def output_host_path(self) -> Path:
        return self.workspace / self.ssb_task.output_name

    def _prepare_workspace(self) -> None:
        task = self.ssb_task
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=False, exist_ok=False)
        shutil.copy2(task.input_path, self.workspace / task.input_name)
        (self.workspace / "instruction.txt").write_text(
            task.instruction + "\n", encoding="utf-8"
        )
        manifest = {
            "uid": task.uid,
            "benchmark": "spreadsheetbench-v2",
            "category": task.category,
            "task_id": task.task_id,
            "input_file": task.input_name,
            "output_file": task.output_name,
            "run_id": self.run_id,
            "workspace": str(self.workspace),
            "sandbox_runtime": self.config.sandbox_runtime,
        }
        (self.workspace / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    def get_task_message(self) -> dict[str, Any]:
        self._require_active()
        task = self.ssb_task
        text = (
            "## Task instructions\n"
            "You need to process a spreadsheet file based on specific "
            "instructions.\n"
            f"Instruction: {task.instruction}\n\n"
            "## Input and Output\n"
            "You are provided with an input spreadsheet file and you need to "
            "generate an output spreadsheet file at the output path.\n"
            f"Input file: /workspace/{task.input_name}\n"
            f"Output path: /workspace/{task.output_name}\n\n"
            "Strictly avoid altering cells that already contain values unless "
            "the instruction requires it, and preserve the original "
            "formatting as much as possible. When the output file is complete "
            "and verified, run `submit`."
        )
        return {
            "role": "user",
            "content": text,
            "extra": {
                "task": {
                    "uid": task.uid,
                    "category": task.category,
                    "task_id": task.task_id,
                    "run_id": self.run_id,
                    "workspace": str(self.workspace),
                    "sandbox_runtime": self.config.sandbox_runtime,
                }
            },
        }

    async def execute(self, action: str | dict[str, str]) -> StepOutput:
        command = action.get("command", "") if isinstance(action, dict) else action
        if command.strip().split(None, 1)[:1] == ["submit"]:
            if self.output_host_path.is_file():
                raise Submitted(self.ssb_task.output_name)
            return StepOutput(
                f"No output file at /workspace/{self.ssb_task.output_name}. "
                "Save the completed workbook there, verify it, then run "
                "`submit`."
            )
        return await super().execute(action)

    def _command_container_resource_args(self) -> list[str]:
        """Benchmark-specific resource guards for the interactive sandbox."""
        return []

    async def _start_command_container(self) -> None:
        if not self.config.sandbox_image:
            raise ValueError("sandbox_image is required for the Docker sandbox runtime")
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            self._container_name,
            "--network",
            "none",
            "--read-only",
            *self._command_container_resource_args(),
            "--tmpfs",
            "/tmp:rw,nosuid,nodev",
            "-v",
            f"{self.workspace}:/workspace",
            "-w",
            "/workspace",
            self.config.sandbox_image,
            "sleep",
            "infinity",
        ]
        returncode, output, timed_out = await self._run_process(
            cmd,
            timeout=max(120, self.config.command_timeout_sec),
        )
        if timed_out:
            raise RuntimeError("Timed out starting Docker command container")
        output = output.strip()
        if returncode != 0:
            raise RuntimeError(f"Failed to start Docker command container: {output}")
        self.command_container_id = output

    def _command_env(self, *, docker: bool) -> dict[str, str]:
        del docker
        task = self.ssb_task
        return {
            # LibreOffice needs a writable profile directory; the container
            # root filesystem is read-only, so point HOME at the tmpfs.
            "HOME": "/tmp",
            "SSB_INPUT": f"/workspace/{task.input_name}",
            "SSB_OUTPUT": f"/workspace/{task.output_name}",
            "SSB_UID": task.uid,
            "SSB_RUN_ID": self.run_id,
        }

    def get_info(self) -> dict[str, Any]:
        task = self.ssb_task
        return {
            "uid": task.uid,
            "benchmark": "spreadsheetbench-v2",
            "category": task.category,
            "task_id": task.task_id,
            "run_id": self.run_id,
            "workspace": str(self.workspace),
            "input_path": str(task.input_path),
            "golden_path": str(task.golden_path),
            "answer_position": task.answer_position,
            "sandbox_runtime": self.config.sandbox_runtime,
            "sandbox_image": self.config.sandbox_image,
            "command_container_id": self.command_container_id,
            "command_history": list(self.command_history),
        }


def build_spreadsheet_system_prompt(action_block_language: str) -> str:
    return SPREADSHEET_SYSTEM_PROMPT.format(
        action_block_language=action_block_language
    )


__all__ = [
    "ALL_CATEGORIES",
    "CELL_GRADED_CATEGORIES",
    "SPREADSHEET_SYSTEM_PROMPT",
    "SpreadsheetBenchEnvironment",
    "SpreadsheetBenchTask",
    "build_spreadsheet_system_prompt",
    "load_spreadsheetbench_tasks",
]
