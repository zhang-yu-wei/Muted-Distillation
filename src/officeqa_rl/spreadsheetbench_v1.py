"""SpreadsheetBench v1 tasks and isolated agent environment.

V1 is an online-judge-style spreadsheet benchmark: the model sees the first
workbook for an instruction and produces one reusable Python solution.  The
evaluation runner replays that solution on every workbook variant belonging
to the instruction.  Golden workbooks and the held-out input variants are
never mounted in the agent container.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from officeqa_rl.full_corpus_env import FullCorpusEnvironment
from officeqa_rl.spreadsheetbench import SpreadsheetBenchEnvironment
from officeqa_rl.types import EnvironmentConfig, StepOutput, Submitted


@dataclass(frozen=True)
class SpreadsheetBenchV1Case:
    """One workbook variant for an instruction."""

    index: int
    input_path: Path
    golden_path: Path

    @property
    def output_name(self) -> str:
        return _case_output_name(self)


@dataclass(frozen=True)
class SpreadsheetBenchV1Task:
    """One instruction and all workbook variants used by its official score."""

    uid: str
    task_id: str
    instruction: str
    instruction_type: str
    answer_position: str
    cases: tuple[SpreadsheetBenchV1Case, ...]

    @property
    def first_case(self) -> SpreadsheetBenchV1Case:
        return self.cases[0]

    @property
    def input_path(self) -> Path:
        """Compatibility alias for the agent-visible first input."""
        return self.first_case.input_path

    @property
    def golden_path(self) -> Path:
        """Compatibility alias; grading only, never exposed to the agent."""
        return self.first_case.golden_path

    @property
    def input_name(self) -> str:
        return self.first_case.input_path.name

    @property
    def output_name(self) -> str:
        return _case_output_name(self.first_case)

    @property
    def solution_name(self) -> str:
        return "solution.py"


def _safe_uid(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("_")


def _case_output_name(case: SpreadsheetBenchV1Case) -> str:
    name = case.input_path.name
    for input_suffix in ("_input.xlsx", "_init.xlsx"):
        if name.endswith(input_suffix):
            return f"{name[:-len(input_suffix)]}_output.xlsx"
    if name == "initial.xlsx":
        return "output.xlsx"
    return f"case_{case.index}_output.xlsx"


def _spreadsheet_dir(data_root: Path, row: dict[str, Any], task_id: str) -> Path:
    """Resolve both released layouts and user-extracted archive layouts."""
    candidates: list[Path] = []
    spreadsheet_path = row.get("spreadsheet_path")
    if isinstance(spreadsheet_path, str) and spreadsheet_path.strip():
        candidate = data_root / spreadsheet_path
        candidates.append(candidate.parent if candidate.suffix else candidate)
    candidates.append(data_root / "spreadsheet" / task_id)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"SpreadsheetBench v1 task {task_id!r} has no workbook directory; "
        f"checked: {', '.join(str(path) for path in candidates)}"
    )


def _load_cases(task_dir: Path, task_id: str) -> tuple[SpreadsheetBenchV1Case, ...]:
    suffix_pairs = (
        ("_input.xlsx", "_answer.xlsx"),
        ("_init.xlsx", "_golden.xlsx"),
    )

    def discover(*, exact: bool) -> list[tuple[int, Path, Path]]:
        discovered: list[tuple[int, Path, Path]] = []
        for input_suffix, golden_suffix in suffix_pairs:
            pattern = (
                re.compile(
                    rf"^(\d+)_{re.escape(task_id)}{re.escape(input_suffix)}$"
                )
                if exact
                else re.compile(rf"^(\d+)_.+{re.escape(input_suffix)}$")
            )
            for input_path in task_dir.glob(f"*{input_suffix}"):
                match = pattern.match(input_path.name)
                if not match:
                    continue
                prefix = input_path.name[: -len(input_suffix)]
                discovered.append(
                    (
                        int(match.group(1)),
                        input_path,
                        input_path.with_name(prefix + golden_suffix),
                    )
                )
        return discovered

    indexed_pairs = discover(exact=True) or discover(exact=False)
    # Five files in the Verified 400 release use these unnumbered names.
    initial_path = task_dir / "initial.xlsx"
    if not indexed_pairs and initial_path.is_file():
        indexed_pairs = [(1, initial_path, task_dir / "golden.xlsx")]

    if not indexed_pairs:
        raise FileNotFoundError(
            f"No supported input workbook (*_input.xlsx, *_init.xlsx, or "
            f"initial.xlsx) found in {task_dir}"
        )

    # The Verified archive packages task 42930 as 1_42930_init.xlsx plus
    # 1_43930_golden.xlsx. Accept a lone golden workbook as the unambiguous
    # pair, while preserving strict filename matching for multi-case tasks.
    if len(indexed_pairs) == 1 and not indexed_pairs[0][2].is_file():
        golden_candidates = [
            path
            for path in task_dir.glob("*.xlsx")
            if path.name == "golden.xlsx" or path.name.endswith("_golden.xlsx")
        ]
        if len(golden_candidates) == 1:
            index, input_path, _ = indexed_pairs[0]
            indexed_pairs = [(index, input_path, golden_candidates[0])]

    cases: list[SpreadsheetBenchV1Case] = []
    seen: set[int] = set()
    for index, input_path, golden_path in sorted(indexed_pairs):
        if index in seen:
            raise ValueError(f"Duplicate case index {index} in {task_dir}")
        seen.add(index)
        if not golden_path.is_file():
            raise FileNotFoundError(
                f"Missing answer workbook for {input_path}: {golden_path}"
            )
        cases.append(
            SpreadsheetBenchV1Case(
                index=index,
                input_path=input_path.resolve(),
                golden_path=golden_path.resolve(),
            )
        )
    return tuple(cases)


def _answer_position(row: dict[str, Any]) -> str:
    """Combine Verified's separate answer_sheet with its cell ranges."""
    answer_position = str(row["answer_position"])
    answer_sheet = row.get("answer_sheet")
    if not isinstance(answer_sheet, str) or not answer_sheet.strip():
        return answer_position
    quoted_sheet = "'" + answer_sheet.strip().replace("'", "''") + "'"
    return ",".join(
        part if "!" in part else f"{quoted_sheet}!{part}"
        for part in answer_position.split(",")
    )


def load_spreadsheetbench_v1_tasks(data_root: Path) -> list[SpreadsheetBenchV1Task]:
    """Load a full or Verified v1 release rooted at ``dataset.json``."""
    data_root = data_root.expanduser().resolve()
    dataset_json = data_root / "dataset.json"
    if not dataset_json.is_file():
        raise FileNotFoundError(
            f"Missing SpreadsheetBench v1 dataset file: {dataset_json}. "
            "Point --data-root at the extracted release directory."
        )

    rows = json.loads(dataset_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {dataset_json}")

    tasks: list[SpreadsheetBenchV1Task] = []
    seen_uids: set[str] = set()
    for row in rows:
        task_id = str(row["id"])
        uid = f"SSB1__{_safe_uid(task_id)}"
        if uid in seen_uids:
            raise ValueError(f"Duplicate SpreadsheetBench v1 task id: {task_id}")
        seen_uids.add(uid)
        task_dir = _spreadsheet_dir(data_root, row, task_id)
        tasks.append(
            SpreadsheetBenchV1Task(
                uid=uid,
                task_id=task_id,
                instruction=str(row["instruction"]).strip(),
                instruction_type=str(row.get("instruction_type", "unknown")),
                answer_position=_answer_position(row),
                cases=_load_cases(task_dir, task_id),
            )
        )
    return tasks


SPREADSHEET_V1_SYSTEM_PROMPT = """You are an autonomous spreadsheet automation engineer working in a Linux sandbox.

Your job: solve the spreadsheet instruction by writing a reusable Python program at /workspace/solution.py. The program must read the input path from $SSB_INPUT and save the completed workbook to $SSB_OUTPUT. It will be evaluated on additional workbook variants that have the same task but different values, so never hard-code values copied from this workbook.

You can issue exactly one command per turn. Put the command in exactly one fenced {action_block_language} block and do not write anything after the block.

Available actions:
- Run shell commands. python3 has openpyxl, pandas, numpy, xlsxwriter, and matplotlib installed; LibreOffice (`soffice`) is available.
- `view_xlsx <file> list`: list workbook sheets.
- `view_xlsx <file> content "<sheet>" [start_row [end_row]]`: inspect values and formulas.
- `submit`: finish. It succeeds only after both /workspace/solution.py and $SSB_OUTPUT exist.

Required workflow:
1. Use the supplied Instruction type and Answer position, then inspect $SSB_INPUT and infer a general solution from labels, formulas, and workbook structure. Answer position identifies the destination cells; it does not contain their correct values.
2. Write /workspace/solution.py. It must use os.environ["SSB_INPUT"] and os.environ["SSB_OUTPUT"], preserve unrelated cells and formatting, and derive results from workbook contents rather than fixed answers.
3. Run `python3 /workspace/solution.py` to produce $SSB_OUTPUT for the visible workbook.
4. Reopen and verify the output, then run `submit`.

Example:
```{action_block_language}
view_xlsx "$SSB_INPUT" list
```

Example:
```{action_block_language}
submit
```

Never ask follow-up questions. Golden workbooks and held-out variants are not available in the sandbox."""


class SpreadsheetBenchV1Environment(SpreadsheetBenchEnvironment):
    """Agent sandbox containing case 1 only and requiring reusable code."""

    # Match the limits already applied to saved-solution replay. An unlimited
    # interactive command can otherwise OOM the host and kill all DP workers.
    CONTAINER_MEMORY_LIMIT = "8g"
    CONTAINER_PIDS_LIMIT = 256

    def _command_container_resource_args(self) -> list[str]:
        return [
            "--memory", self.CONTAINER_MEMORY_LIMIT,
            "--memory-swap", self.CONTAINER_MEMORY_LIMIT,
            "--pids-limit", str(self.CONTAINER_PIDS_LIMIT),
        ]

    def __init__(
        self,
        config: EnvironmentConfig,
        task: SpreadsheetBenchV1Task,
        *,
        run_id: str,
    ) -> None:
        super().__init__(config, task, run_id=run_id)  # type: ignore[arg-type]
        self.ssb_task = task  # narrow the runtime object after the shared init

    @property
    def solution_host_path(self) -> Path:
        return self.workspace / self.ssb_task.solution_name

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
            "benchmark": "spreadsheetbench-v1",
            "task_id": task.task_id,
            "instruction_type": task.instruction_type,
            "answer_position": task.answer_position,
            "input_file": task.input_name,
            "output_file": task.output_name,
            "solution_file": task.solution_name,
            "visible_case": task.first_case.index,
            "case_count": len(task.cases),
            "run_id": self.run_id,
            "workspace": str(self.workspace),
            "sandbox_runtime": self.config.sandbox_runtime,
            "container_memory_limit": self.CONTAINER_MEMORY_LIMIT,
            "container_pids_limit": self.CONTAINER_PIDS_LIMIT,
        }
        (self.workspace / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    def get_task_message(self) -> dict[str, Any]:
        self._require_active()
        task = self.ssb_task
        return {
            "role": "user",
            "content": (
                "## SpreadsheetBench v1 task\n"
                f"Instruction: {task.instruction}\n"
                f"Instruction type: {task.instruction_type}\n"
                f"Answer position: {task.answer_position}\n\n"
                f"Visible input: /workspace/{task.input_name}\n"
                f"Visible output: /workspace/{task.output_name}\n"
                f"Reusable program: /workspace/{task.solution_name}\n\n"
                "Create a general solution.py, run it on the visible input, "
                "verify the output, and submit. The evaluator will run the "
                "same program on the other workbook variants."
            ),
            "extra": {
                "task": {
                    "uid": task.uid,
                    "task_id": task.task_id,
                    "instruction_type": task.instruction_type,
                    "answer_position": task.answer_position,
                    "run_id": self.run_id,
                    "workspace": str(self.workspace),
                    "sandbox_runtime": self.config.sandbox_runtime,
                }
            },
        }

    async def execute(self, action: str | dict[str, str]) -> StepOutput:
        command = action.get("command", "") if isinstance(action, dict) else action
        if command.strip().split(None, 1)[:1] == ["submit"]:
            missing = []
            if not self.solution_host_path.is_file():
                missing.append(f"/workspace/{self.ssb_task.solution_name}")
            if not self.output_host_path.is_file():
                missing.append(f"/workspace/{self.ssb_task.output_name}")
            if not missing:
                raise Submitted(self.ssb_task.solution_name)
            return StepOutput(
                "Cannot submit yet; missing " + ", ".join(missing) + "."
            )
        # Skip SpreadsheetBenchEnvironment.execute because its submit check
        # accepts an output workbook without the required reusable program.
        return await FullCorpusEnvironment.execute(self, action)

    def _command_env(self, *, docker: bool) -> dict[str, str]:
        del docker
        task = self.ssb_task
        return {
            "HOME": "/tmp",
            "SSB_INPUT": f"/workspace/{task.input_name}",
            "SSB_OUTPUT": f"/workspace/{task.output_name}",
            "SSB_SOLUTION": f"/workspace/{task.solution_name}",
            "SSB_UID": task.uid,
            "SSB_RUN_ID": self.run_id,
        }

    def get_info(self) -> dict[str, Any]:
        task = self.ssb_task
        return {
            "uid": task.uid,
            "benchmark": "spreadsheetbench-v1",
            "task_id": task.task_id,
            "instruction_type": task.instruction_type,
            "run_id": self.run_id,
            "workspace": str(self.workspace),
            "visible_input_path": str(task.input_path),
            "case_count": len(task.cases),
            "answer_position": task.answer_position,
            "sandbox_runtime": self.config.sandbox_runtime,
            "sandbox_image": self.config.sandbox_image,
            "container_memory_limit": self.CONTAINER_MEMORY_LIMIT,
            "container_pids_limit": self.CONTAINER_PIDS_LIMIT,
            "command_container_id": self.command_container_id,
            "command_history": list(self.command_history),
        }


def build_spreadsheet_v1_system_prompt(action_block_language: str) -> str:
    return SPREADSHEET_V1_SYSTEM_PROMPT.format(
        action_block_language=action_block_language
    )


__all__ = [
    "SPREADSHEET_V1_SYSTEM_PROMPT",
    "SpreadsheetBenchV1Case",
    "SpreadsheetBenchV1Environment",
    "SpreadsheetBenchV1Task",
    "build_spreadsheet_v1_system_prompt",
    "load_spreadsheetbench_v1_tasks",
]
