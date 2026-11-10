"""Grading for SpreadsheetBench 2 outputs.

Two steps, both matching the official pipeline in
``third_party/SpreadsheetBench-2``:

1. ``recalc_with_libreoffice`` refreshes the output workbook's cached formula
   values (their ``evaluation/open_spreadsheet.py`` step) inside the sandbox
   image, because agent-written files carry formulas without computed values.
2. ``compare_output`` scores the recalculated workbook against the golden one
   using the official comparison primitives imported from their
   ``evaluation/evaluation.py`` (loaded from ``third_party`` unmodified).

``compare_output`` mirrors their ``compare_workbooks_with_regression`` +
``process_single_item`` aggregation, re-implemented only so the per-cell
mismatch messages survive for failure analysis (the official function keeps
just the first one).
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import subprocess
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_EVALUATION_PY = (
    _REPO_ROOT / "third_party" / "SpreadsheetBench-2" / "evaluation" / "evaluation.py"
)
DEFAULT_SANDBOX_IMAGE = "spreadsheetbench-v2-sandbox"

_official_module: Any = None


def official_evaluation():
    """Load the benchmark's evaluation.py once, from third_party, unmodified."""
    global _official_module
    if _official_module is None:
        spec = importlib.util.spec_from_file_location(
            "ssb2_official_evaluation", OFFICIAL_EVALUATION_PY
        )
        if spec is None or spec.loader is None:
            raise FileNotFoundError(
                f"Official evaluation code not found at {OFFICIAL_EVALUATION_PY}; "
                "clone https://github.com/RUCKBReasoning/SpreadsheetBench-2 "
                "into third_party/"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _official_module = module
    return _official_module


def comparison_flags(category: str, spreadsheet_path: str) -> tuple[bool, bool]:
    """(with_font_color, with_formula), per the official process_single_item."""
    if category != "Debugging":
        return False, False
    return "Color" in spreadsheet_path, "Embedded" in spreadsheet_path


async def recalc_with_libreoffice(
    xlsx_path: Path,
    *,
    sandbox_image: str = DEFAULT_SANDBOX_IMAGE,
    timeout_sec: float = 600.0,
) -> None:
    """Recalculate one workbook in place inside a one-shot sandbox container.

    /usr/bin/python3 (not the image's default 3.11) owns the UNO bindings.
    """
    xlsx_path = xlsx_path.resolve()
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        # Run as the host user so the rewritten workbook stays owned by the
        # caller (a root-owned result file breaks reruns and cleanup).
        # LibreOffice needs a writable profile dir for any uid: HOME=/tmp.
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "-v",
        f"{xlsx_path.parent}:/data",
        sandbox_image,
        "/usr/bin/python3",
        "/opt/recalc_xlsx.py",
        f"/data/{xlsx_path.name}",
    ]
    process = await asyncio.to_thread(
        subprocess.run,
        command,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    if process.returncode != 0:
        detail = (process.stdout + process.stderr).strip()
        raise RuntimeError(f"LibreOffice recalculation failed: {detail[-2000:]}")


def compare_output(
    input_file: Path,
    golden_file: Path,
    output_file: Path,
    answer_position: str,
    *,
    with_font_color: bool = False,
    with_formula: bool = False,
    max_cell_errors: int = 50,
) -> dict[str, Any]:
    """Score one output workbook with the official cell comparison.

    Returns regression/modification/strict accuracy plus a capped list of
    per-cell mismatch messages for analysis.
    """
    import openpyxl

    ev = official_evaluation()
    empty = {
        "accuracy": 0.0,
        "regression_accuracy": 0.0,
        "modification_accuracy": 0.0,
        "regression_cells": {"correct": 0, "total": 0},
        "modification_cells": {"correct": 0, "total": 0},
        "cell_errors": [],
        "cell_error_count": 0,
    }
    if not output_file.is_file():
        return {**empty, "error": "output file not exist"}
    if not input_file.is_file():
        return {**empty, "error": "input file not exist"}

    need_raw = with_formula
    try:
        wb_input = openpyxl.load_workbook(filename=input_file, data_only=not need_raw)
        wb_gt = openpyxl.load_workbook(filename=golden_file, data_only=not need_raw)
        wb_proc = openpyxl.load_workbook(filename=output_file, data_only=not need_raw)
        if not with_formula:
            wb_input_f = openpyxl.load_workbook(filename=input_file, data_only=False)
            wb_gt_f = openpyxl.load_workbook(filename=golden_file, data_only=False)
            wb_proc_f = openpyxl.load_workbook(filename=output_file, data_only=False)
        else:
            wb_input_f = wb_gt_f = wb_proc_f = None
    except Exception as error:
        return {**empty, "error": f"workbook load failed: {error}"}

    reg = {"correct": 0, "total": 0}
    mod = {"correct": 0, "total": 0}
    cell_errors: list[str] = []
    error_count = 0

    for sheet_cell_range in ev.parse_answer_position(answer_position):
        if "!" in sheet_cell_range:
            sheet_name, cell_range = sheet_cell_range.split("!")
        else:
            sheet_name, cell_range = wb_gt.sheetnames[0], sheet_cell_range
        sheet_name = sheet_name.strip("'").strip()
        cell_range = cell_range.strip("'").strip()

        regression_cells, modification_cells = ev.classify_cells_by_modification(
            wb_input,
            wb_gt,
            sheet_name,
            cell_range,
            with_font_color,
            with_formula,
            wb_input_formula=wb_input_f,
            wb_answer_formula=wb_gt_f,
        )
        (
            reg_correct,
            reg_total,
            mod_correct,
            mod_total,
            messages,
        ) = ev.cell_level_compare_with_classification(
            wb_gt,
            wb_proc,
            sheet_name,
            regression_cells,
            modification_cells,
            with_font_color,
            with_formula,
            wb_answer_formula=wb_gt_f,
            wb_output_formula=wb_proc_f,
        )
        reg["correct"] += reg_correct
        reg["total"] += reg_total
        mod["correct"] += mod_correct
        mod["total"] += mod_total
        error_count += len(messages)
        room = max_cell_errors - len(cell_errors)
        if room > 0:
            cell_errors.extend(str(message) for message in messages[:room])

    reg_ratio = round(reg["correct"] / reg["total"], 4) if reg["total"] else 0.0
    mod_ratio = round(mod["correct"] / mod["total"], 4) if mod["total"] else 0.0
    # Official leniency: >= 99.8% regression counts as fully preserved.
    if reg_ratio >= 0.998:
        reg_ratio = 1.0
    accuracy = 1.0 if reg_ratio == 1.0 and mod_ratio == 1.0 else 0.0

    return {
        "accuracy": accuracy,
        "regression_accuracy": reg_ratio,
        "modification_accuracy": mod_ratio,
        "regression_cells": reg,
        "modification_cells": mod,
        "cell_errors": cell_errors,
        "cell_error_count": error_count,
        "error": "",
    }


__all__ = [
    "DEFAULT_SANDBOX_IMAGE",
    "OFFICIAL_EVALUATION_PY",
    "compare_output",
    "comparison_flags",
    "official_evaluation",
    "recalc_with_libreoffice",
]
