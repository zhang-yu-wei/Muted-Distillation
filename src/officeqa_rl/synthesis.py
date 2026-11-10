"""Synthetic QA generation for SFT data.

Pure logic only: table-region extraction from the transformed corpus,
authoring-prompt construction, and optional audit/dedup helpers. Author-only
generation does not run answer-verification gates; label validation happens
later through teacher/Gemini result agreement. The Gemini client lives in
``officeqa_rl.gemini_author``; orchestration lives in
``scripts/synthesize_qa.py``.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from officeqa_rl.scoring import score_answer

DEFAULT_TOLERANCE = 0.01
DIFFICULTY_BY_TIER = {"T1": "easy", "T2": "hard", "T3": "hard"}
JACCARD_THRESHOLD = 0.5

# Numeric table cell, tolerant of the corpus' formatting quirks: thousands
# commas, decimals, accounting parentheses, percent signs, unicode minus, and
# trailing revision/preliminary markers such as "3,293r".
_NUMERIC_CELL_RE = re.compile(
    r"[(\-−]?\s*\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*%?\s*[)]?\s*[a-z]{0,2}",
    re.IGNORECASE,
)
_SEPARATOR_CELL_RE = re.compile(r":?-{3,}:?")


# ---------------------------------------------------------------------------
# Table-region extraction
# ---------------------------------------------------------------------------


@dataclass
class TableRegion:
    """One contiguous Markdown pipe table plus its preceding context lines."""

    source_file: str
    start_line: int  # 1-indexed, inclusive, first table line
    end_line: int  # 1-indexed, inclusive
    context: str
    table_text: str
    n_rows: int  # data rows (header and separator excluded)
    n_cols: int
    numeric_density: float

    def excerpt(self) -> str:
        parts = []
        if self.context:
            parts.append(self.context)
        parts.append(self.table_text)
        return "\n".join(parts)

    def span(self) -> dict[str, int]:
        return {"start_line": self.start_line, "end_line": self.end_line}


def _split_table_row(line: str) -> list[str]:
    cells = line.strip().strip("|").split("|")
    return [cell.strip() for cell in cells]


def _is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(_SEPARATOR_CELL_RE.fullmatch(cell) for cell in cells if cell)


def _is_numeric_cell(cell: str) -> bool:
    cell = cell.strip()
    if not cell or cell.lower() == "nan":
        return False
    return _NUMERIC_CELL_RE.fullmatch(cell) is not None


def extract_table_regions(
    text: str,
    source_file: str,
    *,
    min_rows: int = 4,
    min_cols: int = 3,
    min_numeric_density: float = 0.5,
    context_lines: int = 8,
) -> list[TableRegion]:
    """Find qualifying Markdown pipe tables in one transformed corpus file."""
    lines = text.splitlines()
    regions: list[TableRegion] = []
    block_start: int | None = None

    def close_block(start: int, end: int) -> None:
        block = lines[start:end]
        rows = [_split_table_row(line) for line in block]
        data_rows = [row for row in rows[1:] if not _is_separator_row(row)]
        if len(data_rows) < min_rows:
            return
        n_cols = max(len(row) for row in rows)
        if n_cols < min_cols:
            return
        cells = [cell for row in data_rows for cell in row if cell and cell.lower() != "nan"]
        if not cells:
            return
        numeric = sum(1 for cell in cells if _is_numeric_cell(cell))
        density = numeric / len(cells)
        if density < min_numeric_density:
            return
        context_start = max(0, start - 1)
        context: list[str] = []
        for line in reversed(lines[:start]):
            if line.strip().startswith("|"):
                break
            if line.strip():
                context.append(line.strip())
            if len(context) >= context_lines:
                break
            context_start -= 1
        regions.append(
            TableRegion(
                source_file=source_file,
                start_line=start + 1,
                end_line=end,
                context="\n".join(reversed(context)),
                table_text="\n".join(block),
                n_rows=len(data_rows),
                n_cols=n_cols,
                numeric_density=density,
            )
        )

    for index, line in enumerate(lines):
        if line.strip().startswith("|"):
            if block_start is None:
                block_start = index
        elif block_start is not None:
            close_block(block_start, index)
            block_start = None
    if block_start is not None:
        close_block(block_start, len(lines))
    return regions


def window_around(text: str, region: TableRegion, *, max_chars: int) -> str:
    """A window of the source file centered on the region, capped by size.

    Verifier models re-derive the answer from this window instead of the full
    file: the largest transformed bulletins exceed non-Gemini context limits.
    """
    if len(text) <= max_chars:
        return text
    lines = text.splitlines()
    start = region.start_line - 1
    end = region.end_line
    chunk = lines[start:end]
    size = sum(len(line) + 1 for line in chunk)
    lo, hi = start, end
    while size < max_chars and (lo > 0 or hi < len(lines)):
        if lo > 0:
            lo -= 1
            size += len(lines[lo]) + 1
        if size < max_chars and hi < len(lines):
            size += len(lines[hi]) + 1
            hi += 1
    return "\n".join(lines[lo:hi])


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


_SOURCE_FILE_YEAR_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})")


def source_file_year(source_file: str) -> int | None:
    match = _SOURCE_FILE_YEAR_RE.search(source_file)
    return int(match.group(1)) if match else None


def decade_of(source_file: str) -> str:
    year = source_file_year(source_file)
    return f"{year - year % 10}s" if year is not None else "unknown"


def round_robin_sample(buckets: dict[str, list[Any]], n: int, rng: Any) -> list[Any]:
    """Draw ~evenly across buckets (decade stratification)."""
    pools = {key: list(values) for key, values in buckets.items() if values}
    for pool in pools.values():
        rng.shuffle(pool)
    order = sorted(pools)
    rng.shuffle(order)
    picked: list[Any] = []
    while len(picked) < n and pools:
        for key in list(order):
            pool = pools.get(key)
            if not pool:
                pools.pop(key, None)
                order.remove(key)
                continue
            picked.append(pool.pop())
            if len(picked) >= n:
                break
    return picked


def _heading_words(region: TableRegion) -> set[str]:
    heading = region.context.splitlines()[-1] if region.context else ""
    return {word for word in re.findall(r"[a-z]+", heading.lower()) if len(word) > 2}


def find_cross_year_pair(
    region: TableRegion,
    candidates: Iterable[TableRegion],
    *,
    min_word_jaccard: float = 0.6,
) -> TableRegion | None:
    """Find the same table family in a different year's bulletin (for T3)."""
    words = _heading_words(region)
    if not words:
        return None
    year = source_file_year(region.source_file)
    best: tuple[float, TableRegion] | None = None
    for other in candidates:
        if other.source_file == region.source_file:
            continue
        if year is not None and source_file_year(other.source_file) == year:
            continue
        other_words = _heading_words(other)
        if not other_words:
            continue
        union = words | other_words
        similarity = len(words & other_words) / len(union)
        if similarity >= min_word_jaccard and (best is None or similarity > best[0]):
            best = (similarity, other)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Authoring prompt and response parsing
# ---------------------------------------------------------------------------


TIER_INSTRUCTIONS = {
    "T1": (
        "T1 (single lookup): ask for one value from the table, optionally via a "
        "selection criterion such as the highest/lowest value in a labelled span."
    ),
    "T2": (
        "T2 (multi-step, one table): ask for a sum, difference, percent change, "
        "or mean computed from several cells of this table."
    ),
    "T3": (
        "T3 (cross-document): ask for a comparison or computation combining one "
        "value from EACH of the two excerpts (different bulletin issues)."
    ),
}

AUTHORED_FIELDS = ("question", "answer", "difficulty_tier", "cells_used", "computation")


def build_authoring_prompt(
    excerpts: Sequence[tuple[str, str]],
    tier: str,
    exemplars: Sequence[str],
) -> str:
    """Prompt an author model for one strict-JSON question grounded in excerpts."""
    if tier not in TIER_INSTRUCTIONS:
        raise ValueError(f"Unknown tier: {tier}")
    blocks = []
    for source_file, excerpt in excerpts:
        blocks.append(f"### Source excerpt from {source_file}\n{excerpt}")
    exemplar_block = "\n".join(f"- {question}" for question in exemplars)
    return f"""You write quantitative benchmark questions over U.S. Treasury Bulletin tables.

{chr(10).join(blocks)}

Write exactly ONE question of tier {tier}.
{TIER_INSTRUCTIONS[tier]}

Style exemplars from the real benchmark (match their tone and specificity):
{exemplar_block}

Hard requirements:
- The question must be fully self-contained: name the exact period (month/year,
  FY/CY), the exact quantity, and the unit exactly as the table reports it
  (e.g. "in millions of nominal dollars"). Do not mention the table, the
  excerpt, or the bulletin issue by position; a reader with the whole corpus
  must be able to answer it.
- The answer must be a single number (state rounding in the question if any).
- Copy every table value you use into cells_used verbatim, including commas,
  exactly as printed in the excerpt.
- computation must be a pure arithmetic Python expression over those values
  (commas removed), e.g. "(2602 + 1987) / 2". No variables, no functions other
  than round/abs/min/max/sum.
- Do not invent values. If the excerpt cannot support a good {tier} question,
  output {{"skip": true}}.

Output ONLY a JSON object, no prose, with fields:
{{"question": str, "answer": str, "difficulty_tier": "{tier}",
 "cells_used": [{{"row_label": str, "col_label": str, "raw_value": str,
                 "source_file": str}}],
 "computation": str, "units_and_rounding": str}}"""


def parse_authored_json(text: str) -> dict[str, Any]:
    """Parse the author model's strict-JSON reply (tolerating code fences)."""
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("author reply contains no JSON object")
    data = json.loads(stripped[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("author reply is not a JSON object")
    if data.get("skip"):
        return {"skip": True}
    missing = [key for key in AUTHORED_FIELDS if not data.get(key)]
    if missing:
        raise ValueError(f"author reply is missing fields: {', '.join(missing)}")
    if not isinstance(data["cells_used"], list) or not data["cells_used"]:
        raise ValueError("cells_used must be a non-empty list")
    if len(data["cells_used"]) > 24:
        raise ValueError("cells_used is implausibly large")
    question = str(data["question"]).strip()
    if not 20 <= len(question) <= 600:
        raise ValueError("question length out of bounds")
    if not re.search(r"\d", str(data["answer"])):
        raise ValueError("answer contains no digits")
    return data


# ---------------------------------------------------------------------------
# Optional grounding audit helper (not run during author-only generation)
# ---------------------------------------------------------------------------


def normalize_for_grounding(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("−", "-"))


def check_grounding(
    cells: Sequence[dict[str, Any]],
    normalized_sources: dict[str, str],
) -> tuple[bool, list[str]]:
    """Every raw_value must appear verbatim in one of the declared files."""
    failures = []
    for cell in cells:
        raw = normalize_for_grounding(str(cell.get("raw_value", "")).strip())
        if not raw:
            failures.append("empty raw_value")
            continue
        if not any(raw in source for source in normalized_sources.values()):
            failures.append(raw)
    return not failures, failures


# ---------------------------------------------------------------------------
# Optional computation audit helper (not run during author-only generation)
# ---------------------------------------------------------------------------


_ALLOWED_CALLS = {"round", "abs", "min", "max", "sum"}
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Tuple,
    ast.List,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


def safe_eval_expression(expression: str) -> float:
    """Evaluate a pure-arithmetic expression; reject anything else."""
    if len(expression) > 500:
        raise ValueError("expression too long")
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_CALLS:
            raise ValueError(f"disallowed name: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_CALLS:
                raise ValueError("disallowed call")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("non-numeric constant")
    value = eval(  # noqa: S307 - AST-whitelisted arithmetic only
        compile(tree, "<computation>", "eval"),
        {"__builtins__": {}},
        {"round": round, "abs": abs, "min": min, "max": max, "sum": sum},
    )
    return float(value)


def cell_numeric_value(raw_value: str) -> float:
    """Parse a table cell into a number ("(1,613)" -> -1613, "3,293r" -> 3293)."""
    text = normalize_for_grounding(str(raw_value)).strip()
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").strip()
    text = re.sub(r"[a-z]{1,2}$", "", text, flags=re.IGNORECASE).strip()
    text = text.rstrip("%").replace("$", "").replace(",", "").strip()
    value = float(text)
    return -value if negative and value > 0 else value


def format_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def check_literals(expression: str, cells: Sequence[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Every literal in the computation must be a quoted cell value or a small
    constant (counts, 100 for percents); catches invented numbers."""
    cell_values = set()
    for cell in cells:
        try:
            cell_values.add(abs(cell_numeric_value(cell.get("raw_value", ""))))
        except (ValueError, TypeError):
            return False, [f"unparseable cell: {cell.get('raw_value')!r}"]
    tree = ast.parse(expression, mode="eval")
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            literal = abs(float(node.value))
            if literal <= 100:
                continue
            if any(abs(literal - value) <= 1e-9 for value in cell_values):
                continue
            offenders.append(format_number(literal))
    return not offenders, offenders


def check_computation(
    answer: str,
    computation: str,
    cells: Sequence[dict[str, Any]],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[bool, str]:
    """Re-execute the derivation and grade it against the authored answer."""
    literals_ok, offenders = check_literals(computation, cells)
    if not literals_ok:
        return False, f"literals not from cells: {offenders}"
    try:
        value = safe_eval_expression(computation)
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as error:
        return False, f"eval failed: {error}"
    computed = format_number(value)
    if score_answer(answer, computed, tolerance) != 1.0:
        return False, f"computed {computed} does not match authored answer {answer!r}"
    return True, computed


# ---------------------------------------------------------------------------
# Dedup helpers used when assembling the final SFT set in Step 6
# ---------------------------------------------------------------------------


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.lower()).strip()


def char_ngrams(text: str, n: int = 5) -> set[str]:
    text = normalize_question(text)
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def max_question_similarity(
    question: str,
    reference_questions: Sequence[str],
) -> tuple[float, str]:
    grams = char_ngrams(question)
    best_score, best_question = 0.0, ""
    for reference in reference_questions:
        score = jaccard(grams, char_ngrams(reference))
        if score > best_score:
            best_score, best_question = score, reference
    return best_score, best_question


def normalized_answer_key(answer: str) -> str:
    return normalize_for_grounding(str(answer)).replace(",", "").strip().lower()


def answer_collision(
    source_files: Sequence[str],
    answer: str,
    benchmark_rows: Sequence[dict[str, str]],
) -> bool:
    """Same (source_file, answer) as a benchmark row = same cell asked differently."""
    key = normalized_answer_key(answer)
    files = set(source_files)
    for row in benchmark_rows:
        row_files = {name.strip() for name in (row.get("source_files") or "").splitlines()}
        if files & row_files and normalized_answer_key(row.get("answer") or "") == key:
            return True
    return False


__all__ = [
    "DIFFICULTY_BY_TIER",
    "JACCARD_THRESHOLD",
    "TableRegion",
    "answer_collision",
    "build_authoring_prompt",
    "cell_numeric_value",
    "char_ngrams",
    "check_computation",
    "check_grounding",
    "check_literals",
    "decade_of",
    "extract_table_regions",
    "find_cross_year_pair",
    "format_number",
    "jaccard",
    "max_question_similarity",
    "normalize_for_grounding",
    "normalize_question",
    "parse_authored_json",
    "round_robin_sample",
    "safe_eval_expression",
    "source_file_year",
    "window_around",
]
