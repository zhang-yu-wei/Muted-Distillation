"""Filter OfficeQA teacher traces and build a training-ready slime SFT set.

Each accepted episode becomes one row per assistant turn.  Earlier assistant
turns are retained as context with ``step_loss_mask=0``; only the current
teacher turn is supervised.  A reasoning turn reconstructs the teacher output
as ``<think>reasoning_content</think>`` followed by its fenced bash action. A
direct command that did not need explicit reasoning remains a command-only
target.

JSONL is dependency-free and is accepted directly by slime.  A ``.parquet``
output is also supported when pyarrow is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from officeqa_rl.dataset import read_csv_rows
from officeqa_rl.scoring import score_answer
from officeqa_rl.sft_samples import per_turn_training_messages
from officeqa_rl.synthesis import (
    JACCARD_THRESHOLD,
    answer_collision,
    max_question_similarity,
    normalize_question,
)


_BASH_BLOCK_RE = re.compile(r"```bash[ \t]*\n(.*?)\n```", re.DOTALL)
_FENCED_BLOCK_RE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\n.*?\n```", re.DOTALL)
DEFAULT_UID_PATTERN = r"SYN\d+"
_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--dataset-csv", type=Path, required=True)
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        action="append",
        required=True,
        help="Benchmark CSV to exclude; repeat for multiple files",
    )
    parser.add_argument(
        "--provenance-jsonl",
        type=Path,
        help="Synthetic provenance JSONL (auto-detected beside --dataset-csv)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--episode-policy",
        choices=["filtered", "all"],
        default="filtered",
        help="filtered applies outcome/quality gates; all keeps failed episodes",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest path (default: <output stem>_manifest.json)",
    )
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument(
        "--max-duplicate-command-ratio",
        type=float,
        default=0.30,
        help="Reject when (command count - unique command count) / command count exceeds this",
    )
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=JACCARD_THRESHOLD,
        help="Reject character-5-gram similarity at or above this value",
    )
    parser.add_argument(
        "--uid-pattern",
        default=DEFAULT_UID_PATTERN,
        help=(
            "Regex every dataset and trace uid must fully match; trace files may "
            "add a .s<i> attempt suffix (<uid>.s<i>.json) for multi-sample runs"
        ),
    )
    parser.add_argument(
        "--skip-answer-collision",
        action="store_true",
        help=(
            "Disable the benchmark source/answer collision filter. The filter guards "
            "synthetic questions against restating a benchmark cell; on a real "
            "benchmark train split, sharing a source cell with an excluded "
            "benchmark row is expected, not a leak"
        ),
    )
    parser.add_argument(
        "--allow-missing-traces",
        action="store_true",
        help="Build from a partial trace directory; intended for audits, not final training data",
    )
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if not 0.0 <= args.max_duplicate_command_ratio <= 1.0:
        parser.error("--max-duplicate-command-ratio must be in [0, 1]")
    if not 0.0 <= args.jaccard_threshold <= 1.0:
        parser.error("--jaccard-threshold must be in [0, 1]")
    if args.output.suffix not in {".jsonl", ".parquet"}:
        parser.error("--output must end in .jsonl or .parquet")
    try:
        re.compile(args.uid_pattern)
    except re.error as error:
        parser.error(f"--uid-pattern must be a valid regex: {error}")
    return args


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _auto_provenance_path(dataset_csv: Path) -> Path | None:
    candidate = dataset_csv.with_name(f"{dataset_csv.stem}_provenance.jsonl")
    return candidate if candidate.is_file() else None


def _load_provenance_refs(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    refs: dict[str, str] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            uid = item.get("uid")
            if isinstance(uid, str) and uid:
                refs[uid] = f"{path.resolve()}#L{line_number}"
    return refs


def _unique_rows_by_uid(
    rows: list[dict[str, str]], source: Path, uid_re: re.Pattern[str]
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        uid = (row.get("uid") or "").strip()
        if not uid_re.fullmatch(uid):
            raise ValueError(
                f"Every SFT uid must fully match {uid_re.pattern!r}; got {uid!r} in {source}"
            )
        if uid in indexed:
            raise ValueError(f"Duplicate uid {uid!r} in {source}")
        indexed[uid] = row
    if not indexed:
        raise ValueError(f"{source} contains no rows")
    return indexed


def _source_files(row: dict[str, str]) -> list[str]:
    return [item.strip() for item in (row.get("source_files") or "").splitlines() if item.strip()]


def _valid_bash_content(content: Any) -> bool:
    if not isinstance(content, str):
        return False
    matches = list(_BASH_BLOCK_RE.finditer(content))
    if (
        len(matches) != 1
        or len(_FENCED_BLOCK_RE.findall(content)) != 1
        or not matches[0].group(1).strip()
    ):
        return False
    return not content[matches[0].end() :].strip()


def _valid_direct_bash_content(content: Any) -> bool:
    """A command-only response: one bash fence and no surrounding prose."""
    if not _valid_bash_content(content):
        return False
    match = _BASH_BLOCK_RE.search(content)
    assert match is not None
    return not content[: match.start()].strip()


def _duplicate_command_ratio(trace: dict[str, Any]) -> float:
    metadata = trace.get("metadata") or {}
    environment = metadata.get("environment") or {}
    history = environment.get("command_history") or []
    commands = [
        item.get("command", "").strip()
        for item in history
        if isinstance(item, dict)
        and isinstance(item.get("command"), str)
        and item["command"].strip()
    ]
    if not commands:
        return 0.0
    return (len(commands) - len(set(commands))) / len(commands)


def _teacher_model(trace: dict[str, Any]) -> str | None:
    backend = (trace.get("metadata") or {}).get("generation_backend") or {}
    model = backend.get("model_path") or backend.get("model")
    return str(model) if model is not None else None


def _trace_usage(trace: dict[str, Any]) -> dict[str, int]:
    """Sum per-turn usage, avoiding backend-global counters in concurrent runs."""
    totals = {field: 0 for field in _TOKEN_FIELDS}
    generated = (trace.get("metadata") or {}).get("generated_turns") or []
    for turn in generated:
        usage = turn.get("usage") or {} if isinstance(turn, dict) else {}
        for field in _TOKEN_FIELDS:
            value = usage.get(field)
            if isinstance(value, (int, float)):
                totals[field] += int(value)
    return totals


def _add_usage(total: dict[str, int], addition: dict[str, int]) -> None:
    for field in _TOKEN_FIELDS:
        total[field] += addition[field]


def _clean_messages(
    messages: Any,
    *,
    require_final_assistant: bool = True,
) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    if not isinstance(messages[-1], dict) or messages[-1].get("role") != "exit":
        raise ValueError("final raw message must have role 'exit'")
    cleaned = messages[:-1]
    if not cleaned:
        raise ValueError("trace has no messages before exit")
    if require_final_assistant and (
        not isinstance(cleaned[-1], dict) or cleaned[-1].get("role") != "assistant"
    ):
        raise ValueError("final message before exit must have role 'assistant'")
    allowed = {"system", "user", "assistant"}
    for message in cleaned:
        if not isinstance(message, dict) or message.get("role") not in allowed:
            raise ValueError(f"unknown message role: {message!r}")
        if not isinstance(message.get("content"), str):
            raise ValueError("every message content must be text")
    return cleaned


def episode_samples(
    trace: dict[str, Any],
    *,
    trace_path: Path,
    provenance_ref: str | None,
    agreement_score: float,
) -> list[dict[str, Any]]:
    """Create one slime row per assistant turn from an already accepted trace."""
    cleaned = _clean_messages(trace.get("messages"), require_final_assistant=False)
    uid = str(trace["uid"])
    output: list[dict[str, Any]] = []
    for assistant_turn, (target, messages) in enumerate(
        per_turn_training_messages(cleaned), start=1
    ):
        output.append(
            {
                "messages": messages,
                "metadata": {
                    "uid": uid,
                    "assistant_turn": assistant_turn,
                    "score": agreement_score,
                    "steps": trace.get("steps"),
                    "status": trace.get("status"),
                    "teacher_model": _teacher_model(trace),
                    "teacher_submission": trace.get("submission"),
                    "gemini_answer": trace.get("ground_truth"),
                    "target_has_reasoning": bool(
                        (target.get("reasoning_content") or "").strip()
                    ),
                    "trace_file": str(trace_path.resolve()),
                    "provenance_ref": provenance_ref,
                },
            }
        )
    return output


def _write_rows(rows: Iterable[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        if output.suffix == ".jsonl":
            with temporary.open("w", encoding="utf-8") as file:
                for row in rows:
                    file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                    file.write("\n")
        else:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as error:
                raise RuntimeError(
                    "pyarrow is required for .parquet output; use .jsonl or install pyarrow"
                ) from error
            pq.write_table(pa.Table.from_pylist(list(rows)), temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def build(args: argparse.Namespace) -> dict[str, Any]:
    uid_re = re.compile(args.uid_pattern)
    # Trace files are <uid>.json, or <uid>.s<i>.json for multi-attempt runs.
    trace_stem_re = re.compile(rf"(?P<uid>{args.uid_pattern})(?:\.s(?P<sample>\d+))?")
    dataset_rows = read_csv_rows(args.dataset_csv)
    dataset_by_uid = _unique_rows_by_uid(dataset_rows, args.dataset_csv, uid_re)
    benchmark_rows: list[dict[str, str]] = []
    benchmark_sources = []
    for path in args.benchmark_csv:
        rows = read_csv_rows(path)
        benchmark_rows.extend(rows)
        benchmark_sources.append(
            {"file": str(path.resolve()), "rows": len(rows), "sha256": sha256_of_file(path)}
        )
    benchmark_questions = [row.get("question") or "" for row in benchmark_rows]
    benchmark_normalized = {normalize_question(question) for question in benchmark_questions}

    provenance_path = args.provenance_jsonl or _auto_provenance_path(args.dataset_csv)
    provenance_refs = _load_provenance_refs(provenance_path)

    grouped: dict[str, list[tuple[int, Path]]] = {}
    for path in args.trace_dir.glob("*.json"):
        match = trace_stem_re.fullmatch(path.stem)
        if match is None or not path.is_file():
            continue  # not a trace (e.g. the runner's summary.json)
        sample = match.group("sample")
        grouped.setdefault(match.group("uid"), []).append(
            (-1 if sample is None else int(sample), path)
        )
    trace_groups = {
        uid: [path for _, path in sorted(attempts)] for uid, attempts in grouped.items()
    }
    expected_uids = set(dataset_by_uid)
    missing_uids = sorted(expected_uids - set(trace_groups))
    extra_uids = sorted(set(trace_groups) - expected_uids)
    if extra_uids:
        raise ValueError(
            f"Trace directory contains {len(extra_uids)} unknown UIDs: {extra_uids[:10]}"
        )
    if missing_uids and not args.allow_missing_traces:
        raise ValueError(
            f"Trace directory is incomplete: {len(missing_uids)} of {len(expected_uids)} "
            f"UIDs are missing; first missing: {missing_uids[:10]}. Resume generation or "
            "pass --allow-missing-traces for an audit build."
        )

    rejections: Counter[str] = Counter()
    decisions: list[dict[str, Any]] = []
    retained_question_by_uid: dict[str, str] = {}
    retained_uids: list[str] = []
    output_rows: list[dict[str, Any]] = []
    raw_usage = {field: 0 for field in _TOKEN_FIELDS}
    retained_usage = {field: 0 for field in _TOKEN_FIELDS}

    episode_paths: list[tuple[str, Path | None]] = [
        (uid, path)
        for uid in sorted(expected_uids)
        for path in (trace_groups.get(uid) or [None])
    ]
    for uid, path in episode_paths:
        if path is None:
            rejections["missing_trace"] += 1
            decisions.append({"uid": uid, "accepted": False, "rejection": "missing_trace"})
            continue

        row = dataset_by_uid[uid]
        decision: dict[str, Any] = {"uid": uid, "trace_file": path.name}
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            reason = "malformed_trace"
            decision["detail"] = str(error)
            trace = None
        if trace is not None and not isinstance(trace, dict):
            reason = "malformed_trace"
            decision["detail"] = "top-level trace JSON must be an object"
            trace = None

        if trace is not None:
            usage = _trace_usage(trace)
            _add_usage(raw_usage, usage)
            label = str(row.get("answer") or "")
            submission = str(trace.get("submission") or "")
            decision.update(
                {
                    "gemini_answer": label,
                    "teacher_submission": submission,
                    "agreement_score": None,
                }
            )
            try:
                agreement = score_answer(label, submission, tolerance=0.01)
            except (TypeError, ValueError):
                agreement = 0.0
            decision["agreement_score"] = agreement

            reason = None
            trace_uid = trace.get("uid")
            if trace_uid != uid:
                reason = "uid_mismatch"
            elif normalize_question(str(trace.get("question") or "")) != normalize_question(
                row.get("question") or ""
            ):
                reason = "question_mismatch"
            elif str(trace.get("ground_truth") or "") != label:
                reason = "label_mismatch"

            if reason is None and args.episode_policy == "filtered":
                if trace.get("status") != "submitted":
                    reason = "not_submitted"
                elif agreement != 1.0:
                    reason = "answer_disagreement"
                elif not isinstance(trace.get("steps"), int) or trace["steps"] > args.max_steps:
                    reason = "too_many_steps"

            if reason is None:
                try:
                    cleaned = _clean_messages(
                        trace.get("messages"),
                        require_final_assistant=args.episode_policy == "filtered",
                    )
                except ValueError as error:
                    reason = "invalid_message_structure"
                    decision["detail"] = str(error)
                else:
                    assistant_messages = [
                        message for message in cleaned if message["role"] == "assistant"
                    ]
                    if not assistant_messages:
                        reason = "no_assistant_turns"
                    elif args.episode_policy == "filtered":
                        if any(
                            not _valid_bash_content(message.get("content"))
                            for message in assistant_messages
                        ):
                            reason = "invalid_action_format"
                        elif any(
                            not (message.get("reasoning_content") or "").strip()
                            and not _valid_direct_bash_content(message.get("content"))
                            for message in assistant_messages
                        ):
                            reason = "unreasoned_non_direct_turn"

                    if reason is None and args.episode_policy == "filtered":
                        duplicate_ratio = _duplicate_command_ratio(trace)
                        decision["duplicate_command_ratio"] = duplicate_ratio
                        if duplicate_ratio > args.max_duplicate_command_ratio:
                            reason = "duplicate_commands"

            question = row.get("question") or ""
            if reason is None and normalize_question(question) in benchmark_normalized:
                reason = "benchmark_question_exact"
            if reason is None:
                similarity, match = max_question_similarity(question, benchmark_questions)
                decision["benchmark_max_jaccard"] = similarity
                if similarity >= args.jaccard_threshold:
                    reason = "benchmark_question_similar"
                    decision["similar_to"] = match
            if (
                reason is None
                and not args.skip_answer_collision
                and answer_collision(_source_files(row), label, benchmark_rows)
            ):
                reason = "benchmark_source_answer_collision"
            # Attempts of the same uid share one question; dedup only across uids.
            other_questions = [
                item for other_uid, item in retained_question_by_uid.items() if other_uid != uid
            ]
            if (
                reason is None
                and args.episode_policy == "filtered"
                and normalize_question(question) in {
                normalize_question(item) for item in other_questions
                }
            ):
                reason = "synthetic_question_exact"
            if reason is None and args.episode_policy == "filtered":
                similarity, match = max_question_similarity(question, other_questions)
                decision["synthetic_max_jaccard"] = similarity
                if similarity >= args.jaccard_threshold:
                    reason = "synthetic_question_similar"
                    decision["similar_to"] = match

            if reason is None:
                samples = episode_samples(
                    trace,
                    trace_path=path,
                    provenance_ref=provenance_refs.get(uid),
                    agreement_score=agreement,
                )
                output_rows.extend(samples)
                retained_question_by_uid[uid] = question
                retained_uids.append(uid)
                _add_usage(retained_usage, usage)
                decision.update({"accepted": True, "samples": len(samples)})
            else:
                rejections[reason] += 1
                decision.update({"accepted": False, "rejection": reason})
        else:
            rejections[reason] += 1
            decision.update({"accepted": False, "rejection": reason})
        decisions.append(decision)

    if not output_rows:
        raise ValueError("No traces passed the SFT filters; no output was written")

    # Final fail-closed assertions before writing training data.
    assert all(uid_re.fullmatch(uid) for uid in retained_uids)
    assert not (
        {normalize_question(item) for item in retained_question_by_uid.values()}
        & benchmark_normalized
    )
    assert all(row["messages"][-1]["role"] == "assistant" for row in output_rows)
    assert all(row["messages"][-1].get("step_loss_mask") == 1 for row in output_rows)

    reasoning_samples = sum(
        bool(row["metadata"]["target_has_reasoning"]) for row in output_rows
    )
    direct_command_samples = len(output_rows) - reasoning_samples

    _write_rows(output_rows, args.output)
    manifest_path = args.manifest or args.output.with_name(f"{args.output.stem}_manifest.json")
    manifest = {
        "format_version": 1,
        "episode_policy": args.episode_policy,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "trace_dir": str(args.trace_dir.resolve()),
            "dataset_csv": str(args.dataset_csv.resolve()),
            "dataset_sha256": sha256_of_file(args.dataset_csv),
            "provenance_jsonl": str(provenance_path.resolve()) if provenance_path else None,
            "provenance_sha256": sha256_of_file(provenance_path) if provenance_path else None,
            "benchmark_csvs": benchmark_sources,
        },
        "filters": {
            "quality_filters_enabled": args.episode_policy == "filtered",
            "uid_pattern": args.uid_pattern,
            "benchmark_answer_collision_enabled": not args.skip_answer_collision,
            "max_steps": args.max_steps,
            "max_duplicate_command_ratio": args.max_duplicate_command_ratio,
            "question_char_5gram_jaccard_threshold": args.jaccard_threshold,
            "answer_tolerance": 0.01,
            "require_submitted": args.episode_policy == "filtered",
            "require_answer_agreement": args.episode_policy == "filtered",
            "allow_unreasoned_direct_command_turns": True,
            "reject_unreasoned_turns_with_surrounding_prose": (
                args.episode_policy == "filtered"
            ),
            "require_one_trailing_fenced_bash_block": True,
        },
        "counts": {
            "expected_traces": len(expected_uids),
            "found_traces": len(trace_groups),
            "found_trace_files": sum(len(paths) for paths in trace_groups.values()),
            "missing_traces": len(missing_uids),
            "retained_episodes": len(retained_uids),
            "retained_uids": len(set(retained_uids)),
            "sft_samples": len(output_rows),
            "reasoning_sft_samples": reasoning_samples,
            "direct_command_sft_samples": direct_command_samples,
        },
        "missing_uids": missing_uids,
        "rejections": dict(sorted(rejections.items())),
        "usage": {"all_found_traces": raw_usage, "retained_traces": retained_usage},
        "output": {
            "file": str(args.output.resolve()),
            "rows": len(output_rows),
            "sha256": sha256_of_file(args.output),
            "input_key": "messages",
        },
        "trace_decisions": decisions,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    counts = manifest["counts"]
    print(
        f"retained {counts['retained_episodes']} episodes as {counts['sft_samples']} SFT samples "
        f"-> {manifest['output']['file']}"
    )
    print(f"rejections: {json.dumps(manifest['rejections'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
