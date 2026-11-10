"""Run SpreadsheetBench 2 tasks with a selected generation backend.

Companion to ``run_full_corpus_inference.py`` (whose backend construction and
pass@k helpers it imports): same per-task result JSONs with full message
traces, resume by skipping existing result files, retry passes, pass@k
sampling, and a summary.json — but tasks come from the SpreadsheetBench 2
``dataset.json`` files, the sandbox holds the input workbook, and grading is
the official cell-level workbook comparison (LibreOffice recalculation +
``third_party/SpreadsheetBench-2/evaluation/evaluation.py``).

Result payloads keep the OfficeQA field names (``score`` is the strict
official accuracy, so pass@k means "task fully correct") and add a
``metrics`` dict with ``regression_accuracy``, ``modification_accuracy`` (the
dense signal: fraction of target cells correct), and capped per-cell error
messages. The graded output workbook of every attempt is kept under
``<output-dir>/outputs/``.

Usage:
    uv run scripts/run_spreadsheetbench_eval.py \
        --data-root data/spreadsheetbench_v2/spreadsheetbench-v2 \
        --generation-backend openai-server --model <served model> \
        --output-dir results/ssb2-<tag>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from officeqa_rl.agent import OfficeQAAgent
from officeqa_rl.spreadsheetbench import (
    ALL_CATEGORIES,
    CELL_GRADED_CATEGORIES,
    SpreadsheetBenchEnvironment,
    SpreadsheetBenchTask,
    build_spreadsheet_system_prompt,
    load_spreadsheetbench_tasks,
)
from officeqa_rl.spreadsheetbench_scoring import (
    compare_output,
    comparison_flags,
    recalc_with_libreoffice,
)
from officeqa_rl.token_usage import add_usage, per_task_means
from officeqa_rl.types import AgentConfig, EnvironmentConfig

from run_full_corpus_inference import (
    _classify_failure,
    _pass_at_k,
    build_generation_backend,
    default_eval_concurrency,
)

logger = logging.getLogger("officeqa_rl.spreadsheetbench")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/spreadsheetbench_v2/spreadsheetbench-v2"),
        help="Directory holding the per-category dataset.json folders",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(CELL_GRADED_CATEGORIES),
        choices=list(ALL_CATEGORIES),
        help="Benchmark categories to run (Visualization has no local grading; "
        "its outputs are saved but scored 0)",
    )
    parser.add_argument(
        "--uid",
        help="Run exactly this task (e.g. Template__01_01); omit for all rows",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--failure-retries", type=int, default=2)
    parser.add_argument("--failure-retry-delay-sec", type=float, default=60.0)
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=1,
        help="Independent attempts per task for pass@k (needs --temperature > 0)",
    )

    parser.add_argument("--step-limit", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        # Offline evals run at temperature 1.0 by standing protocol: greedy
        # decoding inflates per-turn length on OPD-trained checkpoints, and
        # makes pass@k degenerate because every sample repeats one trajectory.
    )
    parser.add_argument("--action-block-language", default="bash")

    parser.add_argument("--sandbox-image", default="spreadsheetbench-v2-sandbox")
    # Workbook edits routinely take longer than corpus greps: bigger default.
    parser.add_argument("--command-timeout-sec", type=int, default=120)
    parser.add_argument("--max-observation-chars", type=int, default=12_000)
    parser.add_argument(
        "--skip-recalc",
        action="store_true",
        help="Skip the LibreOffice cached-value refresh before comparison "
        "(only sound when outputs already carry computed values)",
    )
    parser.add_argument("--recalc-timeout-sec", type=float, default=600.0)

    # Generation-backend flags mirror run_full_corpus_inference.py so its
    # build_generation_backend(args) can be reused unchanged.
    parser.add_argument(
        "--generation-backend",
        choices=["vllm", "gemini", "kimi", "openai-server", "vertex"],
        default="vllm",
    )
    parser.add_argument("--model", "--model-path", dest="model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--safetensors-load-strategy",
        choices=["lazy", "eager", "prefetch", "torchao"],
    )
    parser.add_argument(
        "--gdn-prefill-backend", choices=["flashinfer", "triton", "cutedsl"]
    )
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--serialize-generation-calls",
        "--serialize-vllm-calls",
        dest="serialize_generation_calls",
        action="store_true",
    )
    parser.add_argument("--gemini-api-version", default="v1beta")
    parser.add_argument("--gemini-max-retries", type=int, default=3)
    parser.add_argument("--gemini-quota-retries", type=int, default=10)
    parser.add_argument("--gemini-thinking-budget", type=int, default=0)
    parser.add_argument("--gemini-include-thoughts", action="store_true")
    parser.add_argument("--kimi-base-url", default="https://api.moonshot.ai/v1")
    parser.add_argument("--kimi-max-retries", type=int, default=3)
    parser.add_argument("--kimi-quota-retries", type=int, default=10)
    parser.add_argument(
        "--kimi-reasoning-effort",
        choices=["low", "high", "max"],
        default="max",
    )
    parser.add_argument("--openai-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--openai-max-retries", type=int, default=3)
    parser.add_argument("--openai-quota-retries", type=int, default=10)
    parser.add_argument("--openai-request-timeout-sec", type=float, default=600.0)
    parser.add_argument("--vertex-url")
    parser.add_argument("--vertex-project")
    parser.add_argument("--vertex-location")
    parser.add_argument("--vertex-endpoint-id")
    parser.add_argument(
        "--vertex-protocol", choices=["sglang", "vllm"], default="sglang"
    )
    parser.add_argument("--vertex-tokenizer", default="/mnt/Qwen3.5-4B")
    parser.add_argument("--vertex-instruction")
    parser.add_argument("--vertex-max-retries", type=int, default=3)
    parser.add_argument("--vertex-quota-retries", type=int, default=10)
    parser.add_argument("--vertex-request-timeout-sec", type=float, default=600.0)

    args = parser.parse_args(argv)
    if args.concurrency is None:
        args.concurrency = default_eval_concurrency(args.generation_backend, args.model)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.data_parallel_size < 1:
        parser.error("--data-parallel-size must be at least 1")
    if args.samples_per_task < 1:
        parser.error("--samples-per-task must be at least 1")
    if args.samples_per_task > 1 and args.temperature <= 0.0:
        parser.error(
            "--samples-per-task > 1 needs --temperature > 0: at temperature 0 "
            "every attempt is identical and pass@k equals pass@1"
        )
    if args.generation_backend == "vertex" and not (
        args.vertex_url
        or (args.vertex_project and args.vertex_location and args.vertex_endpoint_id)
    ):
        parser.error(
            "--generation-backend vertex needs --vertex-url or all of "
            "--vertex-project/--vertex-location/--vertex-endpoint-id"
        )
    if args.num_shards < 1:
        parser.error("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    return args


def _result_path(output_dir: Path, uid: str, sample: int | None = None) -> Path:
    name = f"{uid}.json" if sample is None else f"{uid}.s{sample}.json"
    return output_dir / name


def _has_result(output_dir: Path, uid: str, sample: int | None = None) -> bool:
    path = _result_path(output_dir, uid, sample)
    return path.is_file() and path.stat().st_size > 0


def _attempt_key(uid: str, sample: int | None) -> str:
    return uid if sample is None else f"{uid}.s{sample}"


async def run_task(
    args: argparse.Namespace,
    backend: Any,
    task: SpreadsheetBenchTask,
    run_id: str,
    output_copy_path: Path,
) -> dict[str, Any]:
    """Run one task, keep its output workbook, and grade it officially."""
    sampling_params = (
        {} if args.generation_backend == "kimi" else {"temperature": args.temperature}
    )
    episode_backend = (
        backend.for_episode(run_id) if hasattr(backend, "for_episode") else backend
    )
    agent = OfficeQAAgent(
        episode_backend,
        AgentConfig(
            step_limit=args.step_limit,
            action_block_language=args.action_block_language,
        ),
        sampling_params=sampling_params,
        system_prompt=build_spreadsheet_system_prompt(args.action_block_language),
    )
    environment_config = EnvironmentConfig(
        corpus_dir=args.data_root,
        pdf_dir=None,
        workspace_root=args.workspace_root
        or args.output_dir / "workspaces",
        sandbox_image=args.sandbox_image,
        command_timeout_sec=args.command_timeout_sec,
        max_observation_chars=args.max_observation_chars,
        sandbox_runtime="docker",
    )

    async with SpreadsheetBenchEnvironment(
        environment_config,
        task,
        run_id=run_id,
    ) as environment:
        result = await agent.run(environment)
        # Grade whatever workbook exists, submitted or not, like the official
        # evaluator does with its output directory (missing file scores 0).
        output_produced = environment.output_host_path.is_file()
        if output_produced:
            output_copy_path.parent.mkdir(parents=True, exist_ok=True)
            # A failed earlier attempt may have left a stale (possibly
            # root-owned, pre-fix) copy that copy2 cannot overwrite in place.
            output_copy_path.unlink(missing_ok=True)
            shutil.copy2(environment.output_host_path, output_copy_path)

    recalc_error = ""
    if output_produced and not args.skip_recalc:
        try:
            await recalc_with_libreoffice(
                output_copy_path,
                sandbox_image=args.sandbox_image,
                timeout_sec=args.recalc_timeout_sec,
            )
        except Exception as error:
            # Grade anyway: formulas without cached values then read as None,
            # which shows up as mismatches, and the recorded error explains it.
            recalc_error = str(error)

    if task.category == "Visualization":
        metrics = {
            "accuracy": 0.0,
            "error": "Visualization tasks need the official VLM checklist "
            "evaluator (Windows COM chart export); output saved but unscored",
        }
    else:
        with_font_color, with_formula = comparison_flags(
            task.category, str(task.input_path)
        )
        metrics = await asyncio.to_thread(
            compare_output,
            task.input_path,
            task.golden_path,
            output_copy_path,
            task.answer_position,
            with_font_color=with_font_color,
            with_formula=with_formula,
        )
    if recalc_error:
        metrics["recalc_error"] = recalc_error

    return {
        "uid": task.uid,
        "category": task.category,
        "task_id": task.task_id,
        "question": task.instruction,
        "ground_truth": str(task.golden_path),
        "submission": result.submission,
        "status": result.status,
        "steps": result.steps,
        "score": metrics.get("accuracy", 0.0),
        "token_usage": result.metadata.get("token_usage"),
        "metrics": metrics,
        "output_file": str(output_copy_path) if output_produced else None,
        "messages": result.messages,
        "metadata": {
            **result.metadata,
            "generation_backend": {
                "class": type(backend).__name__,
                "kind": args.generation_backend,
                "model_path": backend.model_path,
                "stats": backend.get_stats(),
            },
        },
    }


async def run_full_eval(args: argparse.Namespace) -> dict[str, Any]:
    tasks = load_spreadsheetbench_tasks(args.data_root, args.categories)
    if args.num_shards > 1:
        tasks = [
            task
            for position, task in enumerate(tasks)
            if position % args.num_shards == args.shard_index
        ]
        logger.info(
            "selected deterministic task shard %d/%d (%d tasks)",
            args.shard_index,
            args.num_shards,
            len(tasks),
        )
    if args.uid is not None:
        tasks = [task for task in tasks if task.uid == args.uid]
        if not tasks:
            raise SystemExit(f"SpreadsheetBench UID not found: {args.uid}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = output_dir / "outputs"
    # GCS-FUSE turns one ``Path.is_file`` call per absent attempt into a
    # network metadata request. A fresh pass@8 SSB2 run has 2,376 absent
    # attempts, which can spend minutes here before generation starts. List
    # the directory once and keep the set current as this process writes.
    existing_result_names = {
        entry.name
        for entry in os.scandir(output_dir)
        if entry.name != "summary.json"
        and entry.name.endswith(".json")
        and entry.is_file()
        and entry.stat().st_size > 0
    }

    def has_result(uid: str, sample: int | None) -> bool:
        return _result_path(output_dir, uid, sample).name in existing_result_names

    samples = [None] if args.samples_per_task == 1 else list(range(args.samples_per_task))
    work = [(task, sample) for task in tasks for sample in samples]
    pending = [
        (task, sample)
        for task, sample in work
        if not has_result(task.uid, sample)
    ]
    resumed = len(work) - len(pending)
    logger.info(
        "%d tasks x %d attempts: %d already have results, %d to run (concurrency %d)",
        len(tasks),
        len(samples),
        resumed,
        len(pending),
        args.concurrency,
    )

    backend = build_generation_backend(args)
    semaphore = asyncio.Semaphore(args.concurrency)
    failures: dict[str, str] = {}
    finished = 0
    run_scores: list[float] = []
    started_at = time.monotonic()

    def log_progress() -> None:
        processed = finished + len(failures)
        elapsed = time.monotonic() - started_at
        remaining = len(pending) - processed
        eta_text = "n/a"
        if processed and remaining:
            eta_text = f"~{elapsed / processed * remaining / 60:.0f}m"
        elif not remaining:
            eta_text = "done"
        mean_text = (
            f"{sum(run_scores) / len(run_scores):.3f} over {len(run_scores)} scored"
            if run_scores
            else "n/a"
        )
        logger.info(
            "progress: %d/%d attempts this run (%d failed) | mean accuracy %s | "
            "elapsed %.0fm, eta %s",
            processed,
            len(pending),
            len(failures),
            mean_text,
            elapsed / 60,
            eta_text,
        )

    async def run_one(task: SpreadsheetBenchTask, sample: int | None) -> None:
        nonlocal finished
        async with semaphore:
            key = _attempt_key(task.uid, sample)
            run_id = (
                f"{args.run_id}-{key}"
                if args.run_id
                else f"{key}-{uuid.uuid4().hex[:12]}"
            )
            output_copy = outputs_dir / f"{key}_output.xlsx"
            logger.info("[%s] start", key)
            try:
                payload = await run_task(args, backend, task, run_id, output_copy)
            except Exception as error:
                failures[key] = f"{type(error).__name__}: {error}"
                logger.error("[%s] FAILED: %s", key, failures[key])
                log_progress()
                return
            if payload["status"].startswith("error:"):
                detail = (
                    payload["messages"][-1]["content"]
                    if payload["messages"]
                    else payload["status"]
                )
                failures[key] = detail
                logger.error("[%s] FAILED: %s", key, detail)
                log_progress()
                return
            if sample is not None:
                payload["sample"] = sample
            result_path = _result_path(output_dir, task.uid, sample)
            result_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            existing_result_names.add(result_path.name)
            failures.pop(key, None)
            finished += 1
            run_scores.append(payload["score"])
            logger.info(
                "[%s] %s accuracy=%.0f%% modification=%.1f%% steps=%d",
                key,
                payload["status"],
                100 * payload["score"],
                100 * payload["metrics"].get("modification_accuracy", 0.0),
                payload["steps"],
            )
            log_progress()

    to_run = pending
    for retry_pass in range(args.failure_retries + 1):
        if not to_run:
            break
        if retry_pass:
            logger.info(
                "retry pass %d/%d: %d failed attempts, pausing %.0fs first",
                retry_pass,
                args.failure_retries,
                len(to_run),
                args.failure_retry_delay_sec,
            )
            if args.failure_retry_delay_sec > 0:
                await asyncio.sleep(args.failure_retry_delay_sec)
        await asyncio.gather(*(run_one(task, sample) for task, sample in to_run))

        def retryable(key: str) -> bool:
            # An over-length request cannot succeed when replayed.
            return _classify_failure(failures[key]) != "context_length"

        to_run = [
            (task, sample)
            for task, sample in pending
            if _attempt_key(task.uid, sample) in failures
            and retryable(_attempt_key(task.uid, sample))
        ]

    # Aggregate whatever result files exist (this run + resumed).
    per_task: dict[str, dict[str, Any]] = {}
    token_usage = None
    category_of = {task.uid: task.category for task in tasks}
    for task in tasks:
        payloads = [
            json.loads(
                _result_path(output_dir, task.uid, sample).read_text(encoding="utf-8")
            )
            for sample in samples
            if has_result(task.uid, sample)
        ]
        if not payloads:
            continue
        for payload in payloads:
            episode = payload.get("token_usage")
            if isinstance(episode, dict):
                token_usage = add_usage(token_usage, episode)
        per_task[task.uid] = {
            "category": task.category,
            "sample_scores": [payload["score"] for payload in payloads],
            "modification_accuracies": [
                payload["metrics"].get("modification_accuracy", 0.0)
                for payload in payloads
            ],
            "regression_accuracies": [
                payload["metrics"].get("regression_accuracy", 0.0)
                for payload in payloads
            ],
            "statuses": [payload["status"] for payload in payloads],
            "steps": [payload["steps"] for payload in payloads],
            "n_attempts": len(payloads),
            "n_success": sum(1 for payload in payloads if payload["score"] >= 0.5),
        }

    if token_usage is not None:
        # Per task, not per attempt: a pass@k run pays for every attempt.
        token_usage.update(per_task_means(token_usage, len(per_task)))

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    all_scores = [s for entry in per_task.values() for s in entry["sample_scores"]]
    all_mod = [
        m for entry in per_task.values() for m in entry["modification_accuracies"]
    ]
    all_statuses = [s for entry in per_task.values() for s in entry["statuses"]]

    by_category: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for uid, entry in per_task.items():
        grouped[category_of[uid]].append(entry)
    for category, entries in grouped.items():
        by_category[category] = {
            "tasks": len(entries),
            "accuracy": mean([s for e in entries for s in e["sample_scores"]]),
            "modification_accuracy": mean(
                [m for e in entries for m in e["modification_accuracies"]]
            ),
        }

    pass_at_k = None
    if args.samples_per_task > 1:
        pass_at_k = {}
        for k in range(1, args.samples_per_task + 1):
            estimates = [
                _pass_at_k(entry["n_attempts"], entry["n_success"], k)
                for entry in per_task.values()
                if entry["n_attempts"] >= k
            ]
            pass_at_k[str(k)] = mean(estimates)

    summary = {
        "benchmark": "spreadsheetbench-v2",
        "data_root": str(args.data_root),
        "categories": list(args.categories),
        "generation_backend": args.generation_backend,
        "model": args.model,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "total_rows": len(tasks),
        "samples_per_task": args.samples_per_task,
        "completed": len(per_task),
        "resumed": resumed,
        "failed": failures,
        "failure_counts": dict(
            Counter(_classify_failure(detail) for detail in failures.values())
        ),
        "accuracy": mean(all_scores),
        "modification_accuracy": mean(all_mod),
        "scored": len(all_scores),
        "by_category": by_category,
        "pass_at_k": pass_at_k,
        "status_counts": dict(Counter(all_statuses)),
        "token_usage": token_usage,
        "results": per_task,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "completed %d/%d (failed %d) | accuracy %s | modification %s | %s",
        summary["completed"],
        summary["total_rows"],
        len(failures),
        f"{summary['accuracy']:.3f}" if summary["accuracy"] is not None else "n/a",
        f"{summary['modification_accuracy']:.3f}"
        if summary["modification_accuracy"] is not None
        else "n/a",
        summary["status_counts"],
    )
    for category, stats in by_category.items():
        logger.info(
            "  %s: %d tasks, accuracy %.3f, modification %.3f",
            category,
            stats["tasks"],
            stats["accuracy"] or 0.0,
            stats["modification_accuracy"] or 0.0,
        )
    if failures:
        logger.warning(
            "%d attempts are MISSING from the score: %s",
            len(failures),
            ", ".join(sorted(failures)),
        )
    logger.info("summary written to %s", output_dir / "summary.json")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    summary = asyncio.run(run_full_eval(args))
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "results"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
