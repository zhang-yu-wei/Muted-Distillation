#!/usr/bin/env python3
"""Run an eval-ready SpreadsheetBench v1 or Verified release.

For each instruction the agent sees case 1 and writes ``solution.py``.  That
single program is then replayed in clean sandboxes on every workbook case,
matching v1's multi-workbook protocol.  Results are resumable and include the
official hard/soft restriction scores, per-cell diagnostics, full traces, and
pass@k estimates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from officeqa_rl.agent import OfficeQAAgent
from officeqa_rl.spreadsheetbench_scoring import recalc_with_libreoffice
from officeqa_rl.spreadsheetbench_v1 import (
    SpreadsheetBenchV1Environment,
    SpreadsheetBenchV1Task,
    build_spreadsheet_v1_system_prompt,
    load_spreadsheetbench_v1_tasks,
)
from officeqa_rl.spreadsheetbench_v1_scoring import (
    aggregate_v1_metrics,
    compare_v1_output,
    replay_solution,
)
from officeqa_rl.token_usage import add_usage, per_task_means
from officeqa_rl.types import AgentConfig, EnvironmentConfig

from run_full_corpus_inference import (
    _classify_failure,
    _pass_at_k,
    build_generation_backend,
    default_eval_concurrency,
    resolve_enable_thinking,
)


logger = logging.getLogger("officeqa_rl.spreadsheetbench_v1")


def _raise_if_engine_failed(backend: str, error: BaseException | str) -> None:
    """A dead local engine cannot be repaired by retrying every trajectory."""
    detail = str(error).lower()
    if backend == "vllm" and any(marker in detail for marker in (
        "enginedeaderror", "enginecore encountered an issue",
        "enginecore encountered a fatal error", "engine core process died",
    )):
        raise RuntimeError(
            "Local inference engine died; stop this evaluator and resume the "
            "same checkpoint with fresh workers. Saved attempts are retained."
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/spreadsheetbench_v1/verified_400"),
        help="Extracted v1 release containing dataset.json and spreadsheet/",
    )
    parser.add_argument("--uid", help="Run one instruction, e.g. SSB1__0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--failure-retries", type=int, default=2)
    parser.add_argument("--failure-retry-delay-sec", type=float, default=60.0)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the ordered task list into this many deterministic shards",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based task shard to run (task position modulo --num-shards)",
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
    parser.add_argument("--sandbox-image", default="spreadsheetbench-v1-sandbox")
    parser.add_argument("--command-timeout-sec", type=int, default=120)
    parser.add_argument("--max-observation-chars", type=int, default=12_000)
    parser.add_argument("--replay-timeout-sec", type=float, default=600.0)
    parser.add_argument("--recalc-timeout-sec", type=float, default=600.0)
    parser.add_argument(
        "--skip-recalc",
        action="store_true",
        help="Skip LibreOffice formula-cache refresh before grading",
    )

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
        help="vLLM safetensors loading strategy; prefetch is best for GCS FUSE",
    )
    parser.add_argument(
        "--gdn-prefill-backend",
        choices=["flashinfer", "triton", "cutedsl"],
        help="Override vLLM's Qwen GDN prefill backend",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Forward enable_thinking to the tokenizer chat template. Local "
            "Gemma-4-12B evals default this to true; Vertex Gemma launchers "
            "must request it explicitly; other models leave it unset."
        ),
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
        "--kimi-reasoning-effort", choices=["low", "high", "max"], default="max"
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
    if args.generation_backend == "vllm":
        args.enable_thinking = resolve_enable_thinking(
            args.model, args.enable_thinking
        )
    if args.concurrency is None:
        args.concurrency = default_eval_concurrency(args.generation_backend, args.model)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.data_parallel_size < 1:
        parser.error("--data-parallel-size must be at least 1")
    if args.samples_per_task < 1:
        parser.error("--samples-per-task must be at least 1")
    if args.num_shards < 1:
        parser.error("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.samples_per_task > 1 and args.temperature <= 0.0:
        parser.error("--samples-per-task > 1 requires --temperature > 0")
    if args.generation_backend == "vertex" and not (
        args.vertex_url
        or (args.vertex_project and args.vertex_location and args.vertex_endpoint_id)
    ):
        parser.error(
            "--generation-backend vertex needs --vertex-url or all of "
            "--vertex-project/--vertex-location/--vertex-endpoint-id"
        )
    return args


def _result_path(output_dir: Path, uid: str, sample: int | None) -> Path:
    suffix = "" if sample is None else f".s{sample}"
    return output_dir / f"{uid}{suffix}.json"


def _has_result(output_dir: Path, uid: str, sample: int | None) -> bool:
    path = _result_path(output_dir, uid, sample)
    return path.is_file() and path.stat().st_size > 0


def _existing_result_names(output_dir: Path) -> set[str]:
    """List resumable results with one directory scan.

    Result directories may live on GCS-FUSE. Probing every expected attempt
    separately turns a pass@8 resume into thousands of serialized metadata
    requests, so only stat the entries that actually exist.
    """
    names: set[str] = set()
    for path in output_dir.glob("*.json"):
        if path.name == "summary.json":
            continue
        try:
            if path.stat().st_size > 0:
                names.add(path.name)
        except FileNotFoundError:
            # An interrupted writer may disappear during the scan; it remains
            # pending and will be regenerated safely.
            continue
    return names


def _attempt_key(uid: str, sample: int | None) -> str:
    return uid if sample is None else f"{uid}.s{sample}"


async def run_task(
    args: argparse.Namespace,
    backend: Any,
    task: SpreadsheetBenchV1Task,
    run_id: str,
    artifact_dir: Path,
    solution_copy_path: Path,
) -> dict[str, Any]:
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
        system_prompt=build_spreadsheet_v1_system_prompt(args.action_block_language),
    )
    environment_config = EnvironmentConfig(
        corpus_dir=args.data_root,
        pdf_dir=None,
        workspace_root=args.workspace_root or args.output_dir / "workspaces",
        sandbox_image=args.sandbox_image,
        command_timeout_sec=args.command_timeout_sec,
        max_observation_chars=args.max_observation_chars,
        sandbox_runtime="docker",
    )

    solution_copy_path.unlink(missing_ok=True)
    async with SpreadsheetBenchV1Environment(
        environment_config, task, run_id=run_id
    ) as environment:
        result = await agent.run(environment)
        solution_produced = environment.solution_host_path.is_file()
        if solution_produced:
            solution_copy_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(environment.solution_host_path, solution_copy_path)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    case_metrics: list[dict[str, Any]] = []
    for case in task.cases:
        output_path = artifact_dir / case.output_name
        output_path.unlink(missing_ok=True)
        replay = await replay_solution(
            solution_copy_path,
            case,
            output_path,
            visible_input_name=task.input_name,
            visible_output_name=task.output_name,
            sandbox_image=args.sandbox_image,
            timeout_sec=args.replay_timeout_sec,
        )
        recalc_error = ""
        if replay["output_produced"] and not args.skip_recalc:
            try:
                await recalc_with_libreoffice(
                    output_path,
                    sandbox_image=args.sandbox_image,
                    timeout_sec=args.recalc_timeout_sec,
                )
            except Exception as error:
                recalc_error = str(error)

        metrics = await asyncio.to_thread(
            compare_v1_output,
            case.golden_path,
            output_path,
            task.answer_position,
        )
        metrics.update(
            {
                "case_index": case.index,
                "input_file": str(case.input_path),
                "golden_file": str(case.golden_path),
                "output_file": str(output_path) if output_path.is_file() else None,
                "replay": replay,
            }
        )
        if recalc_error:
            metrics["recalc_error"] = recalc_error
        case_metrics.append(metrics)

    metrics = aggregate_v1_metrics(case_metrics)
    return {
        "uid": task.uid,
        "task_id": task.task_id,
        "instruction_type": task.instruction_type,
        "question": task.instruction,
        "submission": result.submission,
        "status": result.status,
        "steps": result.steps,
        "score": metrics["hard_restriction"],
        "token_usage": result.metadata.get("token_usage"),
        "metrics": metrics,
        "solution_file": str(solution_copy_path) if solution_produced else None,
        "messages": result.messages,
        "metadata": {
            **result.metadata,
            "case_count": len(task.cases),
            "generation_backend": {
                "class": type(backend).__name__,
                "kind": args.generation_backend,
                "model_path": backend.model_path,
                "enable_thinking": getattr(backend, "enable_thinking", None),
                "stats": backend.get_stats(),
            },
        },
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


async def run_full_eval(args: argparse.Namespace) -> dict[str, Any]:
    tasks = load_spreadsheetbench_v1_tasks(args.data_root)
    if args.uid is not None:
        tasks = [task for task in tasks if task.uid == args.uid]
        if not tasks:
            raise SystemExit(f"SpreadsheetBench v1 UID not found: {args.uid}")
    if args.num_shards > 1:
        tasks = [
            task
            for position, task in enumerate(tasks)
            if position % args.num_shards == args.shard_index
        ]
        logger.info(
            "selected deterministic task shard %d/%d (%d instructions)",
            args.shard_index,
            args.num_shards,
            len(tasks),
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = [None] if args.samples_per_task == 1 else list(range(args.samples_per_task))
    work = [(task, sample) for task in tasks for sample in samples]
    existing_result_names = _existing_result_names(output_dir)
    pending = [
        (task, sample)
        for task, sample in work
        if _result_path(output_dir, task.uid, sample).name
        not in existing_result_names
    ]
    resumed = len(work) - len(pending)
    logger.info(
        "%d instructions x %d attempts: %d resumed, %d pending (concurrency %d)",
        len(tasks),
        len(samples),
        resumed,
        len(pending),
        args.concurrency,
    )

    # A fully resumed run should only rebuild summary.json, not load a large
    # local model or reconnect to an external server.
    backend = build_generation_backend(args) if pending else None
    semaphore = asyncio.Semaphore(args.concurrency)
    failures: dict[str, str] = {}
    finished = 0
    run_scores: list[float] = []
    started_at = time.monotonic()

    def log_progress() -> None:
        processed = finished + len(failures)
        remaining = len(pending) - processed
        elapsed = time.monotonic() - started_at
        eta = "n/a"
        if processed and remaining > 0:
            eta = f"~{elapsed / processed * remaining / 60:.0f}m"
        elif remaining <= 0:
            eta = "done"
        mean_score = _mean(run_scores)
        logger.info(
            "progress %d/%d (%d failed) | hard %s | elapsed %.0fm, eta %s",
            processed,
            len(pending),
            len(failures),
            f"{mean_score:.3f}" if mean_score is not None else "n/a",
            elapsed / 60,
            eta,
        )

    async def run_one(task: SpreadsheetBenchV1Task, sample: int | None) -> None:
        nonlocal finished
        async with semaphore:
            key = _attempt_key(task.uid, sample)
            run_id = (
                f"{args.run_id}-{key}"
                if args.run_id
                else f"{key}-{uuid.uuid4().hex[:12]}"
            )
            logger.info("[%s] start (%d workbook cases)", key, len(task.cases))
            try:
                payload = await run_task(
                    args,
                    backend,
                    task,
                    run_id,
                    output_dir / "outputs" / key,
                    output_dir / "solutions" / f"{key}.py",
                )
            except Exception as error:
                _raise_if_engine_failed(args.generation_backend, error)
                failures[key] = f"{type(error).__name__}: {error}"
                logger.error("[%s] FAILED: %s", key, failures[key])
                log_progress()
                return
            if payload["status"].startswith("error:"):
                failures[key] = (
                    payload["messages"][-1]["content"]
                    if payload["messages"]
                    else payload["status"]
                )
                _raise_if_engine_failed(args.generation_backend, failures[key])
                logger.error("[%s] FAILED: %s", key, failures[key])
                log_progress()
                return
            if sample is not None:
                payload["sample"] = sample
            _result_path(output_dir, task.uid, sample).write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            failures.pop(key, None)
            finished += 1
            run_scores.append(payload["score"])
            logger.info(
                "[%s] %s hard=%.0f%% soft=%.1f%% cells=%.1f%% steps=%d",
                key,
                payload["status"],
                100 * payload["score"],
                100 * payload["metrics"]["soft_restriction"],
                100 * payload["metrics"]["cell_accuracy"],
                payload["steps"],
            )
            log_progress()

    to_run = pending
    for retry_pass in range(args.failure_retries + 1):
        if not to_run:
            break
        if retry_pass:
            logger.info(
                "retry pass %d/%d: %d attempts after %.0fs",
                retry_pass,
                args.failure_retries,
                len(to_run),
                args.failure_retry_delay_sec,
            )
            if args.failure_retry_delay_sec > 0:
                await asyncio.sleep(args.failure_retry_delay_sec)
        await asyncio.gather(*(run_one(task, sample) for task, sample in to_run))

        def retryable(key: str) -> bool:
            return _classify_failure(failures[key]) != "context_length"

        to_run = [
            (task, sample)
            for task, sample in pending
            if _attempt_key(task.uid, sample) in failures
            and retryable(_attempt_key(task.uid, sample))
        ]

    per_task: dict[str, dict[str, Any]] = {}
    token_usage = None
    existing_result_names = _existing_result_names(output_dir)
    for task in tasks:
        payloads = [
            json.loads(
                _result_path(output_dir, task.uid, sample).read_text(
                    encoding="utf-8"
                )
            )
            for sample in samples
            if _result_path(output_dir, task.uid, sample).name
            in existing_result_names
        ]
        if not payloads:
            continue
        for payload in payloads:
            episode = payload.get("token_usage")
            if isinstance(episode, dict):
                token_usage = add_usage(token_usage, episode)
        per_task[task.uid] = {
            "task_id": task.task_id,
            "instruction_type": task.instruction_type,
            "case_count": len(task.cases),
            "hard_restrictions": [
                p["metrics"]["hard_restriction"] for p in payloads
            ],
            "soft_restrictions": [
                p["metrics"]["soft_restriction"] for p in payloads
            ],
            "cell_accuracies": [p["metrics"]["cell_accuracy"] for p in payloads],
            "statuses": [p["status"] for p in payloads],
            "steps": [p["steps"] for p in payloads],
            "n_attempts": len(payloads),
            "n_success": sum(p["score"] >= 0.5 for p in payloads),
        }

    if token_usage is not None:
        # Per task, not per attempt: a pass@k run pays for every attempt.
        token_usage.update(per_task_means(token_usage, len(per_task)))

    hard = [value for item in per_task.values() for value in item["hard_restrictions"]]
    soft = [value for item in per_task.values() for value in item["soft_restrictions"]]
    cells = [value for item in per_task.values() for value in item["cell_accuracies"]]
    statuses = [value for item in per_task.values() for value in item["statuses"]]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in per_task.values():
        grouped[item["instruction_type"]].append(item)
    by_instruction_type = {
        instruction_type: {
            "tasks": len(items),
            "hard_restriction": _mean(
                [value for item in items for value in item["hard_restrictions"]]
            ),
            "soft_restriction": _mean(
                [value for item in items for value in item["soft_restrictions"]]
            ),
            "cell_accuracy": _mean(
                [value for item in items for value in item["cell_accuracies"]]
            ),
        }
        for instruction_type, items in sorted(grouped.items())
    }

    pass_at_k = None
    if args.samples_per_task > 1:
        pass_at_k = {
            str(k): _mean(
                [
                    _pass_at_k(item["n_attempts"], item["n_success"], k)
                    for item in per_task.values()
                    if item["n_attempts"] >= k
                ]
            )
            for k in range(1, args.samples_per_task + 1)
        }

    summary = {
        "benchmark": "spreadsheetbench-v1",
        "data_root": str(args.data_root),
        "generation_backend": args.generation_backend,
        "model": args.model,
        "enable_thinking": args.enable_thinking,
        "total_instructions": len(tasks),
        "total_workbook_cases": sum(len(task.cases) for task in tasks),
        "samples_per_task": args.samples_per_task,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "completed": len(per_task),
        "resumed": resumed,
        "failed": failures,
        "failure_counts": dict(
            Counter(_classify_failure(value) for value in failures.values())
        ),
        "accuracy": _mean(hard),
        "hard_restriction": _mean(hard),
        "soft_restriction": _mean(soft),
        "cell_accuracy": _mean(cells),
        "scored_attempts": len(hard),
        "by_instruction_type": by_instruction_type,
        "pass_at_k": pass_at_k,
        "status_counts": dict(Counter(statuses)),
        "token_usage": token_usage,
        "results": per_task,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "completed %d/%d (failed %d) | hard %s | soft %s | cells %s",
        summary["completed"],
        summary["total_instructions"],
        len(failures),
        f"{summary['hard_restriction']:.3f}" if hard else "n/a",
        f"{summary['soft_restriction']:.3f}" if soft else "n/a",
        f"{summary['cell_accuracy']:.3f}" if cells else "n/a",
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
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
