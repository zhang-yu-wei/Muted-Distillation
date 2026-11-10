#!/usr/bin/env python3
"""Evaluate a model on JobBench with the mini-swe-agent harness.

Each task gets a fresh workspace holding a copy of its ``task_folder`` and an
empty ``output/``; the agent works there under the vendored mini-swe-agent
control flow, and the LLM judge scores whatever landed in ``output/`` against
the task's weighted rubric. The reported score is the same normalized weighted
score the official JobBench judge produces.

Results are resumable (one JSON per attempt, non-empty files are skipped), and
the run ends with an aggregate ``summary.json``.

The eval split is ``main`` — the split the public leaderboard reports. Two
choices change comparability with published numbers and are recorded in every
result file:

``--search-file-policy stage``
    Copies each task's ``files_required_to_search/`` into the workspace. The
    official runner does not; it expects the agent to fetch those documents
    from the web, which a network-less sandbox cannot do. This is the offline
    runner's default so every document named by the task is actually available.
``--judge-model``
    The published leaderboard uses ``grok-4.3``. A different judge is a
    different measuring stick.

Example::

    JUDGE_API_KEY=... VLLM_MODEL=Qwen/Qwen3.5-4B \\
    python3 scripts/run_jobbench_eval.py \\
        --model Qwen/Qwen3.5-4B --output-dir results/jobbench-qwen35-4b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import shutil
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from officeqa_rl.jobbench.agent import (
    JobBenchAgentConfig,
    JobBenchEpisode,
    run_jobbench_episode,
)
from officeqa_rl.jobbench.dataset import (
    EVAL_SPLIT,
    JobBenchTask,
    SearchFilePolicy,
    load_jobbench_tasks,
)
from officeqa_rl.jobbench.environment import JobBenchEnvironmentConfig
from officeqa_rl.jobbench.extraction import require_document_readers
from officeqa_rl.jobbench.judge import JobBenchScore, JudgeConfig, judge_episode
from officeqa_rl.jobbench.models import JobBenchModelConfig
from officeqa_rl.token_usage import add_usage, per_task_means

from run_full_corpus_inference import (
    build_generation_backend,
    default_eval_concurrency,
    resolve_enable_thinking,
)

logger = logging.getLogger("officeqa_rl.jobbench.eval")


def add_generation_backend_args(parser: argparse.ArgumentParser) -> None:
    """Backend selection flags, matching the OfficeQA and SpreadsheetBench runners."""
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
    parser.add_argument(
        "--gdn-prefill-backend",
        choices=["flashinfer", "triton", "cutedsl"],
        help="Override vLLM's Qwen GDN prefill backend",
    )
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--safetensors-load-strategy",
        choices=["lazy", "eager", "prefetch", "torchao"],
        help=(
            "vLLM safetensors loading strategy. 'prefetch' is useful for "
            "models read directly from GCS-FUSE."
        ),
    )
    parser.add_argument(
        "--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Forward enable_thinking to the tokenizer chat template. Local "
            "Gemma-4-12B evals default this to true; other models leave it unset."
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
    parser.add_argument("--vertex-protocol", choices=["sglang", "vllm"], default="sglang")
    parser.add_argument("--vertex-tokenizer", default="/mnt/Qwen3.5-4B")
    parser.add_argument("--vertex-instruction")
    parser.add_argument("--vertex-max-retries", type=int, default=3)
    parser.add_argument("--vertex-quota-retries", type=int, default=10)
    parser.add_argument("--vertex-request-timeout-sec", type=float, default=600.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/jobbench/hf"))
    parser.add_argument("--split", choices=["main", "easy"], default=EVAL_SPLIT)
    parser.add_argument("--uid", help="Run one task, e.g. JB__main__statisticians__task1")
    parser.add_argument("--occupation", help="Run only tasks of this occupation")
    parser.add_argument("--limit", type=int, help="Run only the first N tasks")
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--failure-retries", type=int, default=1)
    parser.add_argument("--failure-retry-delay-sec", type=float, default=30.0)
    parser.add_argument("--keep-workspaces", action="store_true")

    parser.add_argument("--step-limit", type=int, default=60)
    parser.add_argument("--wall-time-limit-sec", type=int, default=0)
    parser.add_argument("--generation-timeout-sec", type=float, default=None,
                        help="Per-call infrastructure deadline; timeouts stay unscored and resumable")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        # Offline evals run at temperature 1.0 by standing protocol: greedy
        # decoding inflates per-turn length on OPD-trained checkpoints, and
        # makes pass@k degenerate because every sample repeats one trajectory.
    )
    parser.add_argument("--sandbox-image", default="jobbench-sandbox")
    parser.add_argument("--sandbox-runtime", choices=["docker", "local"], default="docker")
    parser.add_argument("--command-timeout-sec", type=int, default=180)
    parser.add_argument("--max-observation-chars", type=int, default=12_000)
    parser.add_argument("--container-memory-limit", default="8g")
    parser.add_argument("--max-concurrent-sandbox-boots", type=int, default=8)
    parser.add_argument(
        "--search-file-policy",
        choices=[policy.value for policy in SearchFilePolicy],
        default=SearchFilePolicy.STAGE.value,
        help=(
            "stage (default): also copy files_required_to_search/ into the "
            "network-less workspace. omit: stage only task_folder/, as the "
            "official networked runner does."
        ),
    )

    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-api-base", default=None)
    parser.add_argument("--judge-timeout-sec", type=float, default=300.0)
    parser.add_argument("--judge-max-retries", type=int, default=3)
    parser.add_argument("--judge-concurrency", type=int, default=4)
    parser.add_argument("--judge-max-completion-tokens", type=int, default=8192)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-reasoning-effort", choices=("", "minimal", "low", "medium", "high", "max"), default="")
    parser.add_argument("--judge-truncation-fallback-effort", choices=("", "minimal", "low", "medium", "high", "max"), default="")
    parser.add_argument(
        "--no-judge-images",
        dest="judge_images",
        action="store_false",
        help="Never attach deliverable images to visual rubric items",
    )

    add_generation_backend_args(parser)

    args = parser.parse_args(argv)
    if args.generation_backend == "vllm":
        args.enable_thinking = resolve_enable_thinking(
            args.model, args.enable_thinking
        )
    if args.concurrency is None:
        args.concurrency = default_eval_concurrency(args.generation_backend, args.model)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
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


def judge_config_from_args(args: argparse.Namespace) -> JudgeConfig:
    defaults = JudgeConfig()
    return JudgeConfig(
        model=args.judge_model or defaults.model,
        api_base=args.judge_api_base or defaults.api_base,
        api_key=defaults.api_key,
        timeout_sec=args.judge_timeout_sec,
        max_retries=args.judge_max_retries,
        max_concurrent_rubrics=args.judge_concurrency,
        max_completion_tokens=args.judge_max_completion_tokens,
        temperature=getattr(args, "judge_temperature", defaults.temperature),
        reasoning_effort=getattr(args, "judge_reasoning_effort", ""),
        truncation_fallback_reasoning_effort=getattr(args, "judge_truncation_fallback_effort", ""),
        attach_images=args.judge_images,
    )


def env_config_from_args(
    args: argparse.Namespace, workspace_root: Path
) -> JobBenchEnvironmentConfig:
    return JobBenchEnvironmentConfig(
        workspace_root=workspace_root,
        sandbox_image=args.sandbox_image if args.sandbox_runtime == "docker" else None,
        sandbox_runtime=args.sandbox_runtime,
        command_timeout_sec=args.command_timeout_sec,
        max_observation_chars=args.max_observation_chars,
        container_memory_limit=args.container_memory_limit,
        search_file_policy=SearchFilePolicy(args.search_file_policy),
    )


def _result_path(output_dir: Path, uid: str, sample: int | None) -> Path:
    suffix = "" if sample is None else f".s{sample}"
    return output_dir / f"{uid}{suffix}.json"


def _has_result(output_dir: Path, uid: str, sample: int | None) -> bool:
    path = _result_path(output_dir, uid, sample)
    return path.is_file() and path.stat().st_size > 0


def _validate_resumed_result_protocol(
    output_dir: Path,
    tasks: Sequence[JobBenchTask],
    samples: Sequence[int | None],
    *,
    search_file_policy: str,
) -> set[str]:
    """Refuse to resume attempts produced with another search-file policy."""
    mismatches: list[tuple[Path, str | None]] = []
    # A negative stat on GCS-FUSE may require a network request. Listing once
    # avoids hundreds of serialized misses before model loading on a fresh run.
    existing = {path.name: path for path in output_dir.glob("*.json")}
    saved: set[str] = set()
    for task in tasks:
        for sample in samples:
            name = _result_path(output_dir, task.uid, sample).name
            path = existing.get(name)
            if path is None or not path.is_file() or path.stat().st_size == 0:
                continue
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
                recorded = result.get("protocol", {}).get("search_file_policy")
            except (OSError, json.JSONDecodeError):
                recorded = None
            if recorded != search_file_policy:
                mismatches.append((path, recorded))
            else:
                saved.add(name)
    if mismatches:
        example_path, recorded = mismatches[0]
        raise SystemExit(
            f"Refusing to mix search-file policies in {output_dir}: "
            f"requested {search_file_policy!r}, but {example_path.name} records "
            f"{recorded!r}. Use a new output directory (for example, suffix it "
            "with '-sfstage')."
        )
    return saved


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def build_result(
    task: JobBenchTask,
    episode: JobBenchEpisode,
    score: JobBenchScore,
    *,
    args: argparse.Namespace,
    elapsed_sec: float,
) -> dict[str, Any]:
    return {
        "uid": task.uid,
        "benchmark": "jobbench",
        "split": task.split,
        "occupation": task.occupation,
        "task_id": task.task_id,
        "score": score.normalized_score,
        "total_score": score.total_score,
        "max_score": score.max_score,
        "pass_rate": score.pass_rate,
        "passed_count": score.passed_count,
        "total_count": score.total_count,
        "graded": score.graded,
        "judge_failures": score.judge_failures,
        "judge_model": score.judge_model,
        "exit_status": episode.exit_status,
        "submitted": episode.submitted,
        "steps": episode.steps,
        "output_files": episode.output_files,
        "elapsed_sec": round(elapsed_sec, 2),
        "protocol": {
            "harness": "mini-swe-agent",
            "mini_swe_agent_version": episode.metadata.get("mini_swe_agent_version"),
            "search_file_policy": args.search_file_policy,
            "step_limit": args.step_limit,
            "temperature": args.temperature,
            "model": args.model,
            "generation_backend": args.generation_backend,
            "enable_thinking": getattr(args, "enable_thinking", None),
            "generation_timeout_sec": getattr(args, "generation_timeout_sec", None),
            "judge_max_completion_tokens": getattr(args, "judge_max_completion_tokens", None),
            "judge_timeout_sec": getattr(args, "judge_timeout_sec", None),
            "judge_temperature": getattr(args, "judge_temperature", 0.0),
            "judge_reasoning_effort": getattr(args, "judge_reasoning_effort", ""),
            "judge_truncation_fallback_effort": getattr(args, "judge_truncation_fallback_effort", ""),
        },
        "token_usage": episode.metadata.get("token_usage", {}),
        "rubrics": [
            {
                "index": rubric.index,
                "weight": rubric.weight,
                "passed": rubric.passed,
                "criteria_passed": rubric.criteria_passed,
                "criteria_count": rubric.criteria_count,
                "vision_used": rubric.vision_used,
                "error": rubric.error,
                "overall_reasoning": rubric.overall_reasoning,
            }
            for rubric in score.rubrics
        ],
        "environment": episode.environment,
        "messages": episode.messages,
    }


async def run_attempt(
    args: argparse.Namespace,
    backend: Any,
    task: JobBenchTask,
    sample: int | None,
    *,
    env_config: JobBenchEnvironmentConfig,
    agent_config: JobBenchAgentConfig,
    judge_config: JudgeConfig,
    episode_semaphore: asyncio.Semaphore,
    boot_semaphore: asyncio.Semaphore,
    output_dir: Path,
) -> dict[str, Any]:
    """Run and grade one attempt, writing its result file."""
    started = time.monotonic()
    # vLLM's DP dispatcher otherwise chooses a replica independently for each
    # turn.  Pin the whole agent trajectory to one replica so its growing chat
    # prefix remains in that replica's KV cache, matching the OfficeQA and
    # SpreadsheetBench offline-eval protocol.
    # Keep this identifier short. JobBench already prefixes workspace names
    # with the (sometimes very long) task UID and then truncates the result;
    # embedding the UID here makes concurrent pass@k attempts collide after
    # truncation when --keep-workspaces is enabled.
    episode_run_id = uuid.uuid4().hex[:12]
    episode_backend = (
        backend.for_episode(episode_run_id)
        if hasattr(backend, "for_episode")
        else backend
    )
    model_config = JobBenchModelConfig(
        model_name=args.model,
        generate_timeout_sec=getattr(args, "generation_timeout_sec", None),
        sampling_params={
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
    )
    async with episode_semaphore:
        episode = await run_jobbench_episode(
            task,
            episode_backend,
            agent_config,
            env_config,
            run_id=episode_run_id,
            model_config=model_config,
            boot_semaphore=boot_semaphore,
        )
    try:
        if episode.exit_status == "GenerationTimeoutError":
            # Preserve the partial trajectory separately: it must not become a
            # zero reward or make resume skip this unfinished attempt.
            failures = output_dir / ".infra-failures"
            failures.mkdir(exist_ok=True)
            partial = dict(uid=task.uid, sample=sample, exit_status=episode.exit_status,
                           workspace=str(episode.workspace), messages=episode.messages,
                           token_usage=episode.metadata.get("token_usage", {}))
            (failures / f"{task.uid}.s{sample}.{episode_run_id}.json").write_text(
                json.dumps(partial, indent=2, ensure_ascii=False) + "\n")
            raise RuntimeError("Generation timeout; partial trajectory retained; attempt remains pending")
        score = await judge_episode(
            task, episode.workspace / "output", judge_config
        )
        result = build_result(
            task, episode, score, args=args, elapsed_sec=time.monotonic() - started
        )
        _result_path(output_dir, task.uid, sample).write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return result
    finally:
        if not args.keep_workspaces and episode.exit_status != "GenerationTimeoutError":
            shutil.rmtree(episode.workspace, ignore_errors=True)


def summarize(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Aggregate attempts into the run summary."""
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_uid[result["uid"]].append(result)

    scores = [result["score"] for result in results]
    exit_statuses = Counter(result["exit_status"] for result in results)
    occupations: dict[str, list[float]] = defaultdict(list)
    for result in results:
        occupations[result["occupation"]].append(result["score"])

    usage: dict[str, Any] = {}
    for result in results:
        usage = add_usage(usage, result.get("token_usage", {}))

    summary: dict[str, Any] = {
        "benchmark": "jobbench",
        "split": args.split,
        "model": args.model,
        "judge_model": results[0]["judge_model"] if results else None,
        "judge_temperatures": sorted({
            result.get("protocol", {}).get("judge_temperature", 0.0)
            for result in results
        }),
        "harness": "mini-swe-agent",
        "search_file_policy": args.search_file_policy,
        "step_limit": args.step_limit,
        "temperature": args.temperature,
        "enable_thinking": getattr(args, "enable_thinking", None),
        "attempts": len(results),
        "tasks": len(by_uid),
        "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "mean_rubric_pass_rate": (
            round(sum(r["pass_rate"] for r in results) / len(results), 4)
            if results
            else 0.0
        ),
        "submitted_rate": (
            round(sum(1 for r in results if r["submitted"]) / len(results), 4)
            if results
            else 0.0
        ),
        "empty_output_rate": (
            round(sum(1 for r in results if not r["output_files"]) / len(results), 4)
            if results
            else 0.0
        ),
        "attempts_with_judge_failures": sum(
            1 for result in results if not result["graded"]
        ),
        "mean_steps": (
            round(sum(r["steps"] for r in results) / len(results), 2) if results else 0.0
        ),
        "exit_status_counts": dict(exit_statuses),
        "score_by_occupation": {
            occupation: round(sum(values) / len(values), 4)
            for occupation, values in sorted(occupations.items())
        },
        "token_usage_total": usage,
        "token_usage_per_task_mean": per_task_means(usage, len(by_uid) or 1),
    }

    if args.samples_per_task > 1:
        # A rubric-weighted score is continuous; "success" for pass@k means a
        # fully passed rubric, which is the only unambiguous binary outcome.
        n = args.samples_per_task
        pass_at_k: dict[str, float] = {}
        for k in range(1, n + 1):
            estimates = []
            for attempts in by_uid.values():
                if len(attempts) < n:
                    continue
                successes = sum(1 for attempt in attempts if attempt["score"] >= 1.0)
                estimates.append(_pass_at_k(n, successes, k))
            if estimates:
                pass_at_k[f"pass@{k}"] = round(sum(estimates) / len(estimates), 4)
        summary["pass_at_k"] = pass_at_k
        summary["best_of_n_mean_score"] = round(
            sum(max(a["score"] for a in attempts) for attempts in by_uid.values())
            / len(by_uid),
            4,
        )
    return summary


async def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    require_document_readers()
    tasks = load_jobbench_tasks(args.data_root, args.split)
    if args.uid:
        tasks = [task for task in tasks if task.uid == args.uid]
        if not tasks:
            raise SystemExit(f"No JobBench task with uid {args.uid!r} in {args.split}")
    if args.occupation:
        tasks = [task for task in tasks if task.occupation == args.occupation]
    if args.limit:
        tasks = tasks[: args.limit]
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
    if not tasks:
        raise SystemExit("No tasks selected")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or uuid.uuid4().hex[:12]
    workspace_root = args.workspace_root or (output_dir / "workspaces")
    samples = (
        [None] if args.samples_per_task == 1 else list(range(args.samples_per_task))
    )
    logger.info("Checking retained rollout protocol in %s", output_dir)
    saved_names = _validate_resumed_result_protocol(
        output_dir,
        tasks,
        samples,
        search_file_policy=args.search_file_policy,
    )
    logger.info("Found %d retained attempts; initializing generation backend", len(saved_names))

    env_config = env_config_from_args(args, workspace_root)
    agent_config = JobBenchAgentConfig(
        step_limit=args.step_limit,
        wall_time_limit_seconds=args.wall_time_limit_sec,
    )
    judge_config = judge_config_from_args(args)
    judge_config.require_key()

    backend = build_generation_backend(args)
    episode_semaphore = asyncio.Semaphore(args.concurrency)
    boot_semaphore = asyncio.Semaphore(args.max_concurrent_sandbox_boots)

    pending = [
        (task, sample)
        for task in tasks
        for sample in samples
        if _result_path(output_dir, task.uid, sample).name not in saved_names
    ]
    logger.info(
        "JobBench %s: %d tasks, %d attempts pending (run %s)",
        args.split,
        len(tasks),
        len(pending),
        run_id,
    )

    async def attempt_with_retries(task: JobBenchTask, sample: int | None) -> None:
        for retry in range(args.failure_retries + 1):
            try:
                await run_attempt(
                    args,
                    backend,
                    task,
                    sample,
                    env_config=env_config,
                    agent_config=agent_config,
                    judge_config=judge_config,
                    episode_semaphore=episode_semaphore,
                    boot_semaphore=boot_semaphore,
                    output_dir=output_dir,
                )
                return
            except Exception as error:
                logger.warning(
                    "Attempt failed for %s (sample %s, retry %d/%d): %s",
                    task.uid,
                    sample,
                    retry,
                    args.failure_retries,
                    error,
                )
                if retry >= args.failure_retries:
                    logger.error("Giving up on %s (sample %s)", task.uid, sample)
                    return
                await asyncio.sleep(args.failure_retry_delay_sec)

    await asyncio.gather(
        *(attempt_with_retries(task, sample) for task, sample in pending)
    )

    results: list[dict[str, Any]] = []
    for task in tasks:
        for sample in samples:
            path = _result_path(output_dir, task.uid, sample)
            if path.is_file() and path.stat().st_size > 0:
                results.append(json.loads(path.read_text(encoding="utf-8")))

    summary = summarize(results, args)
    summary["run_id"] = run_id
    summary["missing_attempts"] = len(tasks) * len(samples) - len(results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not args.keep_workspaces:
        # Each attempt already removed its own workspace. Only drop the root if
        # it is now empty: --workspace-root may point at a directory shared
        # with a training run, and an rmtree there would delete live episodes.
        try:
            workspace_root.rmdir()
        except OSError:
            pass
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    summary = asyncio.run(run_eval(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
