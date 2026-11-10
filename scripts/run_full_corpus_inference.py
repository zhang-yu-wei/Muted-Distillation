"""Run OfficeQA full-corpus tasks with a selected generation backend.

With ``--uid`` the script evaluates exactly one benchmark row and writes one
result payload. Without it, every row of ``--dataset-csv`` is evaluated with
bounded concurrency; each row writes ``<output-dir>/<uid>.json``, rows that
already have a non-empty result file are skipped (resume), and an aggregate
``summary.json`` is written and printed at the end.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from officeqa_rl.agent import OfficeQAAgent
from officeqa_rl.backends._errors import is_context_length_error
from officeqa_rl.backends.gemini import GeminiGenerationBackend
from officeqa_rl.backends.kimi import KimiGenerationBackend
from officeqa_rl.backends.openai_server import OpenAIServerGenerationBackend
from officeqa_rl.backends.vertex import VertexSGLangGenerationBackend, rawpredict_url
from officeqa_rl.backends.vllm import VLLMGenerationBackend
from officeqa_rl.dataset import load_task, read_csv_rows, task_from_csv_row
from officeqa_rl.full_corpus_env import FullCorpusEnvironment
from officeqa_rl.prompts import CORPUS_PROFILES
from officeqa_rl.scoring import score_answer
from officeqa_rl.token_usage import add_usage, per_task_means
from officeqa_rl.types import AgentConfig, EnvironmentConfig, OfficeQATask

logger = logging.getLogger("officeqa_rl.inference")


_DENSE_4B_MODEL_COMPONENT = re.compile(
    r"(?:^|[/_.-])4b(?:$|[/_.-])",
    flags=re.IGNORECASE,
)


def default_eval_concurrency(generation_backend: str, model: str) -> int:
    """Return the workload default without overriding an explicit setting.

    Dense 4B model names conventionally contain a standalone ``4B`` path/name
    component.  Do not classify MoE descriptors such as A4B or E4B as 4B
    checkpoints: their total weights have a different memory profile.
    """
    if generation_backend != "vllm":
        return 4
    if _DENSE_4B_MODEL_COMPONENT.search(model):
        return 48
    return 32


def resolve_enable_thinking(model: str, requested: bool | None) -> bool | None:
    """Resolve the local-vLLM chat-template thinking mode.

    Gemma-4-12B's template defaults to thinking disabled, which is the wrong
    protocol for our evals. Other model families are left unset because their
    templates may not accept this model-specific keyword.
    """
    if requested is not None:
        return requested
    normalized_model = re.sub(r"[_/]", "-", model.lower())
    if re.search(r"gemma-?4-?12b", normalized_model):
        return True
    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, required=True)
    parser.add_argument(
        "--uid",
        help="Run exactly this OfficeQA UID; omit to evaluate every dataset row",
    )
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
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument(
        "--page-text-dir",
        type=Path,
        help=(
            "Page-indexed parsed text containing '--- PAGE N ---' markers; "
            "defaults to --corpus-dir"
        ),
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        help="Original PDFs corresponding by basename to page-indexed text files",
    )
    parser.add_argument(
        "--enable-visual-pdf-tool",
        action="store_true",
        help=(
            "After a model rg/grep corpus search, attach the best matching PDF "
            "page plus parsed text from the documents that search matched; "
            "also enable view_page as a fallback"
        ),
    )
    parser.add_argument(
        "--pdf-render-dpi",
        type=int,
        default=144,
        help="Resolution used for automatic retrieval and view_page (default: 144 DPI)",
    )
    parser.add_argument(
        "--generation-backend",
        choices=["vllm", "gemini", "kimi", "openai-server", "vertex"],
        default="vllm",
    )
    parser.add_argument("--model", "--model-path", dest="model", required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path("results/workspaces"))
    parser.add_argument("--output", type=Path, help="Result path (single-UID mode only)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Per-UID results and summary.json directory (full-dataset mode only)",
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=(
            "Concurrent questions in full-dataset mode. For in-process vLLM, "
            "defaults to 48 for dense 4B models and 32 otherwise; API backends "
            "default to 4."
        ),
    )
    parser.add_argument(
        "--failure-retries",
        type=int,
        default=2,
        help="Extra passes over failed questions in full-dataset mode",
    )
    parser.add_argument(
        "--failure-retry-delay-sec",
        type=float,
        default=60.0,
        help="Pause before each retry pass (lets API quota windows recover)",
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=1,
        help="Independent attempts per question for pass@k (full-dataset mode; "
        "needs --temperature > 0 when above 1)",
    )

    # 50 matches training's AGENT_STEP_LIMIT and keeps episodes inside the
    # model context window; the eval path has no per-turn context budgeting,
    # so long episodes would otherwise crash on the model's context limit.
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
    # The system prompt's filename hints must describe the corpus actually
    # mounted: "v1" is the Treasury Bulletin corpus, "v2" the Combined
    # Statement/receipts corpus. Default "v1" keeps existing runs unchanged.
    parser.add_argument("--corpus-profile", choices=sorted(CORPUS_PROFILES),
                        default=os.environ.get("OFFICEQA_CORPUS_PROFILE", "v1"))
    parser.add_argument("--web-search-instruction", action="store_true")

    parser.add_argument("--sandbox-runtime", choices=["docker", "local"], default="docker")
    parser.add_argument("--sandbox-image", default="python:3.11-slim")
    parser.add_argument("--command-timeout-sec", type=int, default=30)
    parser.add_argument("--max-observation-chars", type=int, default=12_000)

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help=(
            "Independent in-process vLLM replicas with internal load balancing. "
            "GPU use is data × tensor × pipeline parallel size (default: 1)."
        ),
    )
    parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        help="Shard MoE experts across the parallel workers (default: disabled).",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
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
        "--gdn-prefill-backend",
        choices=["flashinfer", "triton", "cutedsl"],
        help="Override vLLM's Qwen GDN prefill backend",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable vLLM prefix caching (default: enabled). Hybrid models may "
            "treat this as experimental; use --no-enable-prefix-caching to opt out."
        ),
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
        "--kimi-reasoning-effort",
        choices=["low", "high", "max"],
        default="max",
    )
    parser.add_argument(
        "--vertex-url",
        help=(
            "Full rawPredict URL of a Vertex endpoint whose predict route is "
            "SGLang /generate (or a plain http://host:port/generate server); "
            "overrides the project/location/endpoint-id trio"
        ),
    )
    parser.add_argument("--vertex-project")
    parser.add_argument("--vertex-location")
    parser.add_argument("--vertex-endpoint-id")
    parser.add_argument(
        "--vertex-protocol",
        choices=["sglang", "vllm"],
        default="sglang",
        help="Native predict route protocol (default: sglang)",
    )
    parser.add_argument(
        "--vertex-tokenizer",
        default="/mnt/Qwen3.5-4B",
        help=(
            "HF checkpoint whose tokenizer renders the chat template locally; "
            "the /generate route takes raw prompt text, and the served model "
            "must share this tokenizer's vocabulary"
        ),
    )
    parser.add_argument(
        "--vertex-instruction",
        help=(
            "Optional reasoning-effort instruction prepended to every prompt "
            "as its own leading system block — the same context layout the "
            "OPD teacher sees during training scoring. Empty keeps prompts "
            "byte-identical to previous runs."
        ),
    )
    parser.add_argument("--vertex-max-retries", type=int, default=3)
    parser.add_argument("--vertex-quota-retries", type=int, default=10)
    parser.add_argument(
        "--vertex-request-timeout-sec",
        type=float,
        default=600.0,
        help="Per-call timeout; long-context turns on a busy endpoint take minutes",
    )
    parser.add_argument(
        "--openai-base-url",
        default="http://localhost:8000/v1",
        help="Chat Completions root of a self-hosted vLLM/SGLang server",
    )
    parser.add_argument("--openai-max-retries", type=int, default=3)
    parser.add_argument("--openai-quota-retries", type=int, default=10)
    parser.add_argument(
        "--openai-request-timeout-sec",
        type=float,
        default=600.0,
        help="Per-call timeout; long-context reasoning turns can take minutes",
    )
    args = parser.parse_args(argv)
    if args.generation_backend == "vllm":
        args.enable_thinking = resolve_enable_thinking(
            args.model, args.enable_thinking
        )
    if args.concurrency is None:
        args.concurrency = default_eval_concurrency(args.generation_backend, args.model)
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
    if args.uid is None and args.output is not None:
        parser.error("--output requires --uid; use --output-dir for full-dataset runs")
    if args.uid is not None and args.output_dir is not None:
        parser.error("--output-dir is for full-dataset runs; use --output with --uid")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.data_parallel_size < 1:
        parser.error("--data-parallel-size must be at least 1")
    if args.samples_per_task < 1:
        parser.error("--samples-per-task must be at least 1")
    if args.samples_per_task > 1 and args.uid is not None:
        parser.error("--samples-per-task is for full-dataset runs; single-UID mode runs one attempt")
    if args.samples_per_task > 1 and args.temperature <= 0.0:
        parser.error(
            "--samples-per-task > 1 needs --temperature > 0: at temperature 0 "
            "every attempt is identical and pass@k equals pass@1"
        )
    if args.enable_visual_pdf_tool and args.pdf_dir is None:
        parser.error("--enable-visual-pdf-tool requires --pdf-dir")
    if args.enable_visual_pdf_tool and args.pdf_render_dpi < 1:
        parser.error("--pdf-render-dpi must be at least 1")
    return args


def _api_keys_from_env(file_var: str, list_var: str) -> list[str] | None:
    """Read a secret key pool without putting keys in argv or shell history."""
    keys_file = os.environ.get(file_var)
    if keys_file:
        lines = Path(keys_file).read_text().splitlines()
        keys = [line.strip() for line in lines]
        keys = [key for key in keys if key and not key.startswith("#")]
        if not keys:
            raise SystemExit(f"{file_var} {keys_file} contains no keys")
        return keys
    raw = os.environ.get(list_var, "")
    keys = [key for key in re.split(r"[,\s]+", raw) if key]
    return keys or None


def _gemini_api_keys_from_env() -> list[str] | None:
    """GEMINI_API_KEYS_FILE takes precedence over GEMINI_API_KEYS."""
    return _api_keys_from_env("GEMINI_API_KEYS_FILE", "GEMINI_API_KEYS")


def build_generation_backend(args: argparse.Namespace) -> Any:
    """Construct only the SDK/runtime selected for this inference run."""
    if args.generation_backend == "gemini":
        return GeminiGenerationBackend(
            args.model,
            api_version=args.gemini_api_version,
            max_tokens=args.max_tokens,
            max_retries=args.gemini_max_retries,
            quota_max_retries=args.gemini_quota_retries,
            thinking_budget=args.gemini_thinking_budget,
            include_thoughts=args.gemini_include_thoughts,
            serialize_calls=args.serialize_generation_calls,
            api_keys=_gemini_api_keys_from_env(),
        )
    if args.generation_backend == "kimi":
        return KimiGenerationBackend(
            args.model,
            base_url=args.kimi_base_url,
            max_tokens=args.max_tokens,
            reasoning_effort=args.kimi_reasoning_effort,
            max_retries=args.kimi_max_retries,
            quota_max_retries=args.kimi_quota_retries,
            serialize_calls=args.serialize_generation_calls,
        )
    if args.generation_backend == "vertex":
        vertex_url = args.vertex_url or rawpredict_url(
            args.vertex_project, args.vertex_location, args.vertex_endpoint_id
        )
        return VertexSGLangGenerationBackend(
            args.model,
            base_url=vertex_url,
            protocol=args.vertex_protocol,
            tokenizer_path=args.vertex_tokenizer,
            max_tokens=args.max_tokens,
            max_retries=args.vertex_max_retries,
            quota_max_retries=args.vertex_quota_retries,
            request_timeout_sec=args.vertex_request_timeout_sec,
            serialize_calls=args.serialize_generation_calls,
            system_instruction=args.vertex_instruction or None,
            enable_thinking=args.enable_thinking,
        )
    if args.generation_backend == "openai-server":
        return OpenAIServerGenerationBackend(
            args.model,
            base_url=args.openai_base_url,
            max_tokens=args.max_tokens,
            max_retries=args.openai_max_retries,
            quota_max_retries=args.openai_quota_retries,
            request_timeout_sec=args.openai_request_timeout_sec,
            serialize_calls=args.serialize_generation_calls,
            data_parallel_size=args.data_parallel_size,
        )

    vllm_kwargs: dict[str, Any] = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "enable_expert_parallel": args.enable_expert_parallel,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
    }
    if args.max_model_len is not None:
        vllm_kwargs["max_model_len"] = args.max_model_len
    if args.safetensors_load_strategy is not None:
        vllm_kwargs["safetensors_load_strategy"] = args.safetensors_load_strategy
    if args.enable_prefix_caching is not None:
        vllm_kwargs["enable_prefix_caching"] = args.enable_prefix_caching
    if args.gdn_prefill_backend is not None:
        vllm_kwargs["gdn_prefill_backend"] = args.gdn_prefill_backend
    enable_thinking = resolve_enable_thinking(
        args.model, getattr(args, "enable_thinking", None)
    )
    return VLLMGenerationBackend(
        args.model,
        max_tokens=args.max_tokens,
        serialize_calls=args.serialize_generation_calls,
        enable_thinking=enable_thinking,
        **vllm_kwargs,
    )


async def run_task(
    args: argparse.Namespace,
    backend: Any,
    task: OfficeQATask,
    run_id: str,
) -> dict[str, Any]:
    """Run one task through the shared agent/environment path and score it."""
    sampling_params = (
        {} if args.generation_backend == "kimi" else {"temperature": args.temperature}
    )
    # vLLM's internal DP load balancer otherwise routes every turn
    # independently. Bind the whole episode to one rank so a replica-local
    # prefix cache can reuse the growing conversation prefix.
    episode_backend = (
        backend.for_episode(run_id) if hasattr(backend, "for_episode") else backend
    )
    agent = OfficeQAAgent(
        episode_backend,
        AgentConfig(
            step_limit=args.step_limit,
            action_block_language=args.action_block_language,
            web_search_instruction=args.web_search_instruction,
            corpus_profile=args.corpus_profile,
        ),
        sampling_params=sampling_params,
    )
    environment_config = EnvironmentConfig(
        corpus_dir=args.corpus_dir,
        pdf_dir=args.pdf_dir,
        workspace_root=args.workspace_root,
        sandbox_image=args.sandbox_image,
        command_timeout_sec=args.command_timeout_sec,
        max_observation_chars=args.max_observation_chars,
        sandbox_runtime=args.sandbox_runtime,
        page_text_dir=args.page_text_dir,
        enable_visual_pdf_tool=args.enable_visual_pdf_tool,
        pdf_render_dpi=args.pdf_render_dpi,
    )

    async with FullCorpusEnvironment(
        environment_config,
        task,
        run_id=run_id,
    ) as environment:
        result = await agent.run(environment)

    score = (
        score_answer(task.answer, result.submission, tolerance=0.01)
        if task.answer is not None
        else None
    )
    return {
        "uid": task.uid,
        "question": task.question,
        "ground_truth": task.answer,
        "submission": result.submission,
        "status": result.status,
        "steps": result.steps,
        "score": score,
        # Top-level so cost-per-task analysis never has to walk the trace.
        "token_usage": result.metadata.get("token_usage"),
        "messages": result.messages,
        "metadata": {
            **result.metadata,
            "generation_backend": {
                "class": type(backend).__name__,
                "kind": args.generation_backend,
                "model_path": backend.model_path,
                "enable_thinking": getattr(backend, "enable_thinking", None),
                "stats": backend.get_stats(),
            },
        },
    }


async def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate exactly one UID (single-UID mode)."""
    task = load_task(args.dataset_csv, args.uid)
    run_id = args.run_id or f"{task.uid}-{uuid.uuid4().hex[:12]}"
    backend = build_generation_backend(args)
    payload = await run_task(args, backend, task, run_id)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _result_path(output_dir: Path, uid: str, sample: int | None = None) -> Path:
    """Single-attempt runs keep the flat <uid>.json layout; pass@k runs write
    one <uid>.s<i>.json per attempt so resume works per attempt."""
    name = f"{uid}.json" if sample is None else f"{uid}.s{sample}.json"
    return output_dir / name


def _has_result(output_dir: Path, uid: str, sample: int | None = None) -> bool:
    path = _result_path(output_dir, uid, sample)
    return path.is_file() and path.stat().st_size > 0


def _existing_result_names(output_dir: Path) -> set[str]:
    """List resumable result files without stat-ing every expected path.

    On object-store FUSE mounts, a negative ``stat`` can take hundreds of
    milliseconds.  Checking all task/sample combinations one by one therefore
    adds minutes to pass@k startup when the output directory is empty.  A
    directory listing makes the cost proportional to results that actually
    exist instead.
    """
    names: set[str] = set()
    with os.scandir(output_dir) as entries:
        for entry in entries:
            if entry.name == "summary.json" or not entry.name.endswith(".json"):
                continue
            try:
                if entry.is_file() and entry.stat().st_size > 0:
                    names.add(entry.name)
            except FileNotFoundError:
                # A concurrent writer may replace an entry between readdir and
                # stat.  Treat it as pending; its atomic result will be found on
                # the next resume.
                continue
    return names


def _classify_failure(detail: str) -> str:
    """Bucket a trajectory-failure message so the summary can show what class
    of problem ate rows: quota exhaustion silently shrinks the scored set and
    must be visible, not buried in per-UID error strings."""
    if "RESOURCE_EXHAUSTED" in detail or re.search(r"\b429\b", detail):
        return "resource_exhausted"
    if "UNAVAILABLE" in detail or re.search(r"\b503\b", detail):
        return "unavailable"
    if is_context_length_error(detail):
        return "context_length"
    return "other"


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021): the probability that at
    least one of k attempts drawn from n (with c successes) succeeds."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


async def run_full_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate every dataset row with bounded concurrency, resume, and summary."""
    tasks = [task_from_csv_row(row) for row in read_csv_rows(args.dataset_csv)]
    # Deterministic sharding, same contract as run_spreadsheetbench_v1_eval.py:
    # lets one dataset be split across several endpoints (each shard writes its
    # own output dir, and the per-UID files pool afterwards).
    if args.num_shards > 1:
        tasks = [
            task
            for position, task in enumerate(tasks)
            if position % args.num_shards == args.shard_index
        ]
        logger.info(
            "selected deterministic task shard %d/%d (%d questions)",
            args.shard_index,
            args.num_shards,
            len(tasks),
        )
    output_dir = args.output_dir or Path("results") / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    # sample None keeps the legacy single-attempt file layout; a pass@k run
    # fans each question out into samples_per_task independent attempts.
    samples = [None] if args.samples_per_task == 1 else list(range(args.samples_per_task))
    work = [(task, sample) for task in tasks for sample in samples]
    existing_result_names = _existing_result_names(output_dir)
    pending = [
        (task, sample)
        for task, sample in work
        if _result_path(output_dir, task.uid, sample).name not in existing_result_names
    ]
    resumed = len(work) - len(pending)
    logger.info(
        "%d dataset rows x %d attempts: %d already have results, %d to run (concurrency %d)",
        len(tasks),
        len(samples),
        resumed,
        len(pending),
        args.concurrency,
    )

    # A fully resumed run only needs to rebuild summary.json. Avoid loading a
    # local model or reconnecting to an API when there is no generation work.
    backend = build_generation_backend(args) if pending else None
    semaphore = asyncio.Semaphore(args.concurrency)
    failures: dict[str, str] = {}
    finished = 0
    run_scores: list[float] = []
    started_at = time.monotonic()

    def log_progress() -> None:
        processed = finished + len(failures)
        overall = resumed + processed
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
            "progress: %d/%d trajectories this run (%d failed) | overall %d/%d (%.0f%%) | "
            "mean score this run %s | elapsed %.0fm, eta %s",
            processed,
            len(pending),
            len(failures),
            overall,
            len(work),
            100 * overall / len(work),
            mean_text,
            elapsed / 60,
            eta_text,
        )

    def _item_key(task: OfficeQATask, sample: int | None) -> str:
        return task.uid if sample is None else f"{task.uid}.s{sample}"

    def persist_result(
        task: OfficeQATask,
        sample: int | None,
        payload: dict[str, Any],
    ) -> None:
        nonlocal finished
        key = _item_key(task, sample)
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
        if payload["score"] is not None:
            run_scores.append(payload["score"])

    def persist_context_overflow(
        task: OfficeQATask,
        sample: int | None,
        run_id: str,
        detail: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        terminal = payload or {
            "uid": task.uid,
            "question": task.question,
            "ground_truth": task.answer,
            "submission": None,
            "steps": 0,
            "token_usage": None,
            "messages": [],
            "metadata": {"run_id": run_id},
        }
        terminal["status"] = "context_length_exceeded"
        terminal["submission"] = None
        terminal["score"] = 0.0 if task.answer is not None else None
        metadata = terminal.setdefault("metadata", {})
        metadata["terminal_failure"] = {
            "type": "context_length",
            "detail": detail,
        }
        persist_result(task, sample, terminal)
        logger.error(
            "[%s] context length exceeded; recorded as incorrect without retry",
            _item_key(task, sample),
        )
        log_progress()

    async def run_one(task: OfficeQATask, sample: int | None) -> None:
        nonlocal finished
        async with semaphore:
            key = _item_key(task, sample)
            run_id = (
                f"{args.run_id}-{key}"
                if args.run_id
                else f"{key}-{uuid.uuid4().hex[:12]}"
            )
            logger.info("[%s] start", key)
            try:
                payload = await run_task(args, backend, task, run_id)
            except Exception as error:
                detail = f"{type(error).__name__}: {error}"
                if is_context_length_error(detail):
                    persist_context_overflow(task, sample, run_id, detail)
                    return
                failures[key] = detail
                logger.error("[%s] FAILED: %s", key, failures[key])
                log_progress()
                return
            if payload["status"].startswith("error:"):
                # The agent converts crashes (backend/environment exceptions)
                # into an error status; persisting those as results would make
                # resume skip them, so record a retryable failure instead.
                detail = (
                    payload["messages"][-1]["content"]
                    if payload["messages"]
                    else payload["status"]
                )
                if is_context_length_error(detail):
                    persist_context_overflow(task, sample, run_id, detail, payload)
                    return
                failures[key] = detail
                logger.error("[%s] FAILED: %s", key, detail)
                log_progress()
                return
            persist_result(task, sample, payload)
            logger.info(
                "[%s] %s score=%s steps=%d",
                key,
                payload["status"],
                payload["score"],
                payload["steps"],
            )
            log_progress()

    to_run = pending
    for retry_pass in range(args.failure_retries + 1):
        if not to_run:
            break
        if retry_pass:
            logger.info(
                "retry pass %d/%d: %d failed trajectories, pausing %.0fs first",
                retry_pass,
                args.failure_retries,
                len(to_run),
                args.failure_retry_delay_sec,
            )
            if args.failure_retry_delay_sec > 0:
                await asyncio.sleep(args.failure_retry_delay_sec)
        await asyncio.gather(*(run_one(task, sample) for task, sample in to_run))
        to_run = [
            (task, sample)
            for task, sample in pending
            if _item_key(task, sample) in failures
        ]

    results = {}
    token_usage = None
    for task in tasks:
        payloads = [
            json.loads(
                _result_path(output_dir, task.uid, sample).read_text(encoding="utf-8")
            )
            for sample in samples
            if _result_path(output_dir, task.uid, sample).name in existing_result_names
        ]
        if not payloads:
            continue
        for payload in payloads:
            episode = payload.get("token_usage")
            if isinstance(episode, dict):
                token_usage = add_usage(token_usage, episode)
        if args.samples_per_task == 1:
            payload = payloads[0]
            results[task.uid] = {
                "score": payload["score"],
                "status": payload["status"],
                "steps": payload["steps"],
            }
        else:
            sample_scores = [payload["score"] for payload in payloads]
            scored = [score for score in sample_scores if score is not None]
            results[task.uid] = {
                "sample_scores": sample_scores,
                "n_attempts": len(payloads),
                "n_success": sum(1 for score in scored if score >= 0.5),
                "statuses": [payload["status"] for payload in payloads],
                "steps": [payload["steps"] for payload in payloads],
            }

    all_statuses = []
    scores = []
    for entry in results.values():
        if args.samples_per_task == 1:
            all_statuses.append(entry["status"])
            if entry["score"] is not None:
                scores.append(entry["score"])
        else:
            all_statuses.extend(entry["statuses"])
            scores.extend(score for score in entry["sample_scores"] if score is not None)

    if token_usage is not None:
        # Per task, not per attempt: a pass@k run pays for every attempt, so
        # the cost of one benchmark question is the cost of all k of them.
        token_usage.update(per_task_means(token_usage, len(results)))

    pass_at_k = None
    if args.samples_per_task > 1:
        pass_at_k = {}
        for k in range(1, args.samples_per_task + 1):
            estimates = [
                _pass_at_k(entry["n_attempts"], entry["n_success"], k)
                for entry in results.values()
                if entry["n_attempts"] >= k
            ]
            pass_at_k[str(k)] = (
                sum(estimates) / len(estimates) if estimates else None
            )

    summary = {
        "dataset_csv": str(args.dataset_csv),
        "generation_backend": args.generation_backend,
        "model": args.model,
        "enable_thinking": args.enable_thinking,
        "total_rows": len(tasks),
        "samples_per_task": args.samples_per_task,
        "completed": len(results),
        "resumed": resumed,
        "failed": failures,
        "failure_counts": dict(
            Counter(_classify_failure(detail) for detail in failures.values())
        ),
        "mean_score": sum(scores) / len(scores) if scores else None,
        "scored": len(scores),
        "pass_at_k": pass_at_k,
        "status_counts": dict(Counter(all_statuses)),
        "token_usage": token_usage,
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "completed %d/%d (failed %d) | mean score %s over %d scored | %s",
        summary["completed"],
        summary["total_rows"],
        len(failures),
        f"{summary['mean_score']:.3f}" if summary["mean_score"] is not None else "n/a",
        summary["scored"],
        summary["status_counts"],
    )
    if pass_at_k:
        logger.info(
            "pass@k: %s",
            ", ".join(
                f"pass@{k}={value:.3f}" if value is not None else f"pass@{k}=n/a"
                for k, value in pass_at_k.items()
            ),
        )
    if failures:
        logger.warning(
            "%d rows are MISSING from the score (%s): mean_score covers only "
            "the %d scored rows, so these failures inflate it. Failure "
            "breakdown: %s",
            len(failures),
            ", ".join(sorted(failures)),
            summary["scored"],
            summary["failure_counts"],
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
    if args.uid is not None:
        payload = asyncio.run(run_inference(args))
        print(json.dumps(payload, indent=2))
    else:
        summary = asyncio.run(run_full_eval(args))
        print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
