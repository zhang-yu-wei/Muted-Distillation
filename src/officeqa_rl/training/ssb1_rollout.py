"""Slime per-sample rollout hook for SpreadsheetBench v1 (SSB1) RL training.

This is the ``--custom-generate-function-path
officeqa_rl.training.ssb1_rollout.generate`` entry point. It is a sibling of
:mod:`officeqa_rl.training.rollout`, not a modification of it: the OfficeQA
hook stays byte-identical, and this module imports its shared pieces
(backend construction, sample building, abort semantics, per-sample OPD
teacher scoring) as plain functions. An SSB1 run selects this hook in its
launch script; OfficeQA runs never load this module.

Row contract (produced by ``scripts/build_ssb1_slime_dataset.py``): the
sample's ``metadata`` carries ``benchmark: "ssb1"``, ``task_id``, and
``data_root``. The full task — instruction, answer position, and workbook
cases — is reloaded from ``data_root``'s ``dataset.json`` by ``task_id``, so
golden workbooks never ride in the row. Rows whose ``metadata.benchmark`` is
anything else are forwarded unchanged to the hook named by
``officeqa_ssb1_fallback_generate_path`` (default the OfficeQA hook), which
keeps mixed prompt or eval datasets working.

Episode and reward follow the offline eval protocol
(``scripts/run_spreadsheetbench_v1_eval.py``): the agent sees workbook case 1
inside a ``SpreadsheetBenchV1Environment`` sandbox and must leave a reusable
``solution.py``; after submission that program is replayed on EVERY workbook
case in a fresh network-less container, each output is recalculated with
LibreOffice, and the designated cells are compared against the golden
workbook. The reward is the official soft restriction (fraction of cases
fully correct) during training — hard restriction at a ~30% base rate over
~3 cases is a needlessly sparse signal — and the official hard restriction
during evaluation, matching the offline "strict" numbers. On the Verified-400
eval split every task has exactly one case, so the two are identical there
anyway. Both, plus cell accuracy, are logged in the sample metadata either
way (see :mod:`officeqa_rl.training.ssb1_metrics`).

Settings ride on ``args`` through slime's ``--custom-config-path`` YAML
(``configs/training/ssb1_rollout.yaml``), sharing ``officeqa_workspace_root``,
``officeqa_max_observation_chars``, and
``officeqa_max_concurrent_sandbox_boots`` with the OfficeQA hook and adding
``officeqa_ssb1_*`` keys for the sandbox image, step limit, timeouts, replay
concurrency, and fallback path.

Under DooD the workspace root and every row's ``data_root`` must exist at the
same absolute path on the Docker host and inside the training container. That
is also why replays cannot reuse the eval helper
:func:`officeqa_rl.spreadsheetbench_v1_scoring.replay_solution`: it stages
each case in a ``tempfile`` directory, and a ``/tmp`` path inside the
training container does not exist on the Docker host, so it cannot be
volume-mounted. :func:`_replay_case` keeps the identical container contract
but stages under the DooD-visible workspace root.

RL+OPD (``--use-opd --opd-type sglang``) supports both scoring shapes. The
default is the per-sample path: every emitted training sample's full token
sequence is scored by the served teacher after the episode, exactly as in
:mod:`officeqa_rl.training.rollout`. With ``officeqa_opd_per_turn`` set, the
per-turn engine from :mod:`officeqa_rl.training.opd_rollout` runs instead —
the same scorer, turn plans, and mode knobs (action-only, mask-think,
teacher-no-think, tail/prose modifiers, teacher-think) as the OfficeQA
hill-climb arms, applied to SSB1 episodes.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable
from weakref import WeakKeyDictionary

from slime.utils.types import Sample

from officeqa_rl.agent import OfficeQAAgent
from officeqa_rl.training.opd_rollout import (
    _TurnScorer,
    build_scored_samples,
    parse_per_turn_modes,
)
from officeqa_rl.spreadsheetbench_scoring import recalc_with_libreoffice
from officeqa_rl.spreadsheetbench_v1 import (
    SpreadsheetBenchV1Case,
    SpreadsheetBenchV1Environment,
    SpreadsheetBenchV1Task,
    build_spreadsheet_v1_system_prompt,
    load_spreadsheetbench_v1_tasks,
)
from officeqa_rl.spreadsheetbench_v1_scoring import (
    aggregate_v1_metrics,
    compare_v1_output,
    decode_subprocess_output,
)
from officeqa_rl.training.rollout import (
    _STATUS_BY_EXIT,
    _abort,
    _attach_teacher_log_probs,
    _create_backend,
    _episode_output_tokens,
    _forced_close_turns,
    _log_teacher_failure,
    _make_run_id,
    _require_arg,
    _teacher_scoring_enabled,
    build_training_samples,
)
from officeqa_rl.types import AgentConfig, EnvironmentConfig

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_IMAGE = "spreadsheetbench-v1-sandbox"
DEFAULT_FALLBACK_GENERATE_PATH = "officeqa_rl.training.rollout.generate"
# Hard cap for every container that executes agent-written code. Grading a
# workbook needs well under 1 GiB; the cap exists so one runaway program
# cannot take down the host (2026-08-06: three timeout-orphaned replay
# containers grew to ~1.7 TiB combined and Ray's memory monitor killed the
# training job).
DEFAULT_CONTAINER_MEMORY_LIMIT = "8g"

GenerateFn = Callable[..., Awaitable[Sample | list[Sample]]]

# One task table per resolved data root (train and eval rows use different
# roots). Loading a release walks every task directory, so it happens once per
# process; concurrent first calls wait on a per-loop lock instead of loading
# the same root many times at rollout start.
_TASK_TABLES: dict[str, dict[str, SpreadsheetBenchV1Task]] = {}
_TASK_TABLE_LOCKS: WeakKeyDictionary[
    asyncio.AbstractEventLoop, asyncio.Lock
] = WeakKeyDictionary()

# Replay + recalc boot one-shot containers outside the environment's boot
# semaphore; cap them separately so a batch of episodes finishing together
# cannot stampede the Docker host. Per-loop for the same reason as
# full_corpus_env's boot semaphores.
_REPLAY_SEMAPHORES: WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[int, asyncio.Semaphore]
] = WeakKeyDictionary()

_FALLBACK_GENERATES: dict[str, GenerateFn] = {}


def ssb1_agent_config_from_args(args: Any) -> AgentConfig:
    return AgentConfig(
        step_limit=int(getattr(args, "officeqa_ssb1_step_limit", 50)),
        action_block_language=str(
            getattr(args, "officeqa_ssb1_action_block_language", "bash")
        ),
    )


def ssb1_environment_config_from_args(args: Any, data_root: Path) -> EnvironmentConfig:
    # corpus_dir is required by the shared EnvironmentConfig but the SSB1
    # environment never mounts it (the data root also holds golden workbooks);
    # pass the data root to keep get_info() paths meaningful.
    return EnvironmentConfig(
        corpus_dir=data_root,
        pdf_dir=None,
        workspace_root=Path(_require_arg(args, "officeqa_workspace_root")),
        sandbox_image=str(
            getattr(args, "officeqa_ssb1_sandbox_image", DEFAULT_SANDBOX_IMAGE)
        ),
        command_timeout_sec=int(getattr(args, "officeqa_ssb1_command_timeout_sec", 120)),
        max_observation_chars=int(
            getattr(args, "officeqa_max_observation_chars", 12_000)
        ),
        sandbox_runtime="docker",
        max_concurrent_sandbox_boots=int(
            getattr(args, "officeqa_max_concurrent_sandbox_boots", 4)
        ),
    )


class _TrainingSSB1Environment(SpreadsheetBenchV1Environment):
    """Episode sandbox with a hard memory cap.

    The agent executes arbitrary code inside the episode container; on a
    training host a single unbounded allocation can trip Ray's node memory
    monitor and kill the whole run. Mirrors the parent's
    ``_start_command_container`` docker invocation with resource limits
    inserted — keep in sync with
    ``spreadsheetbench.SpreadsheetBenchEnvironment`` when that changes.
    """

    memory_limit: str = DEFAULT_CONTAINER_MEMORY_LIMIT

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
            "--memory",
            self.memory_limit,
            "--memory-swap",
            self.memory_limit,
            "--pids-limit",
            "512",
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


def _fallback_generate(args: Any) -> GenerateFn:
    """Resolve (and cache) the hook that handles non-SSB1 rows."""
    dotted = str(
        getattr(
            args,
            "officeqa_ssb1_fallback_generate_path",
            DEFAULT_FALLBACK_GENERATE_PATH,
        )
    )
    if dotted not in _FALLBACK_GENERATES:
        module_name, _, attribute = dotted.rpartition(".")
        if not module_name:
            raise ValueError(
                f"officeqa_ssb1_fallback_generate_path is not a dotted path: {dotted!r}"
            )
        _FALLBACK_GENERATES[dotted] = getattr(
            importlib.import_module(module_name), attribute
        )
    return _FALLBACK_GENERATES[dotted]


def _task_table_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _TASK_TABLE_LOCKS.get(loop)
    if lock is None:
        lock = _TASK_TABLE_LOCKS[loop] = asyncio.Lock()
    return lock


async def _load_task(metadata: dict[str, Any]) -> tuple[SpreadsheetBenchV1Task, Path]:
    """Resolve one task from row metadata via its release's dataset.json."""
    data_root = str(metadata.get("data_root") or "").strip()
    task_id = str(metadata.get("task_id") or "").strip()
    if not data_root or not task_id:
        raise ValueError("ssb1 sample metadata must carry data_root and task_id")
    root = Path(data_root).expanduser().resolve()
    key = str(root)
    if key not in _TASK_TABLES:
        async with _task_table_lock():
            if key not in _TASK_TABLES:
                tasks = await asyncio.to_thread(load_spreadsheetbench_v1_tasks, root)
                _TASK_TABLES[key] = {task.task_id: task for task in tasks}
                logger.info("Loaded %d SSB1 tasks from %s", len(tasks), root)
    task = _TASK_TABLES[key].get(task_id)
    if task is None:
        raise ValueError(f"SSB1 task {task_id!r} is not present in {root}")
    return task, root


def _replay_semaphore(limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    semaphores = _REPLAY_SEMAPHORES.setdefault(loop, {})
    if limit not in semaphores:
        semaphores[limit] = asyncio.Semaphore(limit)
    return semaphores[limit]


async def _remove_container(name: str) -> None:
    """Best-effort force-removal; a no-op error when --rm already cleaned up."""
    try:
        await asyncio.to_thread(
            subprocess.run,
            ["docker", "rm", "-f", name],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        logger.exception("Failed to remove replay container %s", name)


async def _replay_case(
    solution_path: Path,
    case: SpreadsheetBenchV1Case,
    workspace: Path,
    *,
    visible_input_name: str,
    visible_output_name: str,
    sandbox_image: str,
    timeout_sec: float,
    memory_limit: str = DEFAULT_CONTAINER_MEMORY_LIMIT,
) -> dict[str, Any]:
    """Run solution.py on one case in a fresh, label-safe sandbox.

    Training port of ``spreadsheetbench_v1_scoring.replay_solution`` with the
    identical container contract (network none, read-only root, tmpfs /tmp,
    the case staged under the visible case-1 aliases), plus two hardenings the
    eval helper lacks:

    - the scratch workspace is a caller-provided directory under the
      DooD-visible workspace root, because a tempfile workspace cannot be
      volume-mounted from inside the training container;
    - the container is named, memory-capped, and force-removed in a finally
      block: ``subprocess.run(timeout=...)`` kills only the docker CLIENT, so
      without the explicit removal a hung or runaway solution.py survives its
      timeout as an orphan with unbounded memory (the 2026-08-06 host-OOM
      incident). A solution killed by the memory cap exits nonzero and the
      case grades as failed, which is the intended reward semantics.
    """
    workspace.mkdir(parents=True, exist_ok=False)
    shutil.copy2(solution_path, workspace / "solution.py")
    shutil.copy2(case.input_path, workspace / visible_input_name)
    container_name = f"ssb1-replay-{uuid.uuid4().hex[:12]}"
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        "none",
        "--read-only",
        "--memory",
        memory_limit,
        "--memory-swap",
        memory_limit,
        "--pids-limit",
        "256",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "-e",
        f"SSB_INPUT=/workspace/{visible_input_name}",
        "-e",
        f"SSB_OUTPUT=/workspace/{visible_output_name}",
        "-v",
        f"{workspace}:/workspace",
        "-w",
        "/workspace",
        sandbox_image,
        "python3",
        "/workspace/solution.py",
    ]
    try:
        try:
            process = await asyncio.to_thread(
                subprocess.run,
                command,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            timed_out = False
            exit_code: int | None = process.returncode
            stdout = decode_subprocess_output(process.stdout)
            stderr = decode_subprocess_output(process.stderr)
        except subprocess.TimeoutExpired as error:
            timed_out = True
            exit_code = None
            stdout = decode_subprocess_output(error.stdout)
            stderr = decode_subprocess_output(error.stderr)
    finally:
        # Covers the timeout path AND cancellation mid-replay; harmless when
        # the container already exited under --rm.
        await _remove_container(container_name)

    return {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "output_produced": (workspace / visible_output_name).is_file(),
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
    }


async def _score_submission(
    args: Any,
    task: SpreadsheetBenchV1Task,
    solution_path: Path,
    scratch_root: Path,
) -> dict[str, Any]:
    """Replay solution.py on every case, recalculate, and grade officially."""
    sandbox_image = str(
        getattr(args, "officeqa_ssb1_sandbox_image", DEFAULT_SANDBOX_IMAGE)
    )
    replay_timeout = float(getattr(args, "officeqa_ssb1_replay_timeout_sec", 600.0))
    recalc_timeout = float(getattr(args, "officeqa_ssb1_recalc_timeout_sec", 600.0))
    skip_recalc = bool(getattr(args, "officeqa_ssb1_skip_recalc", False))
    replay_limit = int(
        getattr(
            args,
            "officeqa_ssb1_max_concurrent_replays",
            getattr(args, "officeqa_max_concurrent_sandbox_boots", 4),
        )
    )
    memory_limit = str(
        getattr(
            args,
            "officeqa_ssb1_container_memory_limit",
            DEFAULT_CONTAINER_MEMORY_LIMIT,
        )
    )

    case_metrics: list[dict[str, Any]] = []
    for case in task.cases:
        workspace = scratch_root / f"case{case.index}"
        output_path = workspace / task.output_name
        async with _replay_semaphore(replay_limit):
            replay = await _replay_case(
                solution_path,
                case,
                workspace,
                visible_input_name=task.input_name,
                visible_output_name=task.output_name,
                sandbox_image=sandbox_image,
                timeout_sec=replay_timeout,
                memory_limit=memory_limit,
            )
            # Skipping the recalc would zero-reward every solution that writes
            # formulas instead of cached values; keep it inside the semaphore
            # so replay+recalc count as one unit of container pressure.
            recalc_error = ""
            if replay["output_produced"] and not skip_recalc:
                try:
                    await recalc_with_libreoffice(
                        output_path,
                        sandbox_image=sandbox_image,
                        timeout_sec=recalc_timeout,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    recalc_error = str(error)
        metrics = await asyncio.to_thread(
            compare_v1_output,
            case.golden_path,
            output_path,
            task.answer_position,
        )
        metrics["case_index"] = case.index
        metrics["replay"] = replay
        if recalc_error:
            metrics["recalc_error"] = recalc_error
        case_metrics.append(metrics)
    return aggregate_v1_metrics(case_metrics)


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Bound the per-sample metadata: keep the scores and per-case
    diagnostics, drop workbook-sized detail (cell error lists, stdout)."""
    cases = []
    for case in metrics.get("case_metrics", []):
        replay = case.get("replay") or {}
        compact = {
            "case_index": case.get("case_index"),
            "accuracy": case.get("accuracy"),
            "cell_accuracy": case.get("cell_accuracy"),
            "cells": case.get("cells"),
            "error": case.get("error", ""),
            "replay_exit_code": replay.get("exit_code"),
            "replay_timed_out": replay.get("timed_out"),
            "output_produced": replay.get("output_produced"),
            "stderr_tail": (replay.get("stderr") or "")[-500:],
        }
        if case.get("recalc_error"):
            compact["recalc_error"] = str(case["recalc_error"])[-500:]
        cases.append(compact)
    return {
        "hard_restriction": metrics["hard_restriction"],
        "soft_restriction": metrics["soft_restriction"],
        "cell_accuracy": metrics["cell_accuracy"],
        "cells": metrics["cells"],
        "test_case_results": metrics["test_case_results"],
        "cases": cases,
    }


async def generate(
    args: Any,
    base_sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    """Run one SSB1 agent rollout, grade it officially, and export samples."""
    metadata = dict(base_sample.metadata or {})
    if metadata.get("benchmark") != "ssb1":
        return await _fallback_generate(args)(
            args, base_sample, sampling_params, evaluation=evaluation
        )

    try:
        task, data_root = await _load_task(metadata)
    except (TypeError, ValueError, FileNotFoundError) as error:
        return _abort(base_sample, f"invalid_task: {error}")

    run_id = _make_run_id(base_sample)
    routing_key = (
        base_sample.session_id
        if getattr(args, "router_policy", None) == "consistent_hashing"
        else None
    )
    try:
        backend = _create_backend(args, evaluation=evaluation, routing_key=routing_key)
        agent_config = ssb1_agent_config_from_args(args)
        agent = OfficeQAAgent(
            backend,
            agent_config,
            sampling_params=sampling_params,
            system_prompt=build_spreadsheet_v1_system_prompt(
                agent_config.action_block_language
            ),
        )
        environment_config = ssb1_environment_config_from_args(args, data_root)
    except (TypeError, ValueError) as error:
        return _abort(base_sample, f"invalid_rollout_config: {error}", run_id=run_id)

    # Per-turn OPD (the hill-climb arms) on SSB1: the same scorer, turn plans
    # and sample construction as the OfficeQA path in
    # :mod:`officeqa_rl.training.opd_rollout`. Selected by the yaml knob
    # ``officeqa_opd_per_turn``; without it the per-sample path below stays
    # exactly as before.
    scorer = None
    modes = None
    watch_task = None
    if (
        not evaluation
        and _teacher_scoring_enabled(args)
        and bool(getattr(args, "officeqa_opd_per_turn", False))
    ):
        from officeqa_rl.training.vertex_teacher_client import get_teacher_client

        try:
            modes = parse_per_turn_modes(args)
            client = get_teacher_client(args)
        except (TypeError, ValueError) as error:
            return _abort(
                base_sample, f"invalid_rollout_config: {error}", run_id=run_id
            )
        scorer = _TurnScorer(
            client,
            affinity_key=run_id,
            tokenizer=backend.tokenizer,
            modes=modes,
        )
        if modes.teacher_think:
            backend.on_turn_prompt = scorer.prestart_think
        watch_task = asyncio.create_task(scorer.watch(backend))

    solution_path: Path | None = None
    scratch_root: Path | None = None
    try:
        try:
            environment = _TrainingSSB1Environment(
                environment_config, task, run_id=run_id
            )
            environment.memory_limit = str(
                getattr(
                    args,
                    "officeqa_ssb1_container_memory_limit",
                    DEFAULT_CONTAINER_MEMORY_LIMIT,
                )
            )
            async with environment as env:
                result = await agent.run(env)
                if result.status == "submitted":
                    # Copy solution.py out before the container goes away; the
                    # replays must never touch the agent's live workspace.
                    scratch_root = env.workspace.with_name(
                        env.workspace.name + "-score"
                    )
                    scratch_root.mkdir(parents=True, exist_ok=False)
                    solution_path = scratch_root / "solution.py"
                    shutil.copy2(env.solution_host_path, solution_path)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # sandbox/workspace failures must not spread
            if scorer is not None:
                await scorer.cancel()
            return _abort(
                base_sample,
                f"environment_failure: {type(error).__name__}: {error}",
                run_id=run_id,
            )
    finally:
        if watch_task is not None:
            watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watch_task

    if scorer is not None:
        # Turns recorded after the watcher's last poll (always including the
        # final turn) get their scoring fired here.
        scorer.fire_new(backend.session_turns)

    terminal_status = _STATUS_BY_EXIT.get(result.status)
    if terminal_status is None:
        if scorer is not None:
            await scorer.cancel()
        return _abort(base_sample, f"agent_failure: {result.status}", run_id=run_id)

    reward = 0.0
    metrics: dict[str, Any] | None = None
    if result.status == "submitted" and solution_path is not None:
        try:
            metrics = await _score_submission(args, task, solution_path, scratch_root)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # replay infrastructure, not policy, failed
            if scorer is not None:
                await scorer.cancel()
            return _abort(
                base_sample,
                f"scoring_failure: {type(error).__name__}: {error}",
                run_id=run_id,
            )
        finally:
            if scratch_root is not None and not bool(
                getattr(args, "officeqa_ssb1_keep_replay_dirs", False)
            ):
                shutil.rmtree(scratch_root, ignore_errors=True)
        reward = float(
            metrics["hard_restriction"] if evaluation else metrics["soft_restriction"]
        )

    extra_metadata = {
        **metadata,
        "uid": task.uid,
        "run_id": run_id,
        "submission": result.submission,
        "exit_status": result.status,
        "steps": result.steps,
        "ssb1_reward_metric": "hard_restriction" if evaluation else "soft_restriction",
        # Exact generated tokens for the whole episode, recorded on every
        # path (training and eval, OPD or not) so the metrics hook can log
        # episode/output_tokens_mean. It cannot be recovered from
        # response_len/mean x steps: that estimate misses turns dropped by
        # truncation or REALIGN and reads ~20% high (measured on OfficeQA).
        "episode_output_tokens": _episode_output_tokens(backend.session_turns),
        # Turns the think budget force-closed. metrics.py only logs
        # episode/forced_close_* when this key is present; the OfficeQA
        # paths attach it and this one did not, which is why a Qwen run
        # whose budget never fired showed nothing at all (2026-09-09).
        "forced_close_turns": _forced_close_turns(backend.session_turns),
    }
    if metrics is not None:
        extra_metadata["ssb1_metrics"] = _compact_metrics(metrics)

    if scorer is not None:
        # Teacher-scoring failures are deliberately fatal, exactly as in the
        # OfficeQA per-turn path: log the reason, then stop the run. The same
        # goes for score/sample misalignment inside the construction.
        try:
            scored_turns = await scorer.results()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await _log_teacher_failure(args, base_sample, run_id, error)
            raise
        try:
            samples, abort_reason = build_scored_samples(
                modes,
                session_turns=backend.session_turns,
                scored_turns=scored_turns,
                base_sample=base_sample,
                reward=reward,
                tokenizer=backend.tokenizer,
                extra_metadata=extra_metadata,
                max_sample_tokens=backend.max_context_len,
            )
        except Exception as error:
            await _log_teacher_failure(args, base_sample, run_id, error)
            raise
        if abort_reason is not None:
            return _abort(base_sample, abort_reason, run_id=run_id)
    else:
        samples = build_training_samples(
            session_turns=backend.session_turns,
            base_sample=base_sample,
            reward=reward,
            tokenizer=backend.tokenizer,
            extra_metadata=extra_metadata,
            max_sample_tokens=backend.max_context_len,
        )
        if not samples:
            return _abort(
                base_sample,
                "empty_trajectory: no trained model tokens were recorded",
                run_id=run_id,
            )

    for sample in samples:
        sample.status = (
            Sample.Status.TRUNCATED
            if sample.metadata.get("truncated")
            else terminal_status
        )
        # Which scoring path this episode actually took. The per-turn branch
        # above needs all three of these; only it reaches build_scored_samples
        # and hence _apply_action_scores, so a run whose per-turn knobs are set
        # in the yaml but which still shows scorer=false has failed the gate
        # rather than the mode logic.
        if isinstance(sample.metadata, dict):
            sample.metadata["opd_path"] = {
                "scorer": scorer is not None,
                "evaluation": bool(evaluation),
                "teacher_enabled": bool(_teacher_scoring_enabled(args)),
                "per_turn": bool(getattr(args, "officeqa_opd_per_turn", False)),
                "use_opd": bool(getattr(args, "use_opd", False)),
                "opd_type": str(getattr(args, "opd_type", None)),
            }

    # Same OPD semantics as the OfficeQA per-sample path: eval rollouts skip
    # the teacher, and a terminal teacher-scoring failure is fatal after being
    # logged, never a silent zero-mask.
    if not evaluation and scorer is None and _teacher_scoring_enabled(args):
        try:
            await _attach_teacher_log_probs(args, samples)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await _log_teacher_failure(args, base_sample, run_id, error)
            raise

    samples[0].prefix_cache_info.cached_tokens = backend.cached_prompt_tokens
    samples[0].prefix_cache_info.total_prompt_tokens = backend.total_prompt_tokens

    return samples[0] if len(samples) == 1 else samples


__all__ = [
    "generate",
    "ssb1_agent_config_from_args",
    "ssb1_environment_config_from_args",
]
