"""Online teacher-trace SFT rollout for SpreadsheetBench 1.

SSB1 twin of :mod:`officeqa_rl.training.teacher_sft_rollout`, selected with
``--rollout-function-path
officeqa_rl.training.ssb1_teacher_sft_rollout.generate_rollout``. A frozen
TEACHER runs the SSB1 agent loop inside the training loop, its finished
episodes become per-assistant-turn samples, and the student trains on them
with hard-label cross-entropy (``--loss-type sft_loss --loss-mask-type
qwen3_5 --disable-compute-advantages-and-returns --debug-train-only``). No
student inference engines run.

Everything benchmark-independent is reused from the OfficeQA module: the
endpoint pool and its per-endpoint concurrency cap, the tokenizer/mask
generator state, the turn-to-sample conversion, and the batch contract (each
call emits exactly ``rollout_batch_size * n_samples_per_prompt`` episodes,
replacing filtered or failed ones). Settings therefore keep the same
``officeqa_sft_*`` YAML keys, including ``officeqa_sft_filter: all``.

What differs is one episode: SSB1 tasks come from the manifest rows rather
than the OfficeQA CSV, the environment is the Docker spreadsheet sandbox, and
grading replays the agent's ``solution.py`` through LibreOffice against every
test case. ``officeqa_sft_trace_dir`` receives one JSON file per completed
episode, named ``<uid>.s<sample index>.json`` and shaped like
``scripts/run_spreadsheetbench_v1_eval.py`` results (same keys, including
``metrics`` and ``messages``), so the SSB1 analysis tooling reads a training
run's traces directly and the teacher compute stays reusable offline.

``officeqa_sft_filter: correct`` keeps episodes whose replay scores
hard_restriction == 1 (every designated cell right in every case); the reward
carried on the samples is the soft restriction, matching the SSB1 RL path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.types import Sample

from officeqa_rl.agent import OfficeQAAgent
from officeqa_rl.spreadsheetbench_v1 import build_spreadsheet_v1_system_prompt
from officeqa_rl.training.rollout import _STATUS_BY_EXIT, _make_run_id
from officeqa_rl.training.ssb1_rollout import (
    DEFAULT_CONTAINER_MEMORY_LIMIT,
    _load_task,
    _score_submission,
    _TrainingSSB1Environment,
    ssb1_agent_config_from_args,
    ssb1_environment_config_from_args,
)
from officeqa_rl.training import teacher_sft_rollout as _shared
from officeqa_rl.training.teacher_sft_rollout import (
    _episode_to_samples,
    _filter_mode,
    _sampling_params,
)

logger = logging.getLogger(__name__)


async def _persist_trace(
    args: Any,
    base_sample: Sample,
    task: Any,
    result: Any,
    metrics: dict[str, Any] | None,
    run_id: str,
    teacher_model: str,
) -> None:
    """Write one finished episode as ``<uid>.s<index>.json``.

    The payload matches ``scripts/run_spreadsheetbench_v1_eval.py`` so offline
    tooling cannot tell a training trace from an eval one. Every completed
    episode is written, including ones ``officeqa_sft_filter: correct``
    rejects — the teacher already paid for them. Persistence failures only
    log; they never cost the episode.
    """
    trace_dir = getattr(args, "officeqa_sft_trace_dir", None)
    if not trace_dir:
        return
    path = Path(trace_dir) / f"{task.uid}.s{base_sample.index}.json"
    payload = {
        "uid": task.uid,
        "task_id": task.task_id,
        "instruction_type": task.instruction_type,
        "question": task.instruction,
        "submission": result.submission,
        "status": result.status,
        "steps": result.steps,
        # The eval scores an episode by hard restriction; the training reward
        # (soft) rides along in metrics, so both conventions are recoverable.
        "score": (metrics or {}).get("hard_restriction"),
        "token_usage": (result.metadata or {}).get("token_usage"),
        "metrics": metrics,
        "solution_file": None,
        "messages": result.messages,
        "metadata": {
            **(result.metadata or {}),
            "case_count": len(task.cases),
            "run_id": run_id,
            "sample_index": base_sample.index,
            "teacher_model": teacher_model,
            "source": "ssb1_teacher_sft_rollout",
        },
    }

    def _write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)

    try:
        await asyncio.to_thread(_write)
    except Exception:
        logger.exception("ssb1 teacher SFT rollout: failed to persist trace %s", path)


async def _run_episode(
    args: Any,
    state: _shared._RolloutState,
    base_sample: Sample,
) -> tuple[list[Sample] | None, str, bool]:
    """Run one teacher episode on an SSB1 task.

    Returns ``(samples, outcome, passed)`` with outcome ``"kept"`` |
    ``"filtered"`` | ``"failed"``, exactly like the OfficeQA runner, so the
    shared batch driver can host either.
    """
    metadata = dict(base_sample.metadata or {})
    if metadata.get("benchmark") != "ssb1":
        logger.warning(
            "ssb1 teacher SFT rollout: sample %s is not an SSB1 row, dropped",
            base_sample.index,
        )
        return None, "failed", False

    async with state.pool.lease() as backend:
        try:
            task, data_root = await _load_task(metadata)
            agent_config = ssb1_agent_config_from_args(args)
            environment_config = ssb1_environment_config_from_args(args, data_root)
        except (TypeError, ValueError, FileNotFoundError) as error:
            logger.warning(
                "ssb1 teacher SFT rollout: invalid task, episode dropped: %s", error
            )
            return None, "failed", False

        run_id = _make_run_id(base_sample)
        agent = OfficeQAAgent(
            backend,
            agent_config,
            sampling_params=_sampling_params(args),
            system_prompt=build_spreadsheet_v1_system_prompt(
                agent_config.action_block_language
            ),
        )
        solution_path: Path | None = None
        scratch_root: Path | None = None
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
                    # Copy solution.py out before the container goes away: the
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
            logger.warning(
                "ssb1 teacher SFT rollout: environment failure, episode dropped "
                "(run_id=%s): %s: %s",
                run_id,
                type(error).__name__,
                error,
            )
            if scratch_root is not None:
                shutil.rmtree(scratch_root, ignore_errors=True)
            return None, "failed", False

    if result.status not in _STATUS_BY_EXIT:
        logger.warning(
            "ssb1 teacher SFT rollout: agent failure %r, episode dropped (run_id=%s)",
            result.status,
            run_id,
        )
        return None, "failed", False

    reward = 0.0
    passed = False
    metrics: dict[str, Any] | None = None
    if result.status == "submitted" and solution_path is not None:
        try:
            metrics = await _score_submission(args, task, solution_path, scratch_root)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # replay infrastructure, not policy, failed
            logger.warning(
                "ssb1 teacher SFT rollout: scoring failure, episode dropped "
                "(run_id=%s): %s: %s",
                run_id,
                type(error).__name__,
                error,
            )
            return None, "failed", False
        finally:
            if scratch_root is not None and not bool(
                getattr(args, "officeqa_ssb1_keep_replay_dirs", False)
            ):
                shutil.rmtree(scratch_root, ignore_errors=True)
        # Same convention as the SSB1 RL path: training carries the soft
        # restriction, while "correct" means the strict per-case metric.
        reward = float(metrics["soft_restriction"])
        passed = float(metrics["hard_restriction"]) == 1.0
    elif scratch_root is not None:
        shutil.rmtree(scratch_root, ignore_errors=True)

    await _persist_trace(
        args, base_sample, task, result, metrics, run_id, backend.model_name
    )

    if _filter_mode(args) == "correct" and not passed:
        return None, "filtered", passed

    samples = _episode_to_samples(
        args, state, base_sample, task, result, reward, run_id, backend
    )
    if not samples:
        logger.warning(
            "ssb1 teacher SFT rollout: no trainable turns, episode dropped "
            "(run_id=%s)",
            run_id,
        )
        return None, "failed", passed
    return samples, "kept", passed


def generate_rollout(
    args: Any, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput:
    """Entry point: the shared batch driver with the SSB1 episode runner."""
    return _shared.generate_rollout(
        args, rollout_id, data_source, evaluation, runner=_run_episode
    )


__all__ = ["generate_rollout"]
