"""Online teacher-trace SFT rollout for JobBench.

JobBench twin of :mod:`officeqa_rl.training.teacher_sft_rollout`, selected with
``--rollout-function-path
officeqa_rl.training.jobbench_teacher_sft_rollout.generate_rollout``. A frozen
TEACHER runs the JobBench agent loop inside the training loop, its finished
episodes become per-assistant-turn samples, and the student trains on them with
hard-label cross-entropy (``--loss-type sft_loss --loss-mask-type qwen3_5
--disable-compute-advantages-and-returns --debug-train-only``). No student
inference engines run.

Everything benchmark-independent is reused from the OfficeQA module: the
endpoint pool and its per-endpoint concurrency cap, the tokenizer/mask
generator state, the turn-to-sample conversion, and the batch contract (each
call emits exactly ``rollout_batch_size * n_samples_per_prompt`` episodes,
replacing filtered or failed ones). Settings therefore keep the same
``officeqa_sft_*`` YAML keys.

What differs is one episode: JobBench tasks are loaded by uid from the release
directory, the agent is mini-swe-agent in a per-episode Docker sandbox, and
grading is an LLM judge over the task's weighted rubric rather than a
programmatic replay.

Two JobBench-specific adaptations are worth knowing about.

* **Episode shape.** :class:`~officeqa_rl.jobbench.agent.JobBenchEpisode`
  exposes ``exit_status`` and a top-level ``generated_turns``, while the shared
  ``_episode_to_samples`` reads ``result.status`` and
  ``result.metadata["generated_turns"]`` and looks the status up in
  ``rollout._STATUS_BY_EXIT`` (whose keys are the lowercase OfficeQA spellings).
  :class:`_SharedEpisodeView` adapts both without copying the transcript.

* **``officeqa_sft_filter: correct`` is not usable here.** JobBench has no
  binary pass: the reward is a continuous normalized rubric score, and on the
  main split even the 397B teacher averages 0.173, so a correctness filter
  would discard nearly every episode the teacher paid for. This module treats
  ``correct`` as an error rather than silently keeping nothing; use ``all``, or
  ``officeqa_sft_min_score`` for a threshold.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

from officeqa_rl.jobbench.agent import run_jobbench_episode
from officeqa_rl.jobbench.dataset import load_task_by_uid
from officeqa_rl.jobbench.judge import judge_episode
from officeqa_rl.training.rollout import _STATUS_BY_EXIT, _make_run_id
from officeqa_rl.training.jobbench_rollout import (
    OUTPUT_DIR_NAME,
    _boot_semaphore,
    _compact_score,
    jobbench_agent_config_from_args,
    jobbench_environment_config_from_args,
    jobbench_judge_config_from_args,
)
from officeqa_rl.training import teacher_sft_rollout as _shared
from officeqa_rl.training.teacher_sft_rollout import (
    _episode_to_samples,
    _filter_mode,
    _sampling_params,
)

logger = logging.getLogger(__name__)

#: JobBench exit statuses -> the lowercase spellings ``rollout._STATUS_BY_EXIT``
#: knows. Anything absent here is an agent failure and drops the episode, which
#: matches the JobBench RL path's ``_STATUS_BY_EXIT.get(...) is None`` check.
_EXIT_ALIAS = {
    "Submitted": "submitted",
    "LimitsExceeded": "limits_exceeded",
    "TimeExceeded": "limits_exceeded",
    "RepeatedFormatError": "limits_exceeded",
}


class _SharedEpisodeView:
    """Present a JobBenchEpisode the way ``_episode_to_samples`` expects.

    A thin attribute view rather than a copy: the transcript of a long JobBench
    episode is large, and duplicating it once per episode would cost real host
    memory in a rollout batch.
    """

    #: Attributes the shared module reads off an episode. Kept as an explicit
    #: list so a future field added there fails loudly at construction rather
    #: than with an AttributeError deep inside sample building, which is how
    #: ``submission`` was first missed.
    _REQUIRED = ("messages", "metadata", "status", "steps", "submission")

    def __init__(self, episode: Any, status: str) -> None:
        self._episode = episode
        self.status = status
        self.messages = episode.messages
        self.steps = episode.steps
        self.submission = episode.submission
        self.metadata = {
            **(episode.metadata or {}),
            "generated_turns": [
                {"finish_reason": turn.finish_reason} for turn in episode.generated_turns
            ],
        }
        missing = [name for name in self._REQUIRED if not hasattr(self, name)]
        if missing:
            raise AttributeError(
                f"_SharedEpisodeView is missing {missing}; teacher_sft_rollout "
                "reads these off the episode"
            )


async def _persist_trace(
    args: Any,
    base_sample: Sample,
    task: Any,
    episode: Any,
    score: Any,
    run_id: str,
    teacher_model: str,
) -> None:
    """Write one finished episode as ``<uid>.s<index>.json``.

    The payload matches ``scripts/run_jobbench_eval.py`` so offline tooling
    cannot tell a training trace from an eval one. Every completed episode is
    written, including ones a filter later rejects — the teacher already paid
    for them. Persistence failures only log; they never cost the episode.
    """
    trace_dir = getattr(args, "officeqa_sft_trace_dir", None)
    if not trace_dir:
        return
    path = Path(trace_dir) / f"{task.uid}.s{base_sample.index}.json"
    payload = {
        "uid": task.uid,
        "benchmark": "jobbench",
        "split": task.split,
        "occupation": task.occupation,
        "task_id": task.task_id,
        "score": None if score is None else score.normalized_score,
        "graded": None if score is None else score.graded,
        "judge_failures": None if score is None else score.judge_failures,
        "jobbench_metrics": None if score is None else _compact_score(score),
        "exit_status": episode.exit_status,
        "submitted": episode.exit_status == "Submitted",
        "steps": episode.steps,
        "output_files": episode.output_files,
        "messages": episode.messages,
        "metadata": {
            **(episode.metadata or {}),
            "run_id": run_id,
            "sample_index": base_sample.index,
            "teacher_model": teacher_model,
            "source": "jobbench_teacher_sft_rollout",
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
        logger.exception("jobbench teacher SFT rollout: failed to persist trace %s", path)


def _min_score(args: Any) -> float:
    """Optional score floor for keeping an episode (0.0 keeps everything)."""
    return float(getattr(args, "officeqa_sft_min_score", 0.0) or 0.0)


async def _run_episode(
    args: Any,
    state: _shared._RolloutState,
    base_sample: Sample,
) -> tuple[list[Sample] | None, str, bool]:
    """Run one teacher episode on a JobBench task.

    Returns ``(samples, outcome, passed)`` with outcome ``"kept"`` |
    ``"filtered"`` | ``"failed"``, exactly like the OfficeQA and SSB1 runners,
    so the shared batch driver can host any of them.
    """
    metadata = dict(base_sample.metadata or {})
    if metadata.get("benchmark") != "jobbench":
        logger.warning(
            "jobbench teacher SFT rollout: sample %s is not a JobBench row, dropped",
            base_sample.index,
        )
        return None, "failed", False

    if _filter_mode(args) == "correct":
        raise ValueError(
            "officeqa_sft_filter: correct is not meaningful on JobBench -- the "
            "reward is a continuous rubric score with no binary pass. Use 'all', "
            "or set officeqa_sft_min_score for a threshold."
        )

    uid = metadata.get("uid")
    data_root = metadata.get("data_root")
    if not uid or not data_root:
        logger.warning(
            "jobbench teacher SFT rollout: row lacks uid/data_root, episode dropped"
        )
        return None, "failed", False

    run_id = _make_run_id(base_sample)
    async with state.pool.lease() as backend:
        try:
            task = await asyncio.to_thread(
                load_task_by_uid, Path(data_root).expanduser().resolve(), uid
            )
            agent_config = jobbench_agent_config_from_args(args)
            env_config = jobbench_environment_config_from_args(args)
            judge_config = jobbench_judge_config_from_args(args)
        except (TypeError, ValueError, FileNotFoundError) as error:
            logger.warning(
                "jobbench teacher SFT rollout: invalid task/config, episode dropped: %s",
                error,
            )
            return None, "failed", False

        boot_limit = int(getattr(args, "officeqa_max_concurrent_sandbox_boots", 4))
        try:
            episode = await run_jobbench_episode(
                task,
                backend,
                agent_config,
                env_config,
                run_id=run_id,
                boot_semaphore=_boot_semaphore(max(1, boot_limit)),
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # sandbox/workspace failures must not spread
            logger.warning(
                "jobbench teacher SFT rollout: environment failure, episode dropped "
                "(run_id=%s): %s: %s",
                run_id,
                type(error).__name__,
                error,
            )
            return None, "failed", False

        alias = _EXIT_ALIAS.get(episode.exit_status)
        if alias is None or alias not in _STATUS_BY_EXIT:
            logger.warning(
                "jobbench teacher SFT rollout: agent failure %r, episode dropped "
                "(run_id=%s)",
                episode.exit_status,
                run_id,
            )
            return None, "failed", False

        # Grade for the trace payload and the sample reward. A judge outage
        # must not cost the teacher episode: the SFT loss is hard-label cross
        # entropy on the teacher's own turns and does not use the reward, so an
        # ungraded episode still trains correctly. It is kept with reward 0.0
        # and flagged, unlike the RL path where an ungradeable episode carries
        # no signal at all.
        score = None
        try:
            score = await judge_episode(
                task, episode.workspace / OUTPUT_DIR_NAME, judge_config
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # judging infrastructure, not the teacher
            logger.warning(
                "jobbench teacher SFT rollout: grading failed, episode kept "
                "unscored (run_id=%s): %s: %s",
                run_id,
                type(error).__name__,
                error,
            )

        reward = float(score.normalized_score) if score is not None and score.graded else 0.0
        await _persist_trace(
            args, base_sample, task, episode, score, run_id, backend.model_name
        )

    floor = _min_score(args)
    if floor > 0.0 and reward < floor:
        return None, "filtered", False

    samples = _episode_to_samples(
        args,
        state,
        base_sample,
        task,
        _SharedEpisodeView(episode, alias),
        reward,
        run_id,
        backend,
    )
    if not samples:
        logger.warning(
            "jobbench teacher SFT rollout: no trainable turns, episode dropped "
            "(run_id=%s)",
            run_id,
        )
        return None, "failed", False
    for sample in samples:
        if isinstance(sample.metadata, dict):
            sample.metadata.setdefault("uid", task.uid)
            sample.metadata.setdefault("split", task.split)
            sample.metadata["graded"] = bool(score is not None and score.graded)
    return samples, "kept", True


def generate_rollout(
    args: Any, rollout_id: int, data_source: Any, evaluation: bool = False
) -> Any:
    """Entry point: the shared batch driver with the JobBench episode runner."""
    return _shared.generate_rollout(
        args, rollout_id, data_source, evaluation, runner=_run_episode
    )


__all__ = ["generate_rollout"]
