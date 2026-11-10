"""Online teacher-trace SFT rollout for OfficeQA.

This is the ``--rollout-function-path
officeqa_rl.training.teacher_sft_rollout.generate_rollout`` entry point. A
frozen TEACHER model — any SGLang or vLLM server, reached directly over
``http://`` or through a Vertex ``:rawPredict`` URL — runs the full OfficeQA
agent loop inside the slime training loop, and every finished episode becomes
per-assistant-turn SFT samples for hard-label cross-entropy training
(``--loss-type sft_loss --loss-mask-type qwen3_5
--disable-compute-advantages-and-returns --debug-train-only``). No student
inference engines are involved; with ``train_async.py`` the teacher generates
rollout N+1 while the actor trains on rollout N.

Teacher settings ride on ``args`` under an ``officeqa_sft_`` prefix, set
through slime's ``--custom-config-path`` YAML::

    officeqa_sft_teacher_url: "https://...:rawPredict,http://host:port/generate"
    officeqa_sft_teacher_protocol: sglang        # or vllm (/v1/completions)
    officeqa_sft_teacher_model: teacher          # served model id (vllm payload)
    officeqa_sft_concurrency: 16                 # concurrent episodes PER ENDPOINT
    officeqa_sft_teacher_timeout_sec: 600        # per-request wait_for
    officeqa_sft_teacher_max_retries: 3          # transient-failure budget
    officeqa_sft_teacher_quota_retries: 10       # 429/503 budget, counted apart
    officeqa_sft_teacher_backoff_sec: 2.0        # exponential base, jittered
    officeqa_sft_filter: correct                 # or "all"
    officeqa_sft_enable_thinking: true           # optional; chat_template_kwargs
                                                 # for the TEACHER prompt (gemma-4
                                                 # pre-closes its thought channel
                                                 # unless this is true; unset =
                                                 # template default, as before)

``officeqa_sft_teacher_url`` takes a COMMA-SEPARATED list of endpoints serving
the same model, like OPD's ``officeqa_opd_teacher_url``: episodes are spread
over the pool with per-episode affinity, and ``officeqa_sft_concurrency``
caps in-flight episodes PER ENDPOINT (a Vertex 397B deployment tolerates only
a handful at once, so a pool-wide cap would not protect it).

plus the standard ``officeqa_*`` environment keys shared with the RL rollout
(corpus dir, sandbox, step limit — see ``officeqa_rl.training.rollout``).

Batch contract: every call emits exactly ``rollout_batch_size *
n_samples_per_prompt`` whole episodes (slime schedules optimizer steps by
``rollout_id``, so per-episode turn counts may vary freely, but the episode
count per rollout must stay fixed). Episodes rejected by the filter or lost to
teacher/environment failures are replaced by pulling more prompts from the
data source; surplus episodes carry over to the next call. Failures here drop
the episode with a warning instead of aborting the batch — the teacher is
frozen, so a lost episode costs data, not correctness.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.async_utils import run
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from officeqa_rl.agent import OfficeQAAgent
from officeqa_rl.backends.vertex import VertexSGLangGenerationBackend
from officeqa_rl.dataset import task_from_slime_fields
from officeqa_rl.full_corpus_env import FullCorpusEnvironment
from officeqa_rl.scoring import score_answer
from officeqa_rl.sft_samples import (
    per_turn_training_messages,
    reasoning_style_for_loss_mask,
)
from officeqa_rl.training.rollout import (
    _STATUS_BY_EXIT,
    _make_run_id,
    _tolerance,
    agent_config_from_args,
    environment_config_from_args,
)

logger = logging.getLogger(__name__)

_FILTER_MODES = ("correct", "all")

# One episode: (samples, outcome, passed) with outcome in
# "kept" | "filtered" | "failed". Benchmarks supply their own.
EpisodeRunner = Any


def _require_arg(args: Any, name: str) -> Any:
    value = getattr(args, name, None)
    if value is None or value == "":
        raise ValueError(
            f"teacher SFT rollout requires args.{name}; set it in the "
            "--custom-config-path YAML"
        )
    return value


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _filter_mode(args: Any) -> str:
    mode = str(getattr(args, "officeqa_sft_filter", "correct")).strip().lower()
    if mode not in _FILTER_MODES:
        raise ValueError(f"officeqa_sft_filter must be one of {_FILTER_MODES}, got {mode!r}")
    return mode


class _EndpointPool:
    """Teacher endpoints with a concurrency cap that binds per endpoint.

    Mirrors the OPD teacher client: an episode is pinned to one endpoint for
    its whole life (so its prefix cache keeps hitting), and ``per_endpoint``
    is the in-flight cap for each endpoint rather than for the pool. Admission
    is capped at ``per_endpoint * len(backends)`` and each admitted episode
    takes the endpoint with the fewest assigned episodes, so a slow episode on
    one endpoint cannot leave another idle — with that admission cap some
    endpoint always has a free slot, so the per-endpoint acquire never blocks.
    """

    def __init__(
        self, backends: list[VertexSGLangGenerationBackend], per_endpoint: int
    ) -> None:
        if not backends:
            raise ValueError("endpoint pool needs at least one backend")
        self.per_endpoint = max(1, int(per_endpoint))
        self._backends = backends
        self._semaphores = [asyncio.Semaphore(self.per_endpoint) for _ in backends]
        # Episodes assigned (queued or running) per endpoint, the balancing key.
        self._assigned = [0] * len(backends)
        self._admission = asyncio.Semaphore(self.per_endpoint * len(backends))

    @property
    def size(self) -> int:
        return len(self._backends)

    @contextlib.asynccontextmanager
    async def lease(self):
        """Hold one endpoint slot for the duration of an episode."""
        async with self._admission:
            index = min(range(len(self._assigned)), key=self._assigned.__getitem__)
            # Count the assignment before awaiting, or concurrent leases all
            # read the same minimum and pile onto one endpoint.
            self._assigned[index] += 1
            try:
                async with self._semaphores[index]:
                    yield self._backends[index]
            finally:
                self._assigned[index] -= 1


class _RolloutState:
    """Per-process state: tokenizer, mask generator, teacher backend pool, and
    the episode carry-over queue that keeps the emitted episode count fixed."""

    def __init__(self, args: Any) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.mask_generator = MultiTurnLossMaskGenerator(
            self.tokenizer, tokenizer_type=args.loss_mask_type
        )
        # How the target turn's reasoning is handed to the chat template
        # (qwen: inline <think> tags; gemma: reasoning_content field).
        self.reasoning_style = reasoning_style_for_loss_mask(args.loss_mask_type)
        urls = [
            url.strip()
            for url in str(_require_arg(args, "officeqa_sft_teacher_url")).split(",")
            if url.strip()
        ]
        if not urls:
            raise ValueError("officeqa_sft_teacher_url contains no URLs")
        max_response_len = int(_require_arg(args, "rollout_max_response_len"))
        enable_thinking = getattr(args, "officeqa_sft_enable_thinking", None)
        if enable_thinking is not None:
            enable_thinking = _as_bool(enable_thinking, "officeqa_sft_enable_thinking")
        self.backends = [
            VertexSGLangGenerationBackend(
                str(getattr(args, "officeqa_sft_teacher_model", "teacher")),
                base_url=url,
                protocol=str(getattr(args, "officeqa_sft_teacher_protocol", "sglang")),
                tokenizer=self.tokenizer,
                max_tokens=max_response_len,
                max_retries=int(getattr(args, "officeqa_sft_teacher_max_retries", 3)),
                quota_max_retries=int(getattr(args, "officeqa_sft_teacher_quota_retries", 10)),
                backoff_base_sec=float(
                    getattr(args, "officeqa_sft_teacher_backoff_sec", 2.0)
                ),
                request_timeout_sec=float(getattr(args, "officeqa_sft_teacher_timeout_sec", 600)),
                enable_thinking=enable_thinking,
            )
            for url in urls
        ]
        self.concurrency = int(getattr(args, "officeqa_sft_concurrency", 16))
        self.pool = _EndpointPool(self.backends, self.concurrency)
        self.episode_queue: list[list[Sample]] = []


_STATE: _RolloutState | None = None


def _get_state(args: Any) -> _RolloutState:
    global _STATE
    if _STATE is None:
        _STATE = _RolloutState(args)
    return _STATE


def _sampling_params(args: Any) -> dict[str, Any]:
    return {
        "temperature": float(args.rollout_temperature),
        "top_p": float(getattr(args, "rollout_top_p", 1.0)),
    }


async def _persist_trace(
    args: Any, base_sample: Sample, task: Any, result: Any, reward: float
) -> None:
    """Write the raw episode as ``<uid>.s<index>.json`` in the same format as
    ``scripts/run_full_corpus_inference.py`` results, so the teacher compute is
    reusable offline: ``scripts/build_sft_dataset.py`` accepts these traces
    directly (multi-sample siblings, any episode policy). Every completed
    episode is saved — including ones the online ``correct`` filter rejects —
    and persistence failures only log; they never cost the episode."""
    trace_dir = getattr(args, "officeqa_sft_trace_dir", None)
    if not trace_dir:
        return
    path = Path(trace_dir) / f"{task.uid}.s{base_sample.index}.json"
    payload = {
        "uid": task.uid,
        "question": task.question,
        "ground_truth": task.answer,
        "submission": result.submission,
        "status": result.status,
        "steps": result.steps,
        "score": reward,
        "token_usage": (result.metadata or {}).get("token_usage"),
        "messages": result.messages,
        "metadata": result.metadata,
    }

    def _write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)

    try:
        await asyncio.to_thread(_write)
    except Exception:
        logger.exception("teacher SFT rollout: failed to persist trace %s", path)


def _episode_to_samples(
    args: Any,
    state: _RolloutState,
    base_sample: Sample,
    task: Any,
    result: Any,
    reward: float,
    run_id: str,
    backend: VertexSGLangGenerationBackend,
) -> list[Sample]:
    """Convert one finished teacher episode into per-assistant-turn samples."""
    cleaned = [message for message in result.messages if message.get("role") != "exit"]
    if not all(isinstance(message.get("content"), str) for message in cleaned):
        logger.warning(
            "teacher SFT rollout: non-text message content, episode dropped (run_id=%s)",
            run_id,
        )
        return []
    generated_turns = result.metadata.get("generated_turns") or []
    finish_reasons = [turn.get("finish_reason") for turn in generated_turns]
    episode_status = _STATUS_BY_EXIT[result.status]
    max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)

    samples: list[Sample] = []
    for turn_number, (target, messages) in enumerate(
        per_turn_training_messages(
            cleaned, getattr(state, "reasoning_style", "qwen")
        ),
        start=1
    ):
        token_ids, loss_mask = state.mask_generator.get_loss_mask(messages)
        if len(token_ids) != len(loss_mask):
            raise ValueError(
                f"teacher SFT rollout produced mismatched token_ids/loss_mask "
                f"lengths: {len(token_ids)=}, {len(loss_mask)=}"
            )
        response_length = state.mask_generator.get_response_lengths([loss_mask])[0]
        if response_length <= 0:
            continue
        if max_context_len and len(token_ids) > max_context_len:
            logger.warning(
                "teacher SFT rollout: turn %d of run_id=%s spans %d tokens "
                "(> rollout_max_context_len=%d), turn dropped",
                turn_number,
                run_id,
                len(token_ids),
                max_context_len,
            )
            continue
        finish_reason = (
            finish_reasons[turn_number - 1] if turn_number <= len(finish_reasons) else None
        )
        truncated = finish_reason == "length"

        sample = copy.deepcopy(base_sample)
        # Siblings of one episode share rollout_id so slime's optimizer-step
        # split and loss reduction count the episode once, not once per turn.
        sample.rollout_id = base_sample.index
        sample.prompt = messages
        sample.tokens = token_ids
        sample.response_length = response_length
        sample.loss_mask = loss_mask[-response_length:]
        sample.response = state.tokenizer.decode(
            token_ids[-response_length:], skip_special_tokens=False
        )
        sample.reward = reward
        sample.status = Sample.Status.TRUNCATED if truncated else episode_status
        sample.metadata = {
            **(base_sample.metadata or {}),
            "uid": task.uid,
            "run_id": run_id,
            "submission": result.submission,
            "exit_status": result.status,
            "steps": result.steps,
            "truncated": truncated,
            "assistant_turn": turn_number,
            "teacher_model": backend.model_name,
            "target_has_reasoning": bool((target.get("reasoning_content") or "").strip()),
        }
        samples.append(sample)
    return samples


async def _run_episode(
    args: Any,
    state: _RolloutState,
    base_sample: Sample,
) -> tuple[list[Sample] | None, str, bool]:
    """Run one teacher episode. Returns (samples, outcome, passed) where
    outcome is "kept" | "filtered" | "failed" and passed marks a correct
    submission regardless of the filter."""
    async with state.pool.lease() as backend:
        try:
            task = task_from_slime_fields(
                base_sample.prompt, base_sample.label, base_sample.metadata
            )
            if task.answer is None:
                raise ValueError(f"sample {task.uid} has no label to grade")
            tolerance = _tolerance(base_sample)
            environment_config = environment_config_from_args(args)
            agent_config = agent_config_from_args(args)
        except (TypeError, ValueError) as error:
            logger.warning("teacher SFT rollout: invalid task, episode dropped: %s", error)
            return None, "failed", False

        run_id = _make_run_id(base_sample)
        agent = OfficeQAAgent(backend, agent_config, sampling_params=_sampling_params(args))
        try:
            async with FullCorpusEnvironment(environment_config, task, run_id=run_id) as env:
                result = await agent.run(env)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # sandbox/workspace failures must not spread
            logger.warning(
                "teacher SFT rollout: environment failure, episode dropped "
                "(run_id=%s): %s: %s",
                run_id,
                type(error).__name__,
                error,
            )
            return None, "failed", False

    if result.status not in _STATUS_BY_EXIT:
        logger.warning(
            "teacher SFT rollout: agent failure %r, episode dropped (run_id=%s)",
            result.status,
            run_id,
        )
        return None, "failed", False

    if result.status == "submitted":
        reward = score_answer(
            ground_truth=task.answer, predicted=result.submission, tolerance=tolerance
        )
    else:
        reward = 0.0
    await _persist_trace(args, base_sample, task, result, reward)
    passed = result.status == "submitted" and reward == 1.0
    if _filter_mode(args) == "correct" and not passed:
        return None, "filtered", passed

    samples = _episode_to_samples(
        args, state, base_sample, task, result, reward, run_id, backend
    )
    if not samples:
        logger.warning(
            "teacher SFT rollout: no trainable turns, episode dropped (run_id=%s)", run_id
        )
        return None, "failed", passed
    return samples, "kept", passed


async def _run_groups(
    args: Any,
    state: _RolloutState,
    groups: list[list[Sample]],
    runner: EpisodeRunner = None,
) -> tuple[list[list[Sample]], dict[str, int]]:
    runner = runner or _run_episode
    results = await asyncio.gather(
        *(
            runner(args, state, attempt)
            for group in groups
            for attempt in group
        )
    )
    stats = {"run": len(results), "kept": 0, "filtered": 0, "failed": 0, "passed": 0}
    episodes: list[list[Sample]] = []
    for samples, outcome, passed in results:
        stats[outcome] += 1
        stats["passed"] += int(passed)
        if samples:
            episodes.append(samples)
    return episodes, stats


def generate_rollout(
    args: Any,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
    *,
    runner: EpisodeRunner = None,
) -> RolloutFnTrainOutput:
    """Drive one rollout's worth of teacher episodes.

    ``runner`` swaps the per-episode body (task loading, environment, scoring)
    without duplicating the batch contract, so a second benchmark reuses the
    endpoint pool, the carry-over queue and the replacement loop verbatim —
    see :mod:`officeqa_rl.training.ssb1_teacher_sft_rollout`.
    """
    assert not evaluation, (
        "teacher SFT rollout has no evaluation branch; run with --debug-train-only "
        "and evaluate checkpoints offline"
    )
    _filter_mode(args)  # validate before any teacher traffic
    state = _get_state(args)
    target = int(args.rollout_batch_size) * int(args.n_samples_per_prompt)
    max_empty_rounds = int(getattr(args, "officeqa_sft_max_empty_rounds", 3))

    totals = {"run": 0, "kept": 0, "filtered": 0, "failed": 0, "passed": 0}
    empty_rounds = 0
    while len(state.episode_queue) < target:
        groups = data_source.get_samples(args.rollout_batch_size)
        episodes, stats = run(_run_groups(args, state, groups, runner))
        for key, value in stats.items():
            totals[key] += value
        logger.info(
            "teacher SFT rollout %d: round ran %d episodes -> kept %d, filtered %d, "
            "failed %d (queue %d/%d)",
            rollout_id,
            stats["run"],
            stats["kept"],
            stats["filtered"],
            stats["failed"],
            min(len(state.episode_queue) + len(episodes), target),
            target,
        )
        if episodes:
            empty_rounds = 0
            state.episode_queue.extend(episodes)
        else:
            empty_rounds += 1
            if empty_rounds >= max_empty_rounds:
                raise RuntimeError(
                    f"teacher SFT rollout produced no usable episodes in "
                    f"{empty_rounds} consecutive rounds ({totals['run']} episodes: "
                    f"{totals['filtered']} filtered, {totals['failed']} failed) — "
                    "the teacher endpoint is down, misconfigured, or never answers "
                    "correctly under officeqa_sft_filter=correct"
                )

    emitted = state.episode_queue[:target]
    del state.episode_queue[:target]

    attempted = max(totals["run"] - totals["failed"], 1)
    metrics = {
        "rollout/teacher_sft/episodes_run": float(totals["run"]),
        "rollout/teacher_sft/episodes_kept": float(totals["kept"]),
        "rollout/teacher_sft/episodes_filtered": float(totals["filtered"]),
        "rollout/teacher_sft/episodes_failed": float(totals["failed"]),
        "rollout/teacher_sft/pass_rate": totals["passed"] / attempted,
        "rollout/teacher_sft/queue_carryover": float(len(state.episode_queue)),
    }
    return RolloutFnTrainOutput(samples=emitted, metrics=metrics)


__all__ = [
    "EpisodeRunner",
    "_EndpointPool",
    "_RolloutState",
    "_episode_to_samples",
    "_filter_mode",
    "_get_state",
    "_sampling_params",
    "generate_rollout",
]
