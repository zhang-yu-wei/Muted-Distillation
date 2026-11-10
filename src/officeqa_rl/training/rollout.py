"""Slime per-sample rollout hook for OfficeQA agent training.

This is the ``--custom-generate-function-path officeqa_rl.training.rollout.generate``
entry point. Slime's default ``sglang_rollout.generate_rollout`` outer loop keeps
owning batching, concurrency, repeated samples, dynamic filtering, and aborts;
this hook owns exactly one sample: build the OfficeQA task, run the shared agent
loop against the current actor through :class:`SlimeSGLangBackend` inside an
isolated :class:`FullCorpusEnvironment`, score the terminal submission, and
export the recorded exact-token turns as slime training samples.

OfficeQA-specific settings ride on ``args`` under an ``officeqa_`` prefix, set
through slime's ``--custom-config-path`` YAML, for example::

    officeqa_corpus_dir: /path/to/treasury_bulletins_parsed/transformed
    officeqa_workspace_root: /mnt/officeqa-rl/results/rollout_workspaces
    officeqa_sandbox_image: python:3.11-slim
    officeqa_step_limit: 120

Under DooD, ``officeqa_workspace_root`` and ``officeqa_corpus_dir`` must be
visible at the same absolute path on the Docker host (see README).

RL+OPD (``--use-opd --opd-type sglang``) additionally scores every emitted
training sample with the served teacher through
:mod:`officeqa_rl.training.vertex_teacher_client`, configured by
``officeqa_opd_teacher_url`` (or ``officeqa_opd_project`` /
``officeqa_opd_location`` / ``officeqa_opd_endpoint_id``) plus optional
``officeqa_opd_concurrency`` / ``officeqa_opd_timeout_sec`` /
``officeqa_opd_attempts`` / ``officeqa_opd_backoff_sec``. When
``officeqa_opd_failure_log`` is set, every terminal teacher-scoring failure is
appended there as a JSON line (timestamp, run_id, uid, error).

Failures of the task/environment kind never crash the surrounding rollout
batch: malformed tasks, environment failures, and empty trajectories return
one removable aborted sample carrying ``metadata["abort_reason"]``.
Teacher-scoring failures are the deliberate exception: they re-raise (after
writing the failure log) so the run stops loudly — an abort would zero-mask
the episode and let training continue without teacher signal, hiding a broken
teacher deployment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slime.agent.trajectory import TrajectoryManager, TurnRecord
from slime.utils.types import Sample

from officeqa_rl.agent import OfficeQAAgent
from officeqa_rl.dataset import task_from_slime_fields
from officeqa_rl.full_corpus_env import FullCorpusEnvironment
from officeqa_rl.scoring import DEFAULT_TOLERANCE, score_answer
from officeqa_rl.training.slime_sglang_backend import SessionTurn, SlimeSGLangBackend
from officeqa_rl.types import AgentConfig, EnvironmentConfig

logger = logging.getLogger(__name__)

# Agent exit statuses that still yield trainable samples. "error: ..." statuses
# mean the trajectory ended for infrastructure reasons, not policy reasons, and
# become removable aborted samples instead.
_STATUS_BY_EXIT = {
    "submitted": Sample.Status.COMPLETED,
    "limits_exceeded": Sample.Status.TRUNCATED,
}

# Stand-in prompt token for an abort that never generated anything. Any valid
# vocabulary index works — the position is masked out of the loss and the
# sample is flagged ``remove_sample`` — it only has to exist so slime's
# prompt_length stays >= 1. See :func:`_abort`.
_ABORT_PAD_TOKEN = 0

_TOKENIZERS: dict[str, Any] = {}


def _get_tokenizer(args: Any) -> Any:
    """Load and cache the actor tokenizer; imported lazily because
    ``slime.utils.processing_utils`` pulls in transformers."""
    checkpoint = args.hf_checkpoint
    if checkpoint not in _TOKENIZERS:
        from slime.utils.processing_utils import load_tokenizer

        _TOKENIZERS[checkpoint] = load_tokenizer(checkpoint, trust_remote_code=True)
    return _TOKENIZERS[checkpoint]


def _require_arg(args: Any, name: str) -> Any:
    value = getattr(args, name, None)
    if value is None or value == "":
        raise ValueError(
            f"OfficeQA rollout requires args.{name}; set it in the "
            "--custom-config-path YAML"
        )
    return value


def environment_config_from_args(args: Any) -> EnvironmentConfig:
    pdf_dir = getattr(args, "officeqa_pdf_dir", None)
    return EnvironmentConfig(
        corpus_dir=Path(_require_arg(args, "officeqa_corpus_dir")),
        pdf_dir=Path(pdf_dir) if pdf_dir else None,
        workspace_root=Path(_require_arg(args, "officeqa_workspace_root")),
        sandbox_image=getattr(args, "officeqa_sandbox_image", None),
        command_timeout_sec=int(getattr(args, "officeqa_command_timeout_sec", 30)),
        max_observation_chars=int(getattr(args, "officeqa_max_observation_chars", 12_000)),
        sandbox_runtime=getattr(args, "officeqa_sandbox_runtime", "docker"),
        max_concurrent_sandbox_boots=int(
            getattr(args, "officeqa_max_concurrent_sandbox_boots", 4)
        ),
    )


def agent_config_from_args(args: Any) -> AgentConfig:
    return AgentConfig(step_limit=int(getattr(args, "officeqa_step_limit", 120)))


def _create_backend(
    args: Any, *, evaluation: bool, routing_key: str | None
) -> SlimeSGLangBackend:
    max_context_len = (
        args.eval_max_context_len if evaluation else args.rollout_max_context_len
    )
    if not max_context_len or int(max_context_len) <= 0:
        raise ValueError(
            "OfficeQA rollout requires --rollout-max-context-len to budget "
            "multi-turn generation"
        )
    tokenizer = _get_tokenizer(args)
    think_budget = int(getattr(args, "officeqa_think_budget", 0) or 0)
    open_id, close_id = _think_marker_ids(tokenizer) if think_budget > 0 else (None, None)
    # Modes whose trained span is cut by the fence/tail planners need a forced
    # close turn as one record; see SlimeSGLangBackend.think_budget_single_record.
    span_planned = any(
        bool(getattr(args, name, False))
        for name in (
            "officeqa_opd_action_only",
            "officeqa_opd_teacher_no_think",
            "officeqa_opd_tail_loss",
            "officeqa_opd_mask_think",
        )
    )
    return SlimeSGLangBackend(
        base_url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
        tokenizer=tokenizer,
        max_context_len=int(max_context_len),
        routing_key=routing_key,
        chat_template_kwargs=_chat_template_kwargs(args),
        # Yaml key officeqa_sock_read_timeout_sec. The default 900s abandons a
        # generate request that queues behind other long episodes, which shows
        # up as agent_failure: SocketTimeoutError and costs the whole episode
        # (12+ lost in one eval on 2026-09-04 once episodes passed ~35k tokens).
        sock_read_timeout_sec=float(
            getattr(args, "officeqa_sock_read_timeout_sec", 900.0) or 900.0
        ),
        think_budget=think_budget,
        think_open_token_id=open_id,
        think_close_token_id=close_id,
        think_budget_single_record=span_planned,
    )


def _think_marker_ids(tokenizer: Any) -> tuple[int, int]:
    """(open, close) token ids of the thought channel in the actor's dialect.

    Mirrors ``opd_rollout._think_dialect`` -- not imported, since that module
    imports this one. Gemma-4 marks the channel with ``<|channel>``/
    ``<channel|>``; Qwen with ``<think>``/``</think>``.
    """
    template = getattr(tokenizer, "chat_template", "") or ""
    names = (
        ("<|channel>", "<channel|>") if "<|channel>thought" in template
        else ("<think>", "</think>")
    )
    ids = tuple(tokenizer.convert_tokens_to_ids(name) for name in names)
    unk = getattr(tokenizer, "unk_token_id", None)
    if any(i is None or i == unk or int(i) < 0 for i in ids):
        raise ValueError(
            f"officeqa_think_budget: tokenizer has no single-token ids for {names}: {ids}"
        )
    return int(ids[0]), int(ids[1])


def _chat_template_kwargs(args: Any) -> dict[str, Any]:
    """Template kwargs for prompt rendering.

    Only set when explicitly requested: Qwen3.5 has no ``enable_thinking``
    parameter and leaves its ``<think>`` block open by default, while Gemma-4
    accepts the flag and defaults it to false, pre-closing the thought channel
    so the policy never reasons. Passing the flag unconditionally would break
    templates that do not take it.
    """
    value = getattr(args, "officeqa_enable_thinking", None)
    return {} if value is None else {"enable_thinking": bool(value)}


def _make_run_id(base_sample: Sample) -> str:
    """Unique per-execution ID: slime runs many samples of one prompt (and one
    OfficeQA uid) concurrently, so workspaces and containers key on this."""
    return (
        f"idx{base_sample.index}"
        f"-grp{base_sample.group_index}"
        f"-rid{base_sample.rollout_id}"
        f"-{uuid.uuid4().hex[:8]}"
    )


def _tolerance(base_sample: Sample) -> float:
    metadata = base_sample.metadata or {}
    tolerance = float(metadata.get("officeqa_reward_tolerance", DEFAULT_TOLERANCE))
    if not 0 <= tolerance <= 1:
        raise ValueError("metadata.officeqa_reward_tolerance must be between 0 and 1")
    return tolerance


def _episode_output_tokens(session_turns: list[Any]) -> int:
    """Tokens the model actually generated in this episode, all turns.

    The assembled samples are not a reliable basis for this: a turn is left
    out whenever ``max_sample_tokens`` cuts the trajectory or REALIGN folds
    its output back into context, and in long episodes that can drop more
    than half the turns — measured 17 recorded turns against 41 taken. Only
    the session's own turn records cover every generation, so the cost metric
    is computed here rather than from sample lengths downstream.
    """
    total = 0
    for session_turn in session_turns:
        turn = getattr(session_turn, "turn", None)
        total += len(getattr(turn, "output_ids", None) or [])
    return total


def _forced_close_turns(session_turns: list[Any]) -> int:
    """Turns whose thought channel hit the think budget and was closed by the
    backend (see SlimeSGLangBackend._generate_with_think_budget)."""
    return sum(
        1
        for session_turn in session_turns
        if (getattr(getattr(session_turn, "turn", None), "usage", None) or {}).get(
            "think_budget_forced"
        )
    )


def _abort(base_sample: Sample, reason: str, *, run_id: str | None = None) -> Sample:
    """Return the sample as removable/aborted instead of raising, so one bad
    task cannot crash unrelated rollout tasks in the batch.

    ``remove_sample`` does not take the sample out of the training batch —
    slime only zeroes its loss mask (ray/rollout.py:719) and still tensorizes
    every per-token field. A ``None`` field therefore kills the whole step
    with "must be real number, not NoneType", so the aborted sample is filled
    with inert per-token values of the right length instead of left empty.

    For the same reason the sample must keep at least one prompt token: slime
    derives ``prompt_length = len(tokens) - response_length`` and left-pads the
    loss mask by ``prompt_length - 1`` (megatron_utils/data.py:130), so a
    tokenless sample makes that pad negative and takes the whole training step
    down with "narrow(): length must be non-negative". Aborts that fire before
    any generation carry no tokens at all, so one is supplied here; the mask
    over it is zero, and ``remove_sample`` keeps it out of the loss.
    """
    logger.warning("OfficeQA rollout aborted (run_id=%s): %s", run_id, reason)
    base_sample.status = Sample.Status.ABORTED
    base_sample.remove_sample = True
    base_sample.reward = 0.0
    response_length = int(base_sample.response_length or 0)
    tokens = list(base_sample.tokens or [])
    if len(tokens) - response_length < 1:
        tokens = [_ABORT_PAD_TOKEN] * (response_length + 1 - len(tokens)) + tokens
    base_sample.tokens = tokens
    base_sample.response_length = response_length
    base_sample.loss_mask = [0] * response_length
    base_sample.rollout_log_probs = [0.0] * response_length
    base_sample.teacher_log_probs = [0.0] * response_length
    metadata = dict(base_sample.metadata or {})
    metadata["abort_reason"] = reason
    if run_id is not None:
        metadata["run_id"] = run_id
    base_sample.metadata = metadata
    return base_sample


def build_training_samples(
    *,
    session_turns: list[SessionTurn],
    base_sample: Sample,
    reward: float,
    tokenizer: Any,
    extra_metadata: dict[str, Any] | None = None,
    max_sample_tokens: int = 0,
) -> list[Sample]:
    """Convert one session's exact recorded turns into slime training samples.

    Every turn is replayed into slime's :class:`TrajectoryManager`, which owns
    the loss-mask semantics: model-generated output tokens are ``1``; the task
    prompt, tool output, and format-error observations arrive as prompt-tail
    tokens and are ``0``. ``reward`` is assigned in full to every emitted
    sibling sample (slime averages within a shared ``rollout_id``), which is
    the chosen fan-out reward semantics.
    """
    manager = TrajectoryManager()
    sid = "session"
    try:
        for session_turn in session_turns:
            turn = session_turn.turn
            manager.record_turn(
                sid,
                turn=TurnRecord(
                    prompt_ids=list(turn.prompt_ids or []),
                    output_ids=list(turn.output_ids or []),
                    finish_reason=turn.finish_reason or "stop",
                    output_log_probs=list(turn.output_log_probs or []),
                ),
                prompt_messages=session_turn.prompt_messages,
                response_message={"role": "assistant", "content": turn.text},
            )
        samples = manager.get_trajectory(
            sid,
            base_sample=base_sample,
            reward=reward,
            extra_metadata=extra_metadata,
            max_sample_tokens=max_sample_tokens,
        )
    finally:
        manager.drop_session(sid)

    for sample in samples:
        response_length = int(sample.response_length or 0)
        assert sample.loss_mask is not None and len(sample.loss_mask) == response_length
        assert (
            sample.rollout_log_probs is not None
            and len(sample.rollout_log_probs) == response_length
        )
        assert len(sample.tokens) >= response_length
        # The manager is tokenizer-free; decode the response region here so
        # slime's logging/debug dumps stay readable.
        sample.response = (
            tokenizer.decode(sample.tokens[-response_length:], skip_special_tokens=False)
            if response_length and sample.tokens
            else ""
        )
    return samples


def _teacher_scoring_enabled(args: Any) -> bool:
    """RL+OPD with a served teacher: slime's train side requires
    ``teacher_log_probs`` on every trainable sample when ``--use-opd
    --opd-type sglang`` is set (``apply_opd_kl_to_advantages`` raises without
    them). The teacher is reached through the Vertex rawPredict client rather
    than slime's ``--custom-rm-path`` reference path because OfficeQA computes
    rewards inline in this hook."""
    return bool(getattr(args, "use_opd", False)) and getattr(args, "opd_type", None) == "sglang"


async def _log_teacher_failure(
    args: Any, base_sample: Sample, run_id: str, error: BaseException
) -> None:
    """Append a terminal teacher-scoring failure to the JSONL configured by
    ``officeqa_opd_failure_log`` (the launchers point it at shared /mnt), so
    the reason survives outside the Ray worker logs. Never raises: this runs
    on the way into the abort path and must not mask the original error."""
    path = getattr(args, "officeqa_opd_failure_log", None)
    if not path:
        return
    record = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rollout_id": base_sample.rollout_id,
        "run_id": run_id,
        "uid": (base_sample.metadata or {}).get("uid"),
        "error": f"{type(error).__name__}: {error}",
    }

    def _append() -> None:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    try:
        await asyncio.to_thread(_append)
    except Exception:
        logger.exception("Failed to write OPD teacher failure log %s", path)


async def _attach_teacher_log_probs(args: Any, samples: list[Sample]) -> None:
    """Score each sample's exact token sequence with the served teacher and
    keep the response-aligned tail. All-or-nothing per episode: any failure
    raises so the caller aborts the whole episode, because a trainable sample
    without ``teacher_log_probs`` breaks the train-side batch packing."""
    from officeqa_rl.training.vertex_teacher_client import get_teacher_client

    for flag in ("officeqa_opd_action_only", "officeqa_opd_mask_think"):
        if getattr(args, flag, False):
            raise ValueError(
                f"{flag} requires per-turn scoring (OPD_SCORE_MODE=per_turn, "
                "officeqa_rl.training.opd_rollout.generate); the per-sample path "
                "scores whole assembled sequences and cannot resolve per-turn "
                "reasoning boundaries"
            )
    client = get_teacher_client(args)
    # Skip the prompt region: we only keep response-token log-probs, and
    # logprob_start_len = prompt_length - 1 returns exactly those (the first
    # window entry is unscored) while letting the teacher reuse radix-cached
    # prompt KV instead of recomputing logits over the whole sequence.
    starts = [
        max(len(sample.tokens) - int(sample.response_length or 0) - 1, 0)
        for sample in samples
    ]
    # One affinity key per episode: all fragments share context heads, so
    # scoring them against the same endpoint reuses its cached prefixes.
    affinity = (samples[0].metadata or {}).get("run_id") if samples else None
    results = await asyncio.gather(
        *(
            client.score_tokens(
                sample.tokens, logprob_start_len=start, affinity_key=affinity
            )
            for sample, start in zip(samples, starts, strict=True)
        ),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, BaseException):
            raise result
    for sample, log_probs in zip(samples, results, strict=True):
        response_length = int(sample.response_length or 0)
        if not 0 < response_length <= len(log_probs):
            raise ValueError(
                f"cannot trim {len(log_probs)} teacher log-probs to "
                f"response_length={response_length}"
            )
        sample.teacher_log_probs = log_probs[-response_length:]


async def generate(
    args: Any,
    base_sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    """Run one OfficeQA agent rollout and return its training sample(s)."""
    try:
        task = task_from_slime_fields(
            base_sample.prompt, base_sample.label, base_sample.metadata
        )
        if task.answer is None:
            raise ValueError(f"sample {task.uid} has no label to grade")
        tolerance = _tolerance(base_sample)
    except (TypeError, ValueError) as error:
        return _abort(base_sample, f"invalid_task: {error}")

    run_id = _make_run_id(base_sample)
    routing_key = (
        base_sample.session_id
        if getattr(args, "router_policy", None) == "consistent_hashing"
        else None
    )
    try:
        backend = _create_backend(args, evaluation=evaluation, routing_key=routing_key)
        agent = OfficeQAAgent(
            backend, agent_config_from_args(args), sampling_params=sampling_params
        )
        environment_config = environment_config_from_args(args)
    except (TypeError, ValueError) as error:
        return _abort(base_sample, f"invalid_rollout_config: {error}", run_id=run_id)

    try:
        async with FullCorpusEnvironment(environment_config, task, run_id=run_id) as env:
            result = await agent.run(env)
    except asyncio.CancelledError:
        raise
    except Exception as error:  # sandbox/workspace failures must not spread
        return _abort(
            base_sample,
            f"environment_failure: {type(error).__name__}: {error}",
            run_id=run_id,
        )

    terminal_status = _STATUS_BY_EXIT.get(result.status)
    if terminal_status is None:
        return _abort(base_sample, f"agent_failure: {result.status}", run_id=run_id)

    if result.status == "submitted":
        reward = score_answer(
            ground_truth=task.answer,
            predicted=result.submission,
            tolerance=tolerance,
        )
    else:
        reward = 0.0

    extra_metadata = {
        **(base_sample.metadata or {}),
        "uid": task.uid,
        "run_id": run_id,
        "submission": result.submission,
        "exit_status": result.status,
        "steps": result.steps,
        "episode_output_tokens": _episode_output_tokens(backend.session_turns),
        "forced_close_turns": _forced_close_turns(backend.session_turns),
    }
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

    # Eval rollouts never train, so they skip the teacher round-trip.
    # Teacher-scoring failures are deliberately FATAL, not aborts: an abort
    # would zero-mask the episode and let training continue without any
    # teacher signal, silently. Write the failure log, then re-raise so the
    # rollout (and the run) stops with the real reason.
    if not evaluation and _teacher_scoring_enabled(args):
        try:
            await _attach_teacher_log_probs(args, samples)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await _log_teacher_failure(args, base_sample, run_id, error)
            raise

    # slime's prefix-cache metrics divide the cached/prompt token sums over
    # all samples, so putting the episode totals on one sample (zeros on the
    # siblings) yields the exact aggregate hit rate.
    samples[0].prefix_cache_info.cached_tokens = backend.cached_prompt_tokens
    samples[0].prefix_cache_info.total_prompt_tokens = backend.total_prompt_tokens

    return samples[0] if len(samples) == 1 else samples


__all__ = [
    "agent_config_from_args",
    "build_training_samples",
    "environment_config_from_args",
    "generate",
]
