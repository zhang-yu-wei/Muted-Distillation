"""W&B metrics that split truncation by cause for OfficeQA rollouts.

slime's built-in ``rollout/truncated_ratio`` and ``eval/<ds>-truncated_ratio``
conflate three different limits. These hooks add sample-weighted cause ratios
and episode-weighted stats next to them:

    rollout/truncated_by/step_limit        agent hit officeqa_step_limit
    rollout/truncated_by/context_exhausted turn clamped by max context len
    rollout/truncated_by/turn_cap          turn hit max response len
    rollout/truncated_by/length_other      finish_reason=="length", cap unknown
    rollout/episode/forced_close_turns_mean turns/episode closed by the think budget
    rollout/episode/forced_close_share     those turns as a share of all turns
    rollout/episode/count                  distinct episodes in the batch
    rollout/episode/limits_exceeded_ratio  episodes that never submitted
    rollout/episode/steps_mean|max         agent steps actually used
    rollout/episode/reward_mean            episode-weighted mean reward; slime's
                                           own reward/eval means are fragment-
                                           weighted and understate pass@1
    rollout/abort/ratio                    aborted (removed) samples in the batch
    rollout/abort/<category>               abort count by abort_reason prefix,
                                           e.g. rollout/abort/opd_teacher_failure

(and the same under ``eval/<dataset>/``). Aborted samples never train (slime
zero-masks them at train-data conversion) but they stay in the logged batch,
which is what lets the abort split here see ``metadata["abort_reason"]``. The cause split mirrors
``officeqa_rl.training.rollout.generate``: ``metadata["truncated"]`` means the
sample's own last turn ended with ``finish_reason == "length"``; otherwise a
TRUNCATED status was inherited from the episode exiting ``limits_exceeded``.

Wired through slime's ``--custom-rollout-log-function-path`` and
``--custom-eval-rollout-log-function-path``. Both hooks return ``None``
(falsy) so slime's default metrics still log; these are purely additive.
"""

from __future__ import annotations

import logging
from typing import Any

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _truncation_metrics(
    samples: list[Sample],
    *,
    max_response_len: int | None,
    max_context_len: int | None,
) -> dict[str, float]:
    counts = {"step_limit": 0, "context_exhausted": 0, "turn_cap": 0, "length_other": 0}
    episodes: dict[Any, dict[str, Any]] = {}
    for sample in samples:
        metadata = sample.metadata or {}
        run_id = metadata.get("run_id") or id(sample)
        episode = episodes.setdefault(
            run_id,
            {
                "exit_status": metadata.get("exit_status"),
                "steps": metadata.get("steps"),
                "reward": sample.reward,
                "output_tokens": metadata.get("episode_output_tokens"),
                # Turns the think budget closed (see rollout._forced_close_turns).
                "forced_closes": metadata.get("forced_close_turns"),
                # False when the judge could not grade the episode. Such a
                # sample is still trained (OPD's KL term does not need a task
                # reward), and its placeholder reward stays in the mean; this
                # flag is what feeds episode/ungraded_ratio below. Absent for
                # harnesses with no LLM judge, which are graded by definition.
                "reward_available": metadata.get("reward_available", True),
            },
        )
        # True generation cost of the episode, recorded once per episode by
        # the rollout hook (see rollout._episode_output_tokens). It cannot be
        # summed from samples: slime's response_len/mean averages
        # ``effective_response_length`` (= sum of the loss mask) over samples,
        # which is TRAINED tokens — action-only distillation keeps just the
        # fenced command — and whole turns are missing whenever truncation or
        # REALIGN drops them, by more than half in long episodes.
        if episode.get("output_tokens") is None:
            episode["output_tokens"] = metadata.get("episode_output_tokens")
        if sample.status != Sample.Status.TRUNCATED:
            continue
        if metadata.get("truncated"):
            if max_context_len and len(sample.tokens or []) >= int(max_context_len):
                counts["context_exhausted"] += 1
            elif max_response_len and int(sample.response_length or 0) >= int(max_response_len):
                counts["turn_cap"] += 1
            else:
                counts["length_other"] += 1
        elif metadata.get("exit_status") == "limits_exceeded":
            counts["step_limit"] += 1

    log_dict: dict[str, float] = {
        f"truncated_by/{cause}": count / len(samples) for cause, count in counts.items()
    }
    log_dict["episode/count"] = float(len(episodes))
    log_dict["episode/limits_exceeded_ratio"] = sum(
        1 for e in episodes.values() if e["exit_status"] == "limits_exceeded"
    ) / len(episodes)
    steps = [e["steps"] for e in episodes.values() if e["steps"] is not None]
    if steps:
        log_dict["episode/steps_mean"] = sum(steps) / len(steps)
        log_dict["episode/steps_max"] = float(max(steps))
    # Episode-weighted mean reward. slime's own ``eval/<ds>`` and reward
    # metrics average over samples, and fan-out gives long (usually failing)
    # episodes one sample per turn — a fragment-weighted mean that sits far
    # below pass@1 (measured: 0.049 vs 0.095 on the base model). Every
    # fragment carries its episode's reward, so grouping by run_id recovers
    # the per-question mean; with one eval sample per prompt this is directly
    # comparable to offline pass@1 numbers.
    rewards = [e["reward"] for e in episodes.values() if e["reward"] is not None]
    if rewards:
        log_dict["episode/reward_mean"] = sum(rewards) / len(rewards)
    # Episodes the judge could not grade are kept and trained under OPD, and
    # their placeholder 0.0 stays in the mean above rather than being filtered
    # out: the mean is over every episode generated, which is the number that
    # stays comparable across rollouts. This ratio is what makes that mean
    # readable -- it is the share of the denominator that is a grading outage
    # rather than a policy outcome, and since those episodes contribute exactly
    # 0.0 the graded-only mean is recoverable as
    # ``episode/reward_mean / (1 - episode/ungraded_ratio)``.
    log_dict["episode/ungraded_ratio"] = sum(
        1 for e in episodes.values() if not e.get("reward_available", True)
    ) / len(episodes)
    generated = [
        e["output_tokens"] for e in episodes.values() if e["output_tokens"] is not None
    ]
    if generated:
        log_dict["episode/output_tokens_mean"] = sum(generated) / len(generated)
    # Think-budget interventions: how many turns per episode the backend had
    # to close, and what share of all turns that is. Both are 0 when the
    # budget is off; a rising share is the length pathology moving into the
    # budget instead of into the response cap.
    forced = [
        (e["forced_closes"], e["steps"])
        for e in episodes.values()
        if e.get("forced_closes") is not None
    ]
    if forced:
        log_dict["episode/forced_close_turns_mean"] = sum(f for f, _ in forced) / len(forced)
        total_steps = sum(int(st) for _, st in forced if st)
        if total_steps:
            log_dict["episode/forced_close_share"] = sum(f for f, _ in forced) / total_steps
    return log_dict


def _abort_metrics(samples: list[Sample]) -> dict[str, float]:
    """Count aborted samples and split by abort_reason category — the prefix
    before the first ':' (opd_teacher_failure, environment_failure, ...).
    ``abort/ratio`` always logs (0.0 when clean) so the W&B line is stable;
    the per-category counts only appear when that category occurred."""
    counts: dict[str, int] = {}
    for sample in samples:
        if sample.status != Sample.Status.ABORTED:
            continue
        reason = (sample.metadata or {}).get("abort_reason") or "unknown"
        category = reason.split(":", 1)[0].strip()
        counts[category] = counts.get(category, 0) + 1
    log_dict: dict[str, float] = {
        "abort/ratio": sum(counts.values()) / len(samples),
    }
    for category, count in counts.items():
        log_dict[f"abort/{category}"] = float(count)
    return log_dict


def log_rollout_extra_metrics(
    rollout_id: int,
    args: Any,
    samples: list[Sample],
    rollout_extra_metrics: dict | None,
    rollout_time: float,
) -> None:
    """--custom-rollout-log-function-path hook; additive, keeps slime defaults."""
    if not samples:
        return None
    metrics = _truncation_metrics(
        samples,
        max_response_len=args.rollout_max_response_len,
        max_context_len=args.rollout_max_context_len,
    )
    metrics |= _abort_metrics(samples)
    log_dict = {f"rollout/{key}": value for key, value in metrics.items()}
    log_dict["rollout/step"] = compute_rollout_step(args, rollout_id)
    logger.info(f"truncation {rollout_id}: {log_dict}")
    logging_utils.log(args, log_dict, step_key="rollout/step")
    return None


def log_eval_extra_metrics(
    rollout_id: int,
    args: Any,
    data: dict[str, dict],
    extra_metrics: dict | None = None,
) -> None:
    """--custom-eval-rollout-log-function-path hook; additive, keeps slime defaults."""
    log_dict: dict[str, float] = {}
    for dataset_name, info in data.items():
        samples = info.get("samples")
        if not samples:
            continue
        metrics = _truncation_metrics(
            samples,
            max_response_len=args.eval_max_response_len,
            max_context_len=args.eval_max_context_len,
        )
        metrics |= _abort_metrics(samples)
        log_dict |= {f"eval/{dataset_name}/{key}": value for key, value in metrics.items()}
    if log_dict:
        log_dict["eval/step"] = compute_rollout_step(args, rollout_id)
        logger.info(f"eval truncation {rollout_id}: {log_dict}")
        logging_utils.log(args, log_dict, step_key="eval/step")
    return None


__all__ = ["log_eval_extra_metrics", "log_rollout_extra_metrics"]
