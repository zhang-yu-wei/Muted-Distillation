"""W&B metrics for JobBench rollouts: reward decomposition on top of the shared hooks.

The reward slime logs is one number per sample — the normalized weighted rubric
score. These hooks add the episode-weighted decomposition, so a flat or moving
reward can be read as behaviour rather than noise::

    rollout/jobbench/episodes            scored (non-aborted) episodes in the batch
    rollout/jobbench/submitted_ratio     episodes that reached the submit sentinel
    rollout/jobbench/empty_output_ratio  episodes that wrote no deliverable at all
    rollout/jobbench/normalized_score    weighted rubric score, mean over episodes
    rollout/jobbench/rubric_pass_rate    fraction of rubric items passed, mean
    rollout/jobbench/output_files        deliverables written, mean over episodes
    rollout/jobbench/steps               model calls per episode, mean

(and the same under ``eval/<dataset>/``).

``empty_output_ratio`` is the one to watch early: a policy that never writes to
``output/`` scores zero on every rubric, and that failure looks identical to a
policy that writes bad deliverables unless it is logged separately.

Episodes aborted before scoring are excluded — they carry no outcome, and the
shared hooks already log their ratio and reasons. Episodes the judge could not
grade are *not* excluded: under OPD they are trained rather than dropped, and
every average here is over the episodes actually generated. Their rubric
numbers are a grading outage rather than an outcome, so read these alongside
``episode/ungraded_ratio`` from the shared hook, which reports how much of the
denominator that is. Note the two hooks differ on what such an episode
contributes: ``episode/reward_mean`` sees the placeholder reward of exactly
0.0, while ``normalized_score`` here sees whatever partial credit the rubrics
that did grade earned (``build_score`` divides by the full rubric weight), so
only the former is exactly recoverable from the ratio.

Wire one of these (not both)
through slime's ``--custom-rollout-log-function-path`` and
``--custom-eval-rollout-log-function-path``.
"""

from __future__ import annotations

import logging
from typing import Any

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.types import Sample

from officeqa_rl.training import metrics as shared_metrics

logger = logging.getLogger(__name__)


def _jobbench_metrics(samples: list[Sample]) -> dict[str, float]:
    episodes: dict[Any, dict[str, Any]] = {}
    for sample in samples:
        metadata = sample.metadata or {}
        if metadata.get("benchmark") != "jobbench":
            continue
        if sample.status == Sample.Status.ABORTED:
            continue
        episodes.setdefault(metadata.get("run_id") or id(sample), metadata)
    if not episodes:
        return {}

    count = len(episodes)
    submitted = empty_output = 0
    score = pass_rate = output_files = steps = 0.0
    for metadata in episodes.values():
        if metadata.get("submitted"):
            submitted += 1
        if not int(metadata.get("n_output_files", 0) or 0):
            empty_output += 1
        scores = metadata.get("jobbench_metrics") or {}
        score += float(scores.get("normalized_score", 0.0))
        pass_rate += float(scores.get("pass_rate", 0.0))
        output_files += float(metadata.get("n_output_files", 0) or 0)
        steps += float(metadata.get("steps", 0) or 0)
    return {
        "jobbench/episodes": float(count),
        "jobbench/submitted_ratio": submitted / count,
        "jobbench/empty_output_ratio": empty_output / count,
        "jobbench/normalized_score": score / count,
        "jobbench/rubric_pass_rate": pass_rate / count,
        "jobbench/output_files": output_files / count,
        "jobbench/steps": steps / count,
    }


def log_rollout_extra_metrics(
    rollout_id: int,
    args: Any,
    samples: list[Sample],
    rollout_extra_metrics: dict | None,
    rollout_time: float,
) -> None:
    """--custom-rollout-log-function-path hook; additive, keeps slime defaults."""
    shared_metrics.log_rollout_extra_metrics(
        rollout_id, args, samples, rollout_extra_metrics, rollout_time
    )
    if not samples:
        return None
    metrics = _jobbench_metrics(samples)
    if metrics:
        log_dict = {f"rollout/{key}": value for key, value in metrics.items()}
        log_dict["rollout/step"] = compute_rollout_step(args, rollout_id)
        logger.info(f"jobbench {rollout_id}: {log_dict}")
        logging_utils.log(args, log_dict, step_key="rollout/step")
    return None


def log_eval_extra_metrics(
    rollout_id: int,
    args: Any,
    data: dict[str, dict],
    extra_metrics: dict | None = None,
) -> None:
    """--custom-eval-rollout-log-function-path hook; additive, keeps slime defaults."""
    shared_metrics.log_eval_extra_metrics(rollout_id, args, data, extra_metrics)
    log_dict: dict[str, float] = {}
    for dataset_name, info in data.items():
        samples = info.get("samples")
        if not samples:
            continue
        metrics = _jobbench_metrics(samples)
        log_dict |= {
            f"eval/{dataset_name}/{key}": value for key, value in metrics.items()
        }
    if log_dict:
        log_dict["eval/step"] = compute_rollout_step(args, rollout_id)
        logger.info(f"jobbench eval {rollout_id}: {log_dict}")
        logging_utils.log(args, log_dict, step_key="eval/step")
    return None


__all__ = ["log_eval_extra_metrics", "log_rollout_extra_metrics"]
