"""W&B metrics for SSB1 rollouts: reward decomposition on top of the shared hooks.

The reward logged by slime is one number per sample (soft restriction during
training, hard restriction during eval — see
:mod:`officeqa_rl.training.ssb1_rollout`). These hooks additionally log the
episode-weighted decomposition so reward movement is diagnosable:

    rollout/ssb1/episodes              scored (non-aborted) episodes in the batch
    rollout/ssb1/submitted_ratio       episodes that reached `submit`
    rollout/ssb1/hard_restriction      official strict score, mean over episodes
    rollout/ssb1/soft_restriction      fraction of cases passed, mean over episodes
    rollout/ssb1/cell_accuracy         designated-cell accuracy, mean over episodes

(and the same under ``eval/<dataset>/``). Episodes that never submitted score
0 on every metric — they are included, so the means match reward semantics.
Aborted samples are excluded here (they carry no episode outcome); the shared
hooks already log their ratio and reasons.

Each function first calls its counterpart in
:mod:`officeqa_rl.training.metrics`, so SSB1 runs keep the truncation-cause
and abort splits; wire these (not both) through slime's
``--custom-rollout-log-function-path`` and
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


def _ssb1_metrics(samples: list[Sample]) -> dict[str, float]:
    episodes: dict[Any, dict[str, Any]] = {}
    for sample in samples:
        metadata = sample.metadata or {}
        if metadata.get("benchmark") != "ssb1":
            continue
        if sample.status == Sample.Status.ABORTED:
            continue
        episodes.setdefault(metadata.get("run_id") or id(sample), metadata)
    if not episodes:
        return {}

    count = len(episodes)
    submitted = 0
    hard = soft = cell = 0.0
    for metadata in episodes.values():
        if metadata.get("exit_status") == "submitted":
            submitted += 1
        scores = metadata.get("ssb1_metrics") or {}
        hard += float(scores.get("hard_restriction", 0.0))
        soft += float(scores.get("soft_restriction", 0.0))
        cell += float(scores.get("cell_accuracy", 0.0))
    return {
        "ssb1/episodes": float(count),
        "ssb1/submitted_ratio": submitted / count,
        "ssb1/hard_restriction": hard / count,
        "ssb1/soft_restriction": soft / count,
        "ssb1/cell_accuracy": cell / count,
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
    metrics = _ssb1_metrics(samples)
    if metrics:
        log_dict = {f"rollout/{key}": value for key, value in metrics.items()}
        log_dict["rollout/step"] = compute_rollout_step(args, rollout_id)
        logger.info(f"ssb1 {rollout_id}: {log_dict}")
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
        metrics = _ssb1_metrics(samples)
        log_dict |= {f"eval/{dataset_name}/{key}": value for key, value in metrics.items()}
    if log_dict:
        log_dict["eval/step"] = compute_rollout_step(args, rollout_id)
        logger.info(f"ssb1 eval {rollout_id}: {log_dict}")
        logging_utils.log(args, log_dict, step_key="eval/step")
    return None


__all__ = ["log_eval_extra_metrics", "log_rollout_extra_metrics"]
