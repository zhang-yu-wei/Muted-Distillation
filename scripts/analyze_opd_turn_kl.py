"""Per-turn KL(student || teacher) analysis for divergence-aware OPD truncation.

The truncation proposal cuts a
student rollout at the first turn whose divergence from the teacher crosses a
threshold: turns up to and including that turn keep the OPD tilt, later turns
are dropped because their prefix now contains an action the teacher would never
take. Before wiring that into training we need to know (a) what the per-turn
divergence distribution actually looks like, and (b) which per-turn statistic
separates "teacher-implausible" turns most cleanly, so the threshold can be set
on real percentiles instead of guesswork.

This script replays existing ``--save-debug-rollout-data`` dumps — no sandbox,
no fresh rollouts. For every trained turn of every episode it pairs

* student log-probs of the sampled tokens: the dump's ``rollout_log_probs``
  (recorded by the rollout engine at sampling time), or optionally re-scored
  against a locally hosted student server (``--student-url``) when the dump's
  engine values should not be trusted or a different checkpoint is analyzed;
* teacher log-probs of the same tokens: scored through
  :class:`officeqa_rl.training.vertex_teacher_client.VertexTeacherClient`
  against the Vertex endpoint (or reused from the dump's ``teacher_log_probs``
  with ``--use-dumped-teacher`` — vanilla OPD dumps only, COPD contrast runs
  store clipped contrasts there, not log-probs).

Per-turn metric reference
=========================

Notation, for one trained turn of n tokens x_1..x_n (the student's sampled
tokens, in context): s_i = log p_student(x_i | prefix) and
t_i = log p_teacher(x_i | prefix) are the two models' log-probs of the SAME
sampled token, and d_i = s_i - t_i is their per-token disagreement. Because
the x_i were drawn from the student, averaging any function of d_i over the
turn is a Monte-Carlo estimate of a student-expectation; in particular
E[d] = KL(student || teacher). All log quantities are in nats. d_i > 0 means
the student likes its own token more than the teacher does; d_i ~ 2 means the
student gave it about e^2 ~ 7x the teacher's probability.

KL family (always available; needs both sides' sampled-token log-probs):

* ``kl_mean`` = mean(d_i) — the "k1" estimate of this turn's per-token
  KL(student || teacher); the quantity drawn in the figure. Unbiased, but on
  a finite turn it can go negative (tokens where the teacher is MORE
  confident than the student pull it down).
* ``kl_sum`` = sum(d_i) — the whole turn's divergence in nats. Grows with
  turn length, so it treats a long mildly-off turn like a short badly-off
  one; compare against ``kl_mean`` to separate "long and drifting" from
  "short and wrong".
* ``kl_k3_mean`` = mean(exp(-d_i) - 1 + d_i) — the "k3" KL estimate.
  Non-negative for every token and lower-variance than k1, so it cannot be
  cancelled to zero by mixed-sign tokens; a good default when turns are
  short. Its exp() blows up where the teacher vastly out-confidences the
  student (very negative d), so the exponent is clipped at 50.
* ``d_max`` — the single worst token: max(d_i). Catches one "impossible
  action" token (a wrong cell reference, a hallucinated filename) inside an
  otherwise teacher-plausible turn, which any mean would dilute away.
* ``d_p95`` — 95th-percentile d_i: like ``d_max`` but robust to a lone
  outlier token.
* ``d_min`` — most negative d_i: the strongest token-level case where the
  teacher out-prefers the student's own sample.
* ``d_std`` — within-turn spread of d_i. A turn can show kl_mean ~ 0 through
  cancellation while student and teacher disagree wildly token by token.
* ``frac_d_gt2`` / ``frac_d_gt5`` — fraction of the turn's tokens with
  d_i > 2 (~7x probability ratio) / d_i > 5 (~150x): "how much of the turn
  does the teacher dislike", in interpretable units.

Sampled-log-prob side statistics (always available):

* ``student_mean_lp`` / ``student_min_lp`` — the student's own average /
  worst log-prob over its sampled tokens. Low values mean the turn itself
  was a low-confidence (high-temperature) sample even before comparing to
  the teacher.
* ``teacher_mean_lp`` — the teacher's average log-prob of the student's
  tokens, i.e. minus the per-token cross-entropy. A teacher-only alternative
  to KL: it needs no student log-probs, which matters if the metric is later
  recomputed at training time from the teacher pass alone.
* ``teacher_min_lp`` / ``teacher_worst_nll`` — the teacher's single worst
  token; ``teacher_worst_nll = -teacher_min_lp`` is the same number oriented
  so that higher = worse, which is what the threshold sweep needs.

Top-k distribution metrics (need live scoring with ``--top-logprobs`` k > 0;
computed per side where that side returned top-k, else absent/None):

* ``student_entropy_mean/max``, ``teacher_entropy_mean/max`` — Shannon
  entropy of the side's next-token distribution at each position, estimated
  from the returned top-k as the exact entropy of those k entries plus the
  leftover probability mass collapsed into a single bucket. That is a lower
  bound on the true entropy, tight when the top-k covers most of the mass.
  Mean = how uncertain the model was across the turn; max = its single most
  uncertain decision point. High teacher entropy on a student turn is itself
  a divergence signal: confident students don't confuse teachers unless the
  state is one the teacher would not have reached.
* ``teacher_entropy_first_{1,2,4,8,16,32}_mean`` — the same teacher entropy,
  averaged over only the first N generated tokens of the turn. These test the
  cheap training variant that scores a short response prefix rather than the
  whole response. Very small N also exposes whether fixed assistant-opening
  tokens make the apparent signal degenerate.
* ``entropy_gap_mean`` = mean(teacher entropy - student entropy) per
  position. Positive = the teacher is more confused than the student on the
  student's own trajectory — the signature of an off-teacher-policy state,
  independent of which particular tokens were sampled.
* ``teacher_margin_mean/max`` — per position, (teacher's top-1 log-prob) -
  (teacher's log-prob of the sampled token); 0 when the student sampled
  exactly what the teacher would have. Unlike d_i this ignores the student's
  own confidence entirely: it asks only "how far below the teacher's
  preferred action was this token", so it is immune to the student being
  generally sharp or flat.
* ``teacher_rank_mean`` — average rank of the sampled token inside the
  teacher's top-k (1 = the teacher's argmax; censored to k+1 when the token
  is below every returned entry, so the value saturates rather than lies).
* ``frac_out_teacher_topk`` — fraction of the turn's tokens that appear
  NOWHERE in the teacher's top-k: the most literal reading of "a student
  action the teacher would never take".
* ``frac_teacher_top1`` — fraction of tokens where the student sampled the
  teacher's argmax: an agreement rate (higher = better, the one
  metric here oriented that way; it is reported but not swept).
* ``student_margin_mean`` — same margin construction on the student side:
  how far below its OWN argmax the student sampled. This isolates pure
  sampling-temperature noise, useful as a control — a turn can look
  divergent simply because temperature-1 sampling picked unlikely tokens
  that neither model preferred.

Threshold sweep: every sweep metric is oriented so that higher = more
divergent, and thresholds are taken at fixed percentiles (p50..p99.5) of the
pooled per-turn distribution. For each (metric, threshold) the report
simulates the figure's cut — the first flagged turn is KEPT (its own prefix
is still teacher-plausible), everything after it is dropped — and reports
fraction of turns flagged, fraction of episodes cut, the mean relative cut
position, the fraction of trained tokens surviving, and the mean reward of
cut vs uncut episodes. A good truncation metric flags few turns, cuts late,
keeps most tokens, and concentrates its cuts on low-reward episodes.

Usage (inside the slime container — needs torch; Vertex scoring needs ADC):

    PYTHONPATH=/mnt/officeqa-rl/src:/mnt/officeqa-rl/third_party/slime \\
    python3 scripts/analyze_opd_turn_kl.py \\
      results/debug_rollout/qwen3.5-4B-grpo-full/rollout_*.pt \\
      --project my-project --location us-east5 \\
      --endpoint-id <endpoint-id> \\
      --hf-checkpoint /mnt/Qwen3.5-4B \\
      --out-dir results/opd_turn_kl/iter69 --plots

    # student re-scored from a local vLLM server instead of dump log-probs:
    #   --student-url http://localhost:8000/v1/completions --student-model qwen

Outputs under --out-dir: ``turns.jsonl`` (one record per trained turn; with
--hf-checkpoint each record also carries ``top_tokens``, the turn's five worst
tokens decoded with their d, teacher log-prob, and the text context that
preceded them), ``summary.json`` (percentiles + threshold sweeps), with
``--dump-tokens`` a ``tokens.jsonl`` holding every turn's full per-token ids
and student/teacher log-prob arrays, and with ``--plots`` three PNGs: metric
ECDFs split by reward, the per-turn-index divergence profile, and the
tokens-kept vs episodes-cut trade-off curve per metric. The printed report
ends with the run's most divergent single tokens in context.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger("analyze_opd_turn_kl")

# Percentiles used both for reporting distributions and as the threshold grid.
_REPORT_PCTS = [50.0, 75.0, 90.0, 95.0, 97.5, 99.0, 99.5]
_FIRST_TOKEN_COUNTS = (1, 2, 4, 8, 16, 32)
_FIRST_TEACHER_ENTROPY_METRICS = [
    f"teacher_entropy_first_{count}_mean" for count in _FIRST_TOKEN_COUNTS
]
# Candidate per-turn metrics for the threshold sweep. Higher value = more
# divergent for every one of them (teacher_min_lp enters negated as
# ``teacher_worst_nll``), so "flag when metric > threshold" is uniform.
# Top-k metrics exist only where the matching side was scored live with
# --top-logprobs > 0; metrics that are None everywhere are skipped.
_SWEEP_METRICS = [
    "kl_mean", "kl_sum", "kl_k3_mean", "d_max", "teacher_worst_nll",
    "teacher_margin_mean", "teacher_margin_max", "frac_out_teacher_topk",
    "teacher_entropy_mean", "entropy_gap_mean", "student_entropy_mean",
    *_FIRST_TEACHER_ENTROPY_METRICS,
]
# Reported in the distribution tables on top of the sweep metrics.
_EXTRA_DIST_METRICS = [
    "teacher_mean_lp", "d_p95", "d_std", "frac_d_gt5",
    "teacher_entropy_max", "teacher_rank_mean", "frac_teacher_top1",
    "student_entropy_max", "student_margin_mean", "student_min_lp",
]


def flatten(node: Any) -> list[dict]:
    if isinstance(node, dict):
        return [node]
    if isinstance(node, (list, tuple)):
        return [sample for child in node for sample in flatten(child)]
    raise TypeError(f"unexpected dump node type: {type(node).__name__}")


def load_samples(paths: list[str]) -> list[dict]:
    import torch

    samples: list[dict] = []
    for path in paths:
        data = torch.load(path, weights_only=False)
        loaded = flatten(data["samples"])
        for sample in loaded:
            sample["_dump_file"] = path
        samples.extend(loaded)
        logger.info("loaded %d samples from %s", len(loaded), path)
    return samples


def trained_spans(mask: list[int]) -> list[tuple[int, int]]:
    """Contiguous loss-mask-1 runs of a sample's response region.

    Each run is one trained turn: consecutive assistant turns are always
    separated by untrained observation tokens in this agent's chat markup, and
    across an episode's fan-out samples every trained turn appears exactly
    once as a mask-1 span (REALIGN re-emissions arrive as mask-0 context).
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for position, bit in enumerate(mask):
        if bit == 1 and start is None:
            start = position
        elif bit != 1 and start is not None:
            spans.append((start, position))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


ScoredSide = tuple[list[float], list[list[float] | None] | None]
"""(sampled log-probs, per-position top-k log-prob lists or None) for one
sample's response region. Tops are None entirely when scored without top-k."""


async def score_samples(
    samples: list[dict],
    client: Any,
    *,
    what: str,
    top_logprobs: int = 0,
    max_in_flight: int = 64,
) -> list[ScoredSide | None]:
    """Teacher/student log-probs of each sample's response tokens (one call per
    sample, ``logprob_start_len = prompt_len - 1``, mirroring the training
    path). Failed samples come back as None and are dropped with a warning
    instead of killing the whole analysis."""
    gate = asyncio.Semaphore(max_in_flight)
    done = 0

    async def one(sample: dict) -> ScoredSide | None:
        nonlocal done
        tokens = list(sample.get("tokens") or [])
        response_length = int(sample.get("response_length") or 0)
        start = max(len(tokens) - response_length - 1, 0)
        affinity = (sample.get("metadata") or {}).get("run_id")
        async with gate:
            try:
                scores = await client.score_tokens(
                    tokens,
                    logprob_start_len=start,
                    affinity_key=affinity,
                    top_logprobs=top_logprobs,
                )
            except Exception as error:
                logger.warning("%s scoring failed for run_id=%s: %s", what, affinity, error)
                return None
        done += 1
        if done % 25 == 0:
            logger.info("%s scoring: %d/%d samples", what, done, len(samples))
        window = scores[-response_length:]
        if top_logprobs:
            return (
                [token.log_prob for token in window],
                [token.top_log_probs for token in window],
            )
        return list(window), None

    return list(await asyncio.gather(*(one(sample) for sample in samples)))


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return math.nan
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(math.floor(rank))
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (rank - low) * (sorted_values[high] - sorted_values[low])


def entropy_lower_bound(top_lps: list[float]) -> float:
    """Entropy of the top-k distribution with the tail collapsed into one
    bucket — a lower bound on the true entropy that tightens as k grows."""
    probs = [math.exp(v) for v in top_lps]
    entropy = -sum(p * v for p, v in zip(probs, top_lps))
    rest = max(0.0, 1.0 - sum(probs))
    if rest > 1e-12:
        entropy -= rest * math.log(rest)
    return entropy


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def topk_side_metrics(
    sampled: list[float], tops: list[list[float] | None] | None, prefix: str
) -> tuple[dict, list[float | None]]:
    """Distribution-aware per-turn statistics for one side (student/teacher).

    Returns (metrics, per-position entropies aligned with the turn; None where
    the server returned no top-k for a position).
    """
    if tops is None:
        return {}, [None] * len(sampled)
    entropies: list[float | None] = []
    margins: list[float] = []
    ranks: list[float] = []
    out_topk: list[bool] = []
    top1: list[bool] = []
    for lp, top in zip(sampled, tops, strict=True):
        if not top:
            entropies.append(None)
            continue
        entropies.append(entropy_lower_bound(top))
        # top is descending; margin = how far the sampled token falls below
        # this side's preferred token (0 when the sampled token IS the top-1).
        margin = max(0.0, top[0] - lp)
        margins.append(margin)
        top1.append(margin < 1e-6)
        # Rank among the returned top-k, censored at k+1 when the sampled
        # token's log-prob sits below every returned entry.
        above = sum(value > lp + 1e-9 for value in top)
        ranks.append(float(min(above + 1, len(top) + 1)))
        out_topk.append(lp < top[-1] - 1e-9)
    known = [entropy for entropy in entropies if entropy is not None]
    metrics = {
        f"{prefix}_entropy_mean": _mean(known),
        f"{prefix}_entropy_max": max(known) if known else None,
        f"{prefix}_margin_mean": _mean(margins),
        f"{prefix}_margin_max": max(margins) if margins else None,
        f"{prefix}_rank_mean": _mean(ranks),
        f"frac_out_{prefix}_topk": _mean([float(flag) for flag in out_topk]),
        f"frac_{prefix}_top1": _mean([float(flag) for flag in top1]),
    }
    return metrics, entropies


def turn_record(
    *,
    sample: dict,
    turn_index: int,
    student: list[float],
    teacher: list[float],
    student_tops: list[list[float] | None] | None,
    teacher_tops: list[list[float] | None] | None,
    top_tokens: list[dict] | None,
) -> dict:
    diffs = [s - t for s, t in zip(student, teacher, strict=True)]
    n = len(diffs)
    sorted_diffs = sorted(diffs)
    kl_mean = sum(diffs) / n
    metadata = sample.get("metadata") or {}
    reward = sample.get("reward")
    record = {
        "dump_file": sample["_dump_file"],
        "run_id": metadata.get("run_id"),
        "uid": metadata.get("uid"),
        "reward": float(reward) if reward is not None else None,
        "exit_status": metadata.get("exit_status"),
        "steps": metadata.get("steps"),
        "turn_index": turn_index,  # 1-based within the episode
        "n_tokens": n,
        "student_mean_lp": sum(student) / n,
        "student_min_lp": min(student),
        "teacher_mean_lp": sum(teacher) / n,
        "teacher_min_lp": min(teacher),
        "teacher_worst_nll": -min(teacher),
        "kl_mean": kl_mean,  # k1 estimate of per-token KL(student||teacher)
        "kl_sum": sum(diffs),  # whole-turn KL estimate
        # k3 estimator exp(-d) - 1 + d: non-negative, lower variance than k1.
        # The exponent is clipped at 50: exp(-d) explodes for very negative d
        # (teacher far more confident than the student on a sampled token).
        "kl_k3_mean": sum(math.exp(min(-d, 50.0)) - 1 + d for d in diffs) / n,
        "d_max": sorted_diffs[-1],
        "d_min": sorted_diffs[0],
        "d_p95": percentile(sorted_diffs, 95.0),
        "d_std": math.sqrt(sum((d - kl_mean) ** 2 for d in diffs) / n),
        "frac_d_gt2": sum(d > 2.0 for d in diffs) / n,
        "frac_d_gt5": sum(d > 5.0 for d in diffs) / n,
    }
    student_metrics, student_entropies = topk_side_metrics(
        student, student_tops, "student"
    )
    teacher_metrics, teacher_entropies = topk_side_metrics(
        teacher, teacher_tops, "teacher"
    )
    record.update(student_metrics)
    record.update(teacher_metrics)
    for count, metric in zip(
        _FIRST_TOKEN_COUNTS, _FIRST_TEACHER_ENTROPY_METRICS, strict=True
    ):
        known = [
            entropy
            for entropy in teacher_entropies[:count]
            if entropy is not None
        ]
        record[metric] = _mean(known)
    # Positive gap = the teacher is more uncertain than the student on the
    # student's own trajectory — a signature of a state the teacher would not
    # have reached.
    gaps = [
        t - s
        for s, t in zip(student_entropies, teacher_entropies, strict=True)
        if s is not None and t is not None
    ]
    record["entropy_gap_mean"] = _mean(gaps)
    if top_tokens is not None:
        record["top_tokens"] = top_tokens
    return record


def build_turn_records(
    samples: list[dict],
    teacher_scores: list[ScoredSide | None],
    student_scores: list[ScoredSide | None] | None,
    tokenizer: Any,
    token_rows: list[dict] | None = None,
) -> list[dict]:
    """``token_rows`` (from --dump-tokens): appended with one row per turn
    carrying the full per-token id/log-prob arrays for custom analysis."""
    records: list[dict] = []
    turn_counters: dict[str, int] = {}
    for index, (sample, teacher_side) in enumerate(
        zip(samples, teacher_scores, strict=True)
    ):
        if teacher_side is None:
            continue
        teacher, teacher_tops = teacher_side
        response_length = int(sample.get("response_length") or 0)
        mask = list(sample.get("loss_mask") or [])
        if student_scores is not None:
            if student_scores[index] is None:
                continue
            student, student_tops = student_scores[index]
        else:
            student = list(sample.get("rollout_log_probs") or [])
            student_tops = None
        if not (len(mask) == len(student) == len(teacher) == response_length):
            logger.warning(
                "sample %d: length mismatch (mask=%d student=%d teacher=%d "
                "response=%d); skipped",
                index, len(mask), len(student), len(teacher), response_length,
            )
            continue
        tokens = list(sample.get("tokens") or [])
        response_tokens = tokens[len(tokens) - response_length :]
        run_id = (sample.get("metadata") or {}).get("run_id") or f"sample{index}"
        for start, end in trained_spans(mask):
            turn_counters[run_id] = turn_counters.get(run_id, 0) + 1
            span_student = student[start:end]
            span_teacher = teacher[start:end]
            top_tokens = None
            if tokenizer is not None:
                diffs = [s - t for s, t in zip(span_student, span_teacher, strict=True)]
                worst = sorted(range(len(diffs)), key=lambda i: -diffs[i])[:5]
                top_tokens = [
                    {
                        "token": tokenizer.decode([response_tokens[start + i]]),
                        "d": round(diffs[i], 3),
                        "teacher_lp": round(span_teacher[i], 3),
                        # Trailing context BEFORE the token, so the reader
                        # sees what the model was in the middle of writing.
                        "ctx": tokenizer.decode(
                            response_tokens[max(0, start + i - 24) : start + i]
                        )[-60:],
                    }
                    for i in worst
                ]
            if token_rows is not None:
                token_row = {
                    "run_id": run_id,
                    "uid": (sample.get("metadata") or {}).get("uid"),
                    "turn_index": turn_counters[run_id],
                    "token_ids": response_tokens[start:end],
                    "student_lp": [round(v, 4) for v in span_student],
                    "teacher_lp": [round(v, 4) for v in span_teacher],
                }
                if teacher_tops is not None:
                    token_row["teacher_entropy"] = [
                        round(entropy_lower_bound(top), 6) if top else None
                        for top in teacher_tops[start:end]
                    ]
                if student_tops is not None:
                    token_row["student_entropy"] = [
                        round(entropy_lower_bound(top), 6) if top else None
                        for top in student_tops[start:end]
                    ]
                token_rows.append(token_row)
            records.append(
                turn_record(
                    sample=sample,
                    turn_index=turn_counters[run_id],
                    student=span_student,
                    teacher=span_teacher,
                    student_tops=(
                        student_tops[start:end] if student_tops is not None else None
                    ),
                    teacher_tops=(
                        teacher_tops[start:end] if teacher_tops is not None else None
                    ),
                    top_tokens=top_tokens,
                )
            )
    return records


def available_metrics(records: list[dict], candidates: list[str]) -> list[str]:
    """Candidates that have at least one non-None value across the records."""
    return [
        metric for metric in candidates
        if any(record.get(metric) is not None for record in records)
    ]


def distribution_summary(records: list[dict]) -> dict:
    """Per-metric percentiles, overall and split by episode reward."""
    groups = {
        "all": records,
        "reward_pos": [r for r in records if (r["reward"] or 0.0) > 0.5],
        "reward_zero": [r for r in records if (r["reward"] or 0.0) <= 0.5],
    }
    summary: dict[str, Any] = {}
    for metric in available_metrics(records, _SWEEP_METRICS + _EXTRA_DIST_METRICS):
        summary[metric] = {}
        for name, group in groups.items():
            values = sorted(
                r[metric] for r in group if r.get(metric) is not None
            )
            summary[metric][name] = {
                "n": len(values),
                "mean": (sum(values) / len(values)) if values else math.nan,
                **{f"p{pct:g}": percentile(values, pct) for pct in _REPORT_PCTS},
            }
    return summary


def sweep_thresholds(records: list[dict]) -> dict:
    """For each metric and each percentile threshold: what a first-flagged-turn
    cut would do to the training stream. The flagged turn itself is kept."""
    episodes: dict[str, list[dict]] = {}
    for record in records:
        episodes.setdefault(record["run_id"], []).append(record)
    for turns in episodes.values():
        turns.sort(key=lambda r: r["turn_index"])

    sweep: dict[str, list[dict]] = {}
    for metric in available_metrics(records, _SWEEP_METRICS):
        values = sorted(r[metric] for r in records if r.get(metric) is not None)
        rows = []
        for pct in _REPORT_PCTS:
            threshold = percentile(values, pct)
            # A turn with no value for this metric is never flagged.
            flagged_turns = sum(
                r.get(metric) is not None and r[metric] > threshold for r in records
            )
            total_tokens = kept_tokens = 0
            cut_episodes = 0
            cut_positions: list[float] = []
            reward_cut: list[float] = []
            reward_uncut: list[float] = []
            for turns in episodes.values():
                first_flagged = next(
                    (
                        i for i, r in enumerate(turns)
                        if r.get(metric) is not None and r[metric] > threshold
                    ),
                    None,
                )
                episode_tokens = sum(r["n_tokens"] for r in turns)
                total_tokens += episode_tokens
                reward = turns[0]["reward"] or 0.0
                if first_flagged is None:
                    kept_tokens += episode_tokens
                    reward_uncut.append(reward)
                else:
                    # Keep turns[0..first_flagged] inclusive, drop the rest.
                    kept_tokens += sum(
                        r["n_tokens"] for r in turns[: first_flagged + 1]
                    )
                    cut_episodes += 1
                    cut_positions.append((first_flagged + 1) / len(turns))
                    reward_cut.append(reward)
            rows.append({
                "pct": pct,
                "threshold": threshold,
                "frac_turns_flagged": flagged_turns / len(records),
                "frac_episodes_cut": cut_episodes / len(episodes),
                "frac_tokens_kept": kept_tokens / total_tokens if total_tokens else 1.0,
                "mean_cut_position": (
                    sum(cut_positions) / len(cut_positions) if cut_positions else None
                ),
                "mean_reward_cut": (
                    sum(reward_cut) / len(reward_cut) if reward_cut else None
                ),
                "mean_reward_uncut": (
                    sum(reward_uncut) / len(reward_uncut) if reward_uncut else None
                ),
            })
        sweep[metric] = rows
    return sweep


def print_report(summary: dict, sweep: dict, n_turns: int, n_episodes: int) -> None:
    print(f"\n{n_turns} trained turns across {n_episodes} episodes")
    header = "metric/group".ljust(36) + "".join(f"p{p:g}".rjust(9) for p in _REPORT_PCTS)
    print("\n== per-turn metric percentiles ==")
    print(header)
    for metric, groups in summary.items():
        for name, stats in groups.items():
            row = f"{metric}/{name}".ljust(36) + "".join(
                f"{stats[f'p{p:g}']:9.3f}" for p in _REPORT_PCTS
            )
            print(row + f"   (n={stats['n']})")
    print("\n== threshold sweep (cut after first flagged turn, flagged turn kept) ==")
    for metric, rows in sweep.items():
        print(f"\n-- {metric} --")
        print(
            "  pct   threshold  turns>thr  episodes-cut  tokens-kept"
            "  cut@pos  R(cut)  R(uncut)"
        )
        for row in rows:
            print(
                f"  {row['pct']:5.1f}  {row['threshold']:9.3f}"
                f"  {row['frac_turns_flagged']:9.1%}"
                f"  {row['frac_episodes_cut']:12.1%}"
                f"  {row['frac_tokens_kept']:11.1%}"
                f"  {row['mean_cut_position'] if row['mean_cut_position'] is not None else float('nan'):7.2f}"
                f"  {row['mean_reward_cut'] if row['mean_reward_cut'] is not None else float('nan'):6.2f}"
                f"  {row['mean_reward_uncut'] if row['mean_reward_uncut'] is not None else float('nan'):8.2f}"
            )


def print_worst_tokens(records: list[dict], n: int = 15) -> None:
    """The n most divergent single tokens across all turns, with the text the
    model was mid-way through writing (needs --hf-checkpoint for decoding)."""
    entries = [
        {**token, "uid": r["uid"], "turn": r["turn_index"], "reward": r["reward"]}
        for r in records
        for token in (r.get("top_tokens") or [])
    ]
    if not entries:
        return
    entries.sort(key=lambda e: -e["d"])
    print(f"\n== most divergent tokens (top {n}) ==")
    print("      d  teacher_lp  R  uid/turn         token         ...context before")
    for e in entries[:n]:
        reward = "1" if (e["reward"] or 0) > 0.5 else "0"
        where = f"{e['uid']}/t{e['turn']}"
        print(
            f"  {e['d']:5.1f}  {e['teacher_lp']:10.2f}  {reward}  {where:15s}"
            f"  {e['token']!r:12s}  ...{e.get('ctx', '')!r}"
        )


# -- plots -------------------------------------------------------------------
# Palette per the repo's dataviz conventions: light surface, categorical
# slot 1 blue / slot 2 orange, text in ink tokens (never the series color).
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e5e4e0"
_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_SERIES = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]


def _style_axis(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_INK_2, labelsize=9)
    ax.set_title(title, color=_INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=_INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=_INK_2, fontsize=9)


def _ecdf(values: list[float]) -> tuple[list[float], list[float]]:
    ordered = sorted(values)
    return ordered, [(i + 1) / len(ordered) for i in range(len(ordered))]


def write_plots(records: list[dict], summary: dict, sweep: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. ECDF of per-turn kl_mean, split by episode reward.
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    clip = summary["kl_mean"]["all"]["p99.5"]
    for values, color, label in (
        ([min(r["kl_mean"], clip) for r in records if (r["reward"] or 0) > 0.5],
         _BLUE, "correct episodes"),
        ([min(r["kl_mean"], clip) for r in records if (r["reward"] or 0) <= 0.5],
         _ORANGE, "wrong episodes"),
    ):
        if values:
            xs, ys = _ecdf(values)
            ax.plot(xs, ys, color=color, linewidth=2, label=f"{label} (n={len(values)})")
    for pct in (90.0, 95.0, 99.0):
        x = summary["kl_mean"]["all"][f"p{pct:g}"]
        ax.axvline(x, color=_GRID, linewidth=1)
        ax.text(x, 0.03, f"p{pct:g}", color=_INK_2, fontsize=8, rotation=90,
                va="bottom", ha="right")
    _style_axis(ax, "Per-turn KL(student‖teacher), k1 estimate — ECDF",
                f"kl_mean (nats/token, clipped at p99.5={clip:.2f})",
                "fraction of turns")
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK_2)
    fig.tight_layout()
    fig.savefig(out_dir / "kl_mean_ecdf.png", facecolor=_SURFACE)
    plt.close(fig)

    # 2. Divergence profile over turn position within the episode.
    by_index: dict[int, list[float]] = {}
    for record in records:
        by_index.setdefault(record["turn_index"], []).append(record["kl_mean"])
    indexes = sorted(i for i, v in by_index.items() if len(v) >= 5)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    medians = [percentile(sorted(by_index[i]), 50.0) for i in indexes]
    p90s = [percentile(sorted(by_index[i]), 90.0) for i in indexes]
    ax.plot(indexes, medians, color=_BLUE, linewidth=2, label="median")
    ax.plot(indexes, p90s, color=_BLUE, linewidth=1.2, linestyle="--", label="p90")
    ax.fill_between(indexes, medians, p90s, color=_BLUE, alpha=0.12, linewidth=0)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    _style_axis(ax, "Per-turn KL by turn position",
                "turn index within episode (positions with ≥5 turns)",
                "kl_mean (nats/token)")
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK_2)
    fig.tight_layout()
    fig.savefig(out_dir / "kl_by_turn_index.png", facecolor=_SURFACE)
    plt.close(fig)

    # 3. Trade-off per metric: episodes cut vs trained tokens kept. The
    # palette carries eight distinguishable series; further metrics stay in
    # the summary tables only.
    plotted = list(sweep)[: len(_SERIES)]
    if len(sweep) > len(_SERIES):
        logger.info(
            "trade-off plot shows the first %d of %d metrics; the rest are in "
            "summary.json", len(_SERIES), len(sweep),
        )
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    for metric, color in zip(plotted, _SERIES):
        rows = sweep[metric]
        xs = [row["frac_episodes_cut"] for row in rows]
        ys = [row["frac_tokens_kept"] for row in rows]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4,
                label=metric)
    _style_axis(ax, "Threshold trade-off (points = percentile thresholds p50…p99.5)",
                "fraction of episodes cut", "fraction of trained tokens kept")
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK_2)
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_tradeoff.png", facecolor=_SURFACE)
    plt.close(fig)
    logger.info("plots written to %s", out_dir)


def make_client(
    url: str, *, backend: str, model: str | None, concurrency: int,
    timeout_sec: float, attempts: int, backoff_sec: float,
    instruction: str | None, hf_checkpoint: str | None,
) -> Any:
    from officeqa_rl.training.vertex_teacher_client import (
        VertexTeacherClient,
        _render_instruction,
    )

    instruction_ids = None
    if instruction:
        if not hf_checkpoint:
            raise SystemExit("--teacher-instruction requires --hf-checkpoint")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)
        instruction_ids = list(
            tokenizer(_render_instruction(instruction), add_special_tokens=False)["input_ids"]
        )
    return VertexTeacherClient(
        teacher_url=url,
        backend=backend,
        model=model,
        max_concurrent=concurrency,
        timeout_sec=timeout_sec,
        max_attempts=attempts,
        backoff_base_sec=backoff_sec,
        instruction_token_ids=instruction_ids,
    )


async def run(args: argparse.Namespace) -> int:
    samples = load_samples(args.dumps)
    samples = [
        sample for sample in samples
        if not (sample.get("metadata") or {}).get("abort_reason")
        and int(sample.get("response_length") or 0) > 0
        and sum(sample.get("loss_mask") or []) > 0
    ]
    if args.limit_samples:
        samples = samples[: args.limit_samples]
    logger.info("%d usable samples after filtering", len(samples))
    if not samples:
        raise SystemExit("no usable samples in the given dumps")

    tokenizer = None
    if args.hf_checkpoint:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_checkpoint, trust_remote_code=True
        )

    clients: list[Any] = []
    try:
        # Teacher log-probs: dump reuse or live Vertex scoring. Dumps carry
        # only sampled-token log-probs, so entropy/margin/rank metrics need
        # live scoring with --top-logprobs > 0.
        if args.use_dumped_teacher:
            teacher_scores: list[ScoredSide | None] = [
                (list(sample["teacher_log_probs"]), None)
                if sample.get("teacher_log_probs") else None
                for sample in samples
            ]
            missing = sum(score is None for score in teacher_scores)
            if missing:
                logger.warning(
                    "%d samples lack dumped teacher_log_probs and are skipped", missing
                )
        else:
            if args.teacher_url:
                teacher_url = args.teacher_url
            elif args.project and args.location and args.endpoint_id:
                from officeqa_rl.training.vertex_teacher_client import rawpredict_url

                teacher_url = ",".join(
                    rawpredict_url(args.project, args.location, endpoint.strip())
                    for endpoint in args.endpoint_id.split(",") if endpoint.strip()
                )
            else:
                raise SystemExit(
                    "teacher source needed: --teacher-url, or --project/--location/"
                    "--endpoint-id, or --use-dumped-teacher"
                )
            teacher_client = make_client(
                teacher_url, backend=args.teacher_backend, model=args.teacher_model,
                concurrency=args.concurrency, timeout_sec=args.timeout_sec,
                attempts=args.attempts, backoff_sec=args.backoff_sec,
                instruction=args.teacher_instruction, hf_checkpoint=args.hf_checkpoint,
            )
            clients.append(teacher_client)
            teacher_scores = await score_samples(
                samples, teacher_client, what="teacher",
                top_logprobs=args.top_logprobs,
            )

        # Student log-probs: dump rollout_log_probs (default) or live re-score.
        student_scores = None
        if args.student_url:
            student_client = make_client(
                args.student_url, backend=args.student_backend,
                model=args.student_model, concurrency=args.concurrency,
                timeout_sec=args.timeout_sec,
                attempts=args.attempts, backoff_sec=args.backoff_sec,
                instruction=None, hf_checkpoint=None,
            )
            clients.append(student_client)
            student_scores = await score_samples(
                samples, student_client, what="student",
                top_logprobs=args.top_logprobs,
            )
    finally:
        for client in clients:
            await client.close()

    token_rows: list[dict] | None = [] if args.dump_tokens else None
    records = build_turn_records(
        samples, teacher_scores, student_scores, tokenizer, token_rows
    )
    if not records:
        raise SystemExit("no trained turns could be scored")
    n_episodes = len({record["run_id"] for record in records})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "turns.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    if token_rows is not None:
        with (out_dir / "tokens.jsonl").open("w", encoding="utf-8") as handle:
            for row in token_rows:
                handle.write(json.dumps(row) + "\n")

    summary = distribution_summary(records)
    sweep = sweep_thresholds(records)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "n_turns": len(records),
                "n_episodes": n_episodes,
                "dumps": args.dumps,
                "student_source": args.student_url or "dump rollout_log_probs",
                "teacher_source": (
                    "dump teacher_log_probs" if args.use_dumped_teacher
                    else "live scoring"
                ),
                "top_logprobs": args.top_logprobs,
                "percentiles": summary,
                "threshold_sweep": sweep,
            },
            indent=2,
        )
    )
    print_report(summary, sweep, len(records), n_episodes)
    print_worst_tokens(records)
    if args.plots:
        write_plots(records, summary, sweep, out_dir)
    print(f"\nper-turn records: {out_dir / 'turns.jsonl'}")
    if token_rows is not None:
        print(f"per-token arrays: {out_dir / 'tokens.jsonl'}")
    print(f"summary:          {out_dir / 'summary.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("dumps", nargs="+", help="rollout_<id>.pt dump files")
    parser.add_argument("--out-dir", default="results/opd_turn_kl/latest")
    parser.add_argument("--hf-checkpoint", default=None,
                        help="tokenizer path; enables per-turn top offending "
                             "tokens in turns.jsonl and --teacher-instruction")
    parser.add_argument("--limit-samples", type=int, default=0,
                        help="analyze only the first N samples (smoke runs)")
    parser.add_argument("--plots", action="store_true",
                        help="write PNG plots next to the JSON outputs")
    parser.add_argument("--dump-tokens", action="store_true",
                        help="also write tokens.jsonl: every turn's full "
                             "per-token ids and student/teacher log-probs")

    teacher = parser.add_argument_group("teacher")
    teacher.add_argument("--teacher-url", default=None,
                         help="full scoring URL(s), comma-separated; overrides "
                              "--project/--location/--endpoint-id")
    teacher.add_argument("--project", default=None)
    teacher.add_argument("--location", default=None)
    teacher.add_argument("--endpoint-id", default=None,
                         help="Vertex endpoint id(s), comma-separated")
    teacher.add_argument("--teacher-backend", default="auto",
                         choices=["auto", "sglang", "vllm"])
    teacher.add_argument("--teacher-model", default=None,
                         help="model name for vLLM-protocol teachers")
    teacher.add_argument("--teacher-instruction", default=None,
                         help="COPD low-resource instruction to condition the "
                              "teacher on (mirrors officeqa_opd_instruction)")
    teacher.add_argument("--use-dumped-teacher", action="store_true",
                         help="reuse teacher_log_probs already in the dump "
                              "(vanilla OPD runs only — COPD contrast dumps "
                              "store contrasts there, not log-probs)")

    student = parser.add_argument_group("student")
    student.add_argument("--student-url", default=None,
                         help="re-score student log-probs against this locally "
                              "hosted server instead of trusting the dump's "
                              "rollout_log_probs (e.g. "
                              "http://localhost:8000/v1/completions)")
    student.add_argument("--student-backend", default="auto",
                         choices=["auto", "sglang", "vllm"])
    student.add_argument("--student-model", default=None)

    parser.add_argument("--concurrency", type=int, default=16,
                        help="max in-flight scoring requests per endpoint")
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--attempts", type=int, default=8,
                        help="retries per scoring request before the sample "
                             "is dropped (exponential backoff between tries)")
    parser.add_argument("--backoff-sec", type=float, default=2.0,
                        help="backoff base: retry N waits ~backoff * 2^(N-1) "
                             "seconds with jitter, capped at 60s")
    parser.add_argument("--top-logprobs", type=int, default=20,
                        help="request the servers' top-k log-probs per scored "
                             "position, enabling entropy/margin/rank metrics "
                             "(0 = sampled-token log-probs only; applies to "
                             "live scoring, not --use-dumped-teacher)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
