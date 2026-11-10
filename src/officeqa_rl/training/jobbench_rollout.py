"""Slime per-sample rollout hook for JobBench RL training.

This is the ``--custom-generate-function-path
officeqa_rl.training.jobbench_rollout.generate`` entry point, a sibling of the
OfficeQA and SpreadsheetBench hooks rather than a modification of either: it
imports their shared pieces (backend construction, sample building, abort
semantics) as plain functions, and a JobBench run selects it in its launcher.

Row contract (produced by ``scripts/build_jobbench_dataset.py``): the sample's
``metadata`` carries ``benchmark: "jobbench"``, ``uid``, and ``data_root``. The
task — instructions, reference files, and rubrics — is reloaded from
``data_root`` by uid, so the rubric (which states the answers outright) never
rides in the row. Rows with a different ``metadata.benchmark`` are forwarded
unchanged to the hook named by ``officeqa_jobbench_fallback_generate_path``.

Episode and reward follow the offline eval protocol
(``scripts/run_jobbench_eval.py``): the vendored mini-swe-agent
``DefaultAgent`` works in a per-episode sandbox holding a copy of the task
folder and an empty ``output/``; after the episode the LLM judge scores
``output/`` rubric item by rubric item and the reward is the normalized
weighted score, the same number the official JobBench judge reports.

Two consequences of an LLM-judge reward, both handled here:

- **Grading can fail.** A judge outage would otherwise inject reward 0 on a
  good episode. When any rubric item could not be graded the sample is aborted
  (``scoring_failure``) instead of trained on.
- **Grading costs money and latency.** One judge call per rubric item, ~9 per
  episode, on every rollout sample. ``officeqa_jobbench_judge_concurrency``
  caps calls per episode; the number of episodes in flight is slime's
  rollout batch size. Budget accordingly before scaling a run.

Training uses the ``easy`` split (63 tasks, every referenced file inside the
task folder) and evaluation the ``main`` split; see
``src/officeqa_rl/jobbench/dataset.py`` for why.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Awaitable, Callable
from weakref import WeakKeyDictionary

from slime.utils.types import Sample

from officeqa_rl.jobbench.agent import (
    JobBenchAgentConfig,
    JobBenchEpisode,
    run_jobbench_episode,
)
from officeqa_rl.jobbench.dataset import SearchFilePolicy, load_task_by_uid
from officeqa_rl.jobbench.environment import (
    OUTPUT_DIR_NAME,
    JobBenchEnvironmentConfig,
)
from officeqa_rl.jobbench.judge import (
    JobBenchScore,
    JudgeConfig,
    is_vertex_judge_api_base,
    judge_episode,
)
from officeqa_rl.jobbench.models import JobBenchModelConfig
from officeqa_rl.training.opd_rollout import (
    _TurnScorer,
    build_scored_samples,
    parse_per_turn_modes,
)
from officeqa_rl.training.rollout import (
    _abort,
    _attach_teacher_log_probs,
    _create_backend,
    _episode_output_tokens,
    _log_teacher_failure,
    _make_run_id,
    _require_arg,
    _teacher_scoring_enabled,
    build_training_samples,
)

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_IMAGE = "jobbench-sandbox"
DEFAULT_FALLBACK_GENERATE_PATH = "officeqa_rl.training.rollout.generate"
DEFAULT_CONTAINER_MEMORY_LIMIT = "8g"
DEFAULT_STEP_LIMIT = 60

#: mini-swe-agent exit statuses that end a well-formed episode. Anything else
#: (an environment crash, an unhandled model error) is an infrastructure
#: failure and aborts the sample rather than scoring it zero.
_STATUS_BY_EXIT: dict[str, Any] = {
    "Submitted": Sample.Status.COMPLETED,
    "LimitsExceeded": Sample.Status.TRUNCATED,
    "TimeExceeded": Sample.Status.TRUNCATED,
    "RepeatedFormatError": Sample.Status.TRUNCATED,
}

GenerateFn = Callable[..., Awaitable[Sample | list[Sample]]]

_FALLBACK_GENERATES: dict[str, GenerateFn] = {}
_BOOT_SEMAPHORES: WeakKeyDictionary[Any, dict[int, asyncio.Semaphore]] = (
    WeakKeyDictionary()
)


def _fallback_generate(args: Any) -> GenerateFn:
    dotted = str(
        getattr(
            args, "officeqa_jobbench_fallback_generate_path", DEFAULT_FALLBACK_GENERATE_PATH
        )
    )
    if dotted not in _FALLBACK_GENERATES:
        module_name, _, attribute = dotted.rpartition(".")
        _FALLBACK_GENERATES[dotted] = getattr(
            importlib.import_module(module_name), attribute
        )
    return _FALLBACK_GENERATES[dotted]


def _boot_semaphore(limit: int) -> asyncio.Semaphore:
    """One shared sandbox-boot limiter per event loop, as the SSB1 hook does."""
    loop = asyncio.get_running_loop()
    semaphores = _BOOT_SEMAPHORES.setdefault(loop, {})
    if limit not in semaphores:
        semaphores[limit] = asyncio.Semaphore(limit)
    return semaphores[limit]


def jobbench_agent_config_from_args(args: Any) -> JobBenchAgentConfig:
    return JobBenchAgentConfig(
        step_limit=int(getattr(args, "officeqa_jobbench_step_limit", DEFAULT_STEP_LIMIT)),
        wall_time_limit_seconds=int(
            getattr(args, "officeqa_jobbench_wall_time_limit_sec", 0)
        ),
        max_consecutive_format_errors=int(
            getattr(args, "officeqa_jobbench_max_consecutive_format_errors", 3)
        ),
    )


def jobbench_environment_config_from_args(args: Any) -> JobBenchEnvironmentConfig:
    return JobBenchEnvironmentConfig(
        workspace_root=Path(_require_arg(args, "officeqa_workspace_root")),
        sandbox_image=str(
            getattr(args, "officeqa_jobbench_sandbox_image", DEFAULT_SANDBOX_IMAGE)
        ),
        # "local" runs commands as plain subprocesses in the workspace. It
        # exists for tests and single-node debugging on a host without Docker;
        # it gives the policy's code no isolation, so never train with it.
        sandbox_runtime=str(
            getattr(args, "officeqa_jobbench_sandbox_runtime", "docker")
        ),
        command_timeout_sec=int(
            getattr(args, "officeqa_jobbench_command_timeout_sec", 180)
        ),
        max_observation_chars=int(
            getattr(args, "officeqa_max_observation_chars", 12_000)
        ),
        container_memory_limit=str(
            getattr(
                args,
                "officeqa_jobbench_container_memory_limit",
                DEFAULT_CONTAINER_MEMORY_LIMIT,
            )
        ),
        search_file_policy=SearchFilePolicy(
            str(
                getattr(
                    args,
                    "officeqa_jobbench_search_file_policy",
                    SearchFilePolicy.OMIT.value,
                )
            )
        ),
    )


def jobbench_judge_config_from_args(args: Any) -> JudgeConfig:
    defaults = JudgeConfig()
    key_env = str(getattr(args, "officeqa_jobbench_judge_api_key_env", "JUDGE_API_KEY"))
    api_key = os.getenv(key_env, "")
    api_base = str(getattr(args, "officeqa_jobbench_judge_api_base", defaults.api_base))
    if not api_key and not is_vertex_judge_api_base(api_base):
        # A judge on Vertex AI's OpenAI-compatible endpoint authenticates with
        # short-lived ADC tokens instead of a static key, so an empty key is
        # only an error for the key-authenticated endpoints.
        raise ValueError(
            f"JobBench rollouts need a judge API key in ${key_env}; export it on "
            "every training node (rollout workers grade their own episodes). "
            "A Vertex AI judge endpoint uses ADC instead and needs no key."
        )
    return JudgeConfig(
        model=str(getattr(args, "officeqa_jobbench_judge_model", defaults.model)),
        api_base=api_base,
        api_key=api_key,
        timeout_sec=float(getattr(args, "officeqa_jobbench_judge_timeout_sec", 300.0)),
        max_retries=int(getattr(args, "officeqa_jobbench_judge_max_retries", 8)),
        # These two were dataclass-only before, so the yaml could not reach
        # them however they were set.
        retry_delay_sec=float(
            getattr(args, "officeqa_jobbench_judge_retry_delay_sec", 4.0)
        ),
        retry_max_delay_sec=float(
            getattr(args, "officeqa_jobbench_judge_retry_max_delay_sec", 60.0)
        ),
        max_concurrent_rubrics=int(
            getattr(args, "officeqa_jobbench_judge_concurrency", 4)
        ),
        max_completion_tokens=int(
            getattr(args, "officeqa_jobbench_judge_max_completion_tokens", 8192)
        ),
        attach_images=bool(getattr(args, "officeqa_jobbench_judge_images", True)),
        # Sampling temperature of the judge (default 0.0, greedy). Greedy
        # decoding is what loops a reasoning judge into spending its whole
        # completion budget without a verdict; 1.0 breaks the loop at the cost
        # of verdict variance.
        temperature=float(
            getattr(args, "officeqa_jobbench_judge_temperature", defaults.temperature)
        ),
        truncation_fallback_reasoning_effort=str(
            getattr(
                args,
                "officeqa_jobbench_judge_truncation_fallback_effort",
                defaults.truncation_fallback_reasoning_effort,
            )
            or ""
        ),
    )


def _compact_score(score: JobBenchScore) -> dict[str, Any]:
    """Per-episode reward decomposition small enough to ride on every sample."""
    return {
        "normalized_score": score.normalized_score,
        "total_score": score.total_score,
        "max_score": score.max_score,
        "pass_rate": score.pass_rate,
        "passed_count": score.passed_count,
        "total_count": score.total_count,
        "n_output_files": score.n_output_files,
        "judge_failures": score.judge_failures,
        "judge_model": score.judge_model,
        "reason": score.reason,
        # One line per rubric item (verdict, error, first words of the judge's
        # reasoning) so a rollout that grades zero everywhere can be told apart
        # from a judge that silently failed every item (2026-09-13: the 9B
        # pure arm's rollout 13 scored 0 on 16/16 episodes with no logged
        # judge error and left nothing to inspect).
        "rubric_verdicts": [
            f"{'P' if r.passed else 'F'}"
            + (f" err={r.error[:60]}" if r.error else "")
            + f" | {(r.overall_reasoning or '')[:100]}"
            for r in getattr(score, "rubrics", [])[:12]
        ],
    }


async def generate(
    args: Any,
    base_sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    """Run one JobBench episode, judge it, and export slime training samples."""
    metadata = dict(base_sample.metadata or {})
    if metadata.get("benchmark") != "jobbench":
        return await _fallback_generate(args)(
            args, base_sample, sampling_params, evaluation=evaluation
        )

    data_root = str(metadata.get("data_root") or "").strip()
    uid = str(metadata.get("uid") or "").strip()
    if not data_root or not uid:
        return _abort(base_sample, "invalid_task: metadata needs data_root and uid")

    run_id = _make_run_id(base_sample)
    try:
        task = await asyncio.to_thread(
            load_task_by_uid, Path(data_root).expanduser().resolve(), uid
        )
    except (ValueError, FileNotFoundError) as error:
        return _abort(base_sample, f"invalid_task: {error}", run_id=run_id)

    routing_key = (
        base_sample.session_id
        if getattr(args, "router_policy", None) == "consistent_hashing"
        else None
    )
    try:
        backend = _create_backend(args, evaluation=evaluation, routing_key=routing_key)
        agent_config = jobbench_agent_config_from_args(args)
        env_config = jobbench_environment_config_from_args(args)
        judge_config = jobbench_judge_config_from_args(args)
        model_config = JobBenchModelConfig(
            model_name=str(getattr(args, "hf_checkpoint", "") or "policy"),
            sampling_params=dict(sampling_params),
        )
    except (TypeError, ValueError) as error:
        return _abort(base_sample, f"invalid_rollout_config: {error}", run_id=run_id)

    # Per-turn OPD (the hill-climb arms) on JobBench: the same scorer, turn
    # plans and sample construction as the OfficeQA and SSB1 paths in
    # :mod:`officeqa_rl.training.opd_rollout`. Selected by the yaml knob
    # ``officeqa_opd_per_turn``; without it the per-sample path below stays
    # exactly as before. The engine's three assumptions all hold here: the
    # shared SlimeSGLangBackend records ``session_turns``, the fenced-command
    # regex accepts mini-swe-agent's ``mswea_bash_command`` tag (its language
    # class is ``[A-Za-z0-9_-]*``), and the Qwen chat template opens
    # ``<think>`` in the prompt so turns carry a think region to mask to.
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

    boot_limit = int(getattr(args, "officeqa_max_concurrent_sandbox_boots", 4))
    episode: JobBenchEpisode | None = None
    try:
        episode = await run_jobbench_episode(
            task,
            backend,
            agent_config,
            env_config,
            run_id=run_id,
            model_config=model_config,
            boot_semaphore=_boot_semaphore(max(1, boot_limit)),
        )
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

    keep_workspace = bool(getattr(args, "officeqa_jobbench_keep_workspaces", False))
    try:
        terminal_status = _STATUS_BY_EXIT.get(episode.exit_status)
        if terminal_status is None:
            if scorer is not None:
                await scorer.cancel()
            return _abort(
                base_sample, f"agent_failure: {episode.exit_status}", run_id=run_id
            )

        score = None
        grading_failure: str | None = None
        try:
            score = await judge_episode(
                task, episode.workspace / OUTPUT_DIR_NAME, judge_config
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # judging infrastructure, not the policy
            grading_failure = f"scoring_failure: {type(error).__name__}: {error}"
        else:
            if not score.graded:
                grading_failure = (
                    "scoring_failure: judge failed on "
                    f"{score.judge_failures} rubric items"
                )

        if grading_failure is not None and scorer is None:
            # Reward-only run: the task reward is the entire learning signal,
            # so an ungradeable episode carries none. Its zero would read as a
            # policy failure in the advantage, so drop the sample.
            return _abort(base_sample, grading_failure, run_id=run_id)

        if grading_failure is not None:
            # OPD run: the teacher's per-token log-probs are independent of the
            # judge, so this trajectory still carries a full distillation
            # signal and dropping it would throw away every turn of teacher
            # supervision over a grading outage. Keep it and let the KL term
            # train it exactly as it would any other episode. The placeholder
            # reward below never reaches an advantage: ``reward_available:
            # False`` excludes this execution from its group's reward
            # centering and pins its own reward advantage to 0
            # (slime/ray/rollout.py::_normalize_rewards_per_rollout).
            logger.warning(
                "JobBench episode kept for distillation without a task reward "
                "(run_id=%s): %s",
                run_id,
                grading_failure,
            )

        reward = 0.0 if grading_failure is not None else float(score.normalized_score)
        extra_metadata = {
            **metadata,
            "uid": task.uid,
            "run_id": run_id,
            "exit_status": episode.exit_status,
            "submitted": episode.submitted,
            "steps": episode.steps,
            "n_output_files": len(episode.output_files),
            "jobbench_metrics": None if score is None else _compact_score(score),
            "episode_output_tokens": _episode_output_tokens(backend.session_turns),
            # Consumed by _normalize_rewards_per_rollout (reward centering) and
            # by metrics._truncation_metrics (episode/reward_mean), so a
            # placeholder reward neither trains nor is reported as a measured 0.
            "reward_available": grading_failure is None,
            "grading_failure": grading_failure,
        }

        if scorer is not None:
            # Teacher-scoring failures are deliberately fatal, exactly as in
            # the OfficeQA and SSB1 per-turn paths: log the reason, then stop
            # the run. The same goes for score/sample misalignment inside the
            # construction.
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
            # Which scoring path this episode actually took. A run whose
            # per-turn knobs are set in the yaml but which still shows
            # scorer=false has failed the gate rather than the mode logic.
            if isinstance(sample.metadata, dict):
                sample.metadata["opd_path"] = {
                    "scorer": scorer is not None,
                    "evaluation": bool(evaluation),
                    "teacher_enabled": bool(_teacher_scoring_enabled(args)),
                    "per_turn": bool(
                        getattr(args, "officeqa_opd_per_turn", False)
                    ),
                    "use_opd": bool(getattr(args, "use_opd", False)),
                    "opd_type": str(getattr(args, "opd_type", None)),
                }

        # Same OPD semantics as the OfficeQA and SSB1 per-sample paths: eval
        # rollouts skip the teacher, and a terminal teacher-scoring failure is
        # logged and then fatal, never a silent zero-mask.
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
    finally:
        if episode is not None and not keep_workspace:
            await asyncio.to_thread(
                shutil.rmtree, episode.workspace, ignore_errors=True
            )


__all__ = [
    "DEFAULT_FALLBACK_GENERATE_PATH",
    "DEFAULT_SANDBOX_IMAGE",
    "generate",
    "jobbench_agent_config_from_args",
    "jobbench_environment_config_from_args",
    "jobbench_judge_config_from_args",
]
