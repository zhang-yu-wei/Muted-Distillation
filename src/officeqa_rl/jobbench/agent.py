"""Run one JobBench episode with the vendored mini-swe-agent control flow.

``DefaultAgent`` is used unmodified: this module only wires it to a
``JobBenchEnvironment``, a ``BackendModel``, and the JobBench prompt templates,
then runs the synchronous loop in a worker thread so callers stay async.

The workspace outlives the episode on purpose. Grading reads
``<workspace>/output/`` after the container is gone, so the caller decides when
to delete it (``JobBenchEpisode.workspace``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from officeqa_rl.backends.protocol import GenerationBackend
from officeqa_rl.jobbench._msa import (
    MINI_SWE_AGENT_VERSION,
    DefaultAgent,
    LimitsExceeded,
)
from officeqa_rl.jobbench.dataset import JobBenchTask
from officeqa_rl.jobbench.environment import (
    JobBenchEnvironment,
    JobBenchEnvironmentConfig,
)
from officeqa_rl.jobbench.models import BackendModel, JobBenchModelConfig, LoopBridge
from officeqa_rl.jobbench.prompts import INSTANCE_TEMPLATE, SYSTEM_TEMPLATE
from officeqa_rl.token_usage import episode_usage
from officeqa_rl.types import GeneratedTurn, Message

logger = logging.getLogger("officeqa_rl.jobbench.agent")

#: Threads available to run concurrent synchronous agent loops. Each episode
#: occupies one for its whole lifetime, blocked on a container command or on a
#: generation call handed back to the event loop, so this is the real ceiling
#: on episode concurrency. asyncio's default executor caps at
#: ``min(32, cpu_count + 4)``, which would silently serialize a 48-episode eval
#: sweep or a 256-episode rollout batch.
_EPISODE_THREADS = int(os.getenv("OFFICEQA_JOBBENCH_EPISODE_THREADS", "512"))
_EXECUTOR: ThreadPoolExecutor | None = None


def episode_executor() -> ThreadPoolExecutor:
    """Return the shared pool that runs synchronous agent loops."""
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(
            max_workers=max(1, _EPISODE_THREADS), thread_name_prefix="jobbench-episode"
        )
    return _EXECUTOR


@dataclass(frozen=True)
class JobBenchAgentConfig:
    """Budgets and templates for one episode."""

    step_limit: int = 60
    wall_time_limit_seconds: int = 0
    max_consecutive_format_errors: int = 3
    system_template: str = SYSTEM_TEMPLATE
    instance_template: str = INSTANCE_TEMPLATE


@dataclass
class JobBenchEpisode:
    """Everything one episode produced, before grading."""

    uid: str
    run_id: str
    exit_status: str
    submission: str
    messages: list[Message]
    steps: int
    workspace: Path
    output_files: list[str]
    generated_turns: list[GeneratedTurn]
    turn_prompts: list[list[Message]]
    environment: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def submitted(self) -> bool:
        return self.exit_status == "Submitted"


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def build_agent(
    backend: GenerationBackend,
    env: JobBenchEnvironment,
    config: JobBenchAgentConfig,
    *,
    model_config: JobBenchModelConfig | None = None,
    bridge: LoopBridge | None = None,
) -> DefaultAgent:
    """Assemble the upstream agent with this repository's model and sandbox."""
    model = BackendModel(backend, model_config, bridge=bridge)
    return DefaultAgent(
        model,
        env,
        system_template=config.system_template,
        instance_template=config.instance_template,
        step_limit=config.step_limit,
        cost_limit=0.0,  # backends report no cost; the step limit is the budget
        wall_time_limit_seconds=config.wall_time_limit_seconds,
        max_consecutive_format_errors=config.max_consecutive_format_errors,
    )


def _run_sync(agent: DefaultAgent, task: JobBenchTask) -> dict[str, Any]:
    try:
        return agent.run(task=task.instructions)
    except LimitsExceeded:  # already recorded as an exit message by the agent
        return agent.messages[-1].get("extra", {})
    except Exception as error:  # the agent records the exit message, then re-raises
        logger.warning("JobBench episode %s crashed: %s", task.uid, error)
        return agent.messages[-1].get("extra", {}) if agent.messages else {
            "exit_status": type(error).__name__,
            "submission": "",
        }


async def run_jobbench_episode(
    task: JobBenchTask,
    backend: GenerationBackend,
    agent_config: JobBenchAgentConfig,
    env_config: JobBenchEnvironmentConfig,
    *,
    run_id: str | None = None,
    model_config: JobBenchModelConfig | None = None,
    boot_semaphore: asyncio.Semaphore | None = None,
    executor: ThreadPoolExecutor | None = None,
) -> JobBenchEpisode:
    """Run one episode and return its transcript, tokens, and workspace."""
    run_id = run_id or new_run_id()
    env = JobBenchEnvironment(env_config, task, run_id=run_id)
    loop = asyncio.get_running_loop()
    bridge = LoopBridge(loop)
    pool = executor or episode_executor()

    try:
        if boot_semaphore is not None:
            async with boot_semaphore:
                await loop.run_in_executor(pool, env.start)
        else:
            await loop.run_in_executor(pool, env.start)
    except BaseException:
        # Nothing will grade this workspace, so do not leave it behind: a
        # failed boot during a long rollout would otherwise accumulate one
        # task-folder copy per attempt on the training host.
        env.cleanup()
        shutil.rmtree(env.workspace, ignore_errors=True)
        raise

    agent = build_agent(
        backend, env, agent_config, model_config=model_config, bridge=bridge
    )
    try:
        exit_extra = await loop.run_in_executor(pool, _run_sync, agent, task)
    except BaseException:
        env.cleanup()
        shutil.rmtree(env.workspace, ignore_errors=True)
        raise
    finally:
        env.cleanup()

    model: BackendModel = agent.model  # type: ignore[assignment]
    output_files = [
        path.relative_to(env.host_output_dir).as_posix() for path in env.output_files()
    ]
    return JobBenchEpisode(
        uid=task.uid,
        run_id=run_id,
        exit_status=str(exit_extra.get("exit_status") or "Unknown"),
        submission=str(exit_extra.get("submission") or ""),
        messages=list(agent.messages),
        steps=model.n_calls,
        workspace=env.workspace,
        output_files=output_files,
        generated_turns=list(model.generated_turns),
        turn_prompts=list(model.turn_prompts),
        environment=env.get_info(),
        metadata={
            "mini_swe_agent_version": MINI_SWE_AGENT_VERSION,
            "split": task.split,
            "occupation": task.occupation,
            "task_id": task.task_id,
            "step_limit": agent_config.step_limit,
            "token_usage": episode_usage(model.generated_turns),
            "n_output_files": len(output_files),
        },
    )


__all__ = [
    "JobBenchAgentConfig",
    "JobBenchEpisode",
    "build_agent",
    "episode_executor",
    "new_run_id",
    "run_jobbench_episode",
]
