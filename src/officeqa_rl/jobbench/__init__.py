"""JobBench: long-horizon professional office tasks under the mini-swe-agent harness.

JobBench gives an agent a folder of source documents (CSV, XLSX, DOCX, PDF,
SQLite) and a professional brief, and scores the deliverables it writes against
an expert-authored weighted rubric. Tasks come from 35 white-collar
occupations; the main split averages 10.6 input files per task.

Layout of this package:

===================  ======================================================
``dataset``          task loading and the easy-train / main-eval split policy
``environment``      per-episode Docker workspace, mini-swe-agent ``Environment``
``models``           mini-swe-agent ``Model`` over this repo's backends
``prompts``          system and instance templates
``agent``            assembles upstream ``DefaultAgent`` and runs one episode
``extraction``       deliverables to judge-readable text and images
``judge``            per-rubric LLM judging, weighted score, reward
===================  ======================================================

See :mod:`officeqa_rl.jobbench.judge` for what the LLM-judge reward does and
does not measure.
"""

from officeqa_rl.jobbench.agent import (
    JobBenchAgentConfig,
    JobBenchEpisode,
    run_jobbench_episode,
)
from officeqa_rl.jobbench.dataset import (
    EVAL_SPLIT,
    TRAIN_SPLIT,
    JobBenchRubric,
    JobBenchTask,
    SearchFilePolicy,
    load_jobbench_tasks,
    load_task_by_uid,
)
from officeqa_rl.jobbench.environment import (
    JobBenchEnvironment,
    JobBenchEnvironmentConfig,
)
from officeqa_rl.jobbench.judge import JobBenchScore, JudgeConfig, judge_episode
from officeqa_rl.jobbench.models import BackendModel, JobBenchModelConfig

__all__ = [
    "BackendModel",
    "EVAL_SPLIT",
    "JobBenchAgentConfig",
    "JobBenchEnvironment",
    "JobBenchEnvironmentConfig",
    "JobBenchEpisode",
    "JobBenchModelConfig",
    "JobBenchRubric",
    "JobBenchScore",
    "JobBenchTask",
    "JudgeConfig",
    "SearchFilePolicy",
    "TRAIN_SPLIT",
    "judge_episode",
    "load_jobbench_tasks",
    "load_task_by_uid",
    "run_jobbench_episode",
]
