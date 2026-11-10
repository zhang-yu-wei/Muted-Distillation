"""JobBench dataset loading, sandbox staging, agent episode, and judging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from officeqa_rl.jobbench.agent import JobBenchAgentConfig, run_jobbench_episode
from officeqa_rl.jobbench.dataset import (
    EVAL_SPLIT,
    TRAIN_SPLIT,
    SearchFilePolicy,
    load_jobbench_tasks,
    load_task_by_uid,
    make_uid,
)
from officeqa_rl.jobbench.environment import (
    SUBMIT_SENTINEL,
    JobBenchEnvironment,
    JobBenchEnvironmentConfig,
)
from officeqa_rl.jobbench.extraction import collect_image_attachments
from officeqa_rl.jobbench.judge import (
    JudgeConfig,
    RubricResult,
    build_score,
    judge_episode,
    parse_judge_json,
)
from officeqa_rl.types import GeneratedTurn


RUBRICS = {
    "rubrics": [
        {
            "rubric": "Does the memo report the 20.2% event rate?",
            "weight": 10,
            "criterion": ["States 20.2%", "Compares it to the 15% assumption"],
        },
        {
            "rubric": "Is there a summary table?",
            "weight": 6,
            "criterion": ["A table of counts is present"],
        },
    ]
}


def _make_release(root: Path) -> Path:
    """Build a two-task release with the on-disk layout the loader expects."""
    for split_dir, occupation, instructions in (
        ("dataset", "statisticians", "Write memo.docx summarising the cohort."),
        ("dataset_easy", "statisticians", "Write a short memo."),
    ):
        task_dir = root / split_dir / occupation / "task1"
        folder = task_dir / "task_folder"
        folder.mkdir(parents=True)
        (folder / "TASK_INSTRUCTIONS.txt").write_text(instructions, encoding="utf-8")
        (folder / "cohort.csv").write_text("year,events\n2006,12\n", encoding="utf-8")
        (task_dir / "RUBRICS.json").write_text(json.dumps(RUBRICS), encoding="utf-8")
        (task_dir / "task_card.md").write_text("# brief\n", encoding="utf-8")
    search_dir = root / "dataset" / "statisticians" / "task1" / "files_required_to_search"
    search_dir.mkdir(parents=True)
    (search_dir / "guidance.txt").write_text("FDA guidance\n", encoding="utf-8")
    return root


@pytest.fixture
def release(tmp_path: Path) -> Path:
    return _make_release(tmp_path / "release")


class _ScriptedBackend:
    """Replays a fixed list of model responses, one per turn."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[list[dict[str, Any]]] = []

    async def generate(self, messages, sampling_params):  # noqa: ANN001
        self.prompts.append([dict(message) for message in messages])
        text = self.responses.pop(0) if self.responses else "no action here"
        return GeneratedTurn(text=text, finish_reason="stop")


def _command(command: str) -> str:
    return f"Doing the next step.\n\n```mswea_bash_command\n{command}\n```"


# ------------------------------------------------------------------- dataset


def test_load_tasks_reads_rubrics_and_reference_files(release: Path) -> None:
    tasks = load_jobbench_tasks(release, EVAL_SPLIT)
    assert len(tasks) == 1
    task = tasks[0]
    assert task.uid == make_uid("main", "statisticians", "task1")
    assert task.instructions.startswith("Write memo.docx")
    assert task.local_reference_files == ("cohort.csv",)
    assert task.search_files == ("guidance.txt",)
    assert [rubric.weight for rubric in task.rubrics] == [10, 6]
    assert task.rubrics[0].criteria == (
        "States 20.2%",
        "Compares it to the 15% assumption",
    )
    assert task.max_score == 16


def test_easy_split_has_no_search_files(release: Path) -> None:
    task = load_jobbench_tasks(release, TRAIN_SPLIT)[0]
    assert task.split == "easy"
    assert task.search_files == ()


def test_load_task_by_uid_round_trips(release: Path) -> None:
    task = load_jobbench_tasks(release, EVAL_SPLIT)[0]
    assert load_task_by_uid(release, task.uid).instructions == task.instructions


def test_load_task_by_uid_rejects_unknown_split(release: Path) -> None:
    with pytest.raises(ValueError):
        load_task_by_uid(release, "JB__nope__statisticians__task1")


# --------------------------------------------------------------- environment


def _env_config(tmp_path: Path, **overrides: Any) -> JobBenchEnvironmentConfig:
    settings: dict[str, Any] = {
        "workspace_root": tmp_path / "workspaces",
        "sandbox_runtime": "local",
        "sandbox_image": None,
        "command_timeout_sec": 30,
    }
    settings.update(overrides)
    return JobBenchEnvironmentConfig(**settings)


def test_workspace_stages_task_folder_but_never_the_rubrics(
    release: Path, tmp_path: Path
) -> None:
    task = load_jobbench_tasks(release, EVAL_SPLIT)[0]
    with JobBenchEnvironment(_env_config(tmp_path), task, run_id="r1") as env:
        staged = {path.name for path in env.workspace.rglob("*") if path.is_file()}
        assert staged == {"TASK_INSTRUCTIONS.txt", "cohort.csv"}
        assert env.host_output_dir.is_dir()
        assert not (env.workspace / "RUBRICS.json").exists()
        assert not (env.workspace / "task_card.md").exists()
        # files_required_to_search is omitted by default, as upstream does
        assert not (env.host_task_folder / "guidance.txt").exists()


def test_search_file_policy_stage_copies_them_in(release: Path, tmp_path: Path) -> None:
    task = load_jobbench_tasks(release, EVAL_SPLIT)[0]
    config = _env_config(tmp_path, search_file_policy=SearchFilePolicy.STAGE)
    with JobBenchEnvironment(config, task, run_id="r1") as env:
        assert (env.host_task_folder / "guidance.txt").is_file()
        assert env.get_info()["staged_search_files"] == ["guidance.txt"]


def test_workspace_path_is_absolute_for_the_docker_mount(
    release: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A relative -v source is read as a named volume, not a host directory.
    monkeypatch.chdir(tmp_path)
    task = load_jobbench_tasks(release, EVAL_SPLIT)[0]
    config = JobBenchEnvironmentConfig(
        workspace_root=Path("relative/workspaces"), sandbox_runtime="local"
    )
    env = JobBenchEnvironment(config, task, run_id="r1")
    assert env.workspace.is_absolute()


def test_commands_run_in_the_task_folder(release: Path, tmp_path: Path) -> None:
    task = load_jobbench_tasks(release, EVAL_SPLIT)[0]
    with JobBenchEnvironment(_env_config(tmp_path), task, run_id="r1") as env:
        output = env.execute({"command": "ls"})
        assert output["returncode"] == 0
        assert "cohort.csv" in output["output"]


def test_submit_is_refused_until_a_deliverable_exists(
    release: Path, tmp_path: Path
) -> None:
    from officeqa_rl.jobbench._msa import Submitted

    task = load_jobbench_tasks(release, EVAL_SPLIT)[0]
    with JobBenchEnvironment(_env_config(tmp_path), task, run_id="r1") as env:
        refused = env.execute({"command": f"echo {SUBMIT_SENTINEL}"})
        assert refused["returncode"] == 1
        assert "is empty" in refused["output"]

        env.execute({"command": 'echo report > "$JB_OUTPUT_DIR/memo.md"'})
        with pytest.raises(Submitted):
            env.execute({"command": f"echo {SUBMIT_SENTINEL}"})


def test_observations_are_truncated(release: Path, tmp_path: Path) -> None:
    task = load_jobbench_tasks(release, EVAL_SPLIT)[0]
    config = _env_config(tmp_path, max_observation_chars=200)
    with JobBenchEnvironment(config, task, run_id="r1") as env:
        output = env.execute({"command": "python3 -c \"print('x' * 5000)\""})
        assert len(output["output"]) < 400
        assert "characters elided" in output["output"]


# --------------------------------------------------------------------- agent


@pytest.mark.asyncio
async def test_episode_runs_writes_a_deliverable_and_submits(
    release: Path, tmp_path: Path
) -> None:
    task = load_jobbench_tasks(release, TRAIN_SPLIT)[0]
    backend = _ScriptedBackend(
        [
            _command("cat TASK_INSTRUCTIONS.txt"),
            _command('printf "20.2%% event rate\\n" > "$JB_OUTPUT_DIR/memo.md"'),
            _command(f"echo {SUBMIT_SENTINEL}"),
        ]
    )
    episode = await run_jobbench_episode(
        task,
        backend,
        JobBenchAgentConfig(step_limit=8),
        _env_config(tmp_path),
        run_id="r-agent",
    )
    assert episode.exit_status == "Submitted"
    assert episode.submitted
    assert episode.output_files == ["memo.md"]
    assert episode.steps == 3
    assert len(episode.generated_turns) == 3
    # The task instructions and both prompt templates reached the model.
    system_prompt = backend.prompts[0][0]["content"]
    assert "mswea_bash_command" in system_prompt
    assert task.instructions in backend.prompts[0][1]["content"]
    # Harness bookkeeping never rides along in a prompt.
    assert all("extra" not in message for prompt in backend.prompts for message in prompt)


@pytest.mark.asyncio
async def test_episode_stops_at_the_step_limit(release: Path, tmp_path: Path) -> None:
    task = load_jobbench_tasks(release, TRAIN_SPLIT)[0]
    backend = _ScriptedBackend([_command("true")] * 10)
    episode = await run_jobbench_episode(
        task,
        backend,
        JobBenchAgentConfig(step_limit=3),
        _env_config(tmp_path),
        run_id="r-limit",
    )
    assert episode.exit_status == "LimitsExceeded"
    assert episode.steps == 3
    assert episode.output_files == []


@pytest.mark.asyncio
async def test_repeated_format_errors_end_the_episode(
    release: Path, tmp_path: Path
) -> None:
    task = load_jobbench_tasks(release, TRAIN_SPLIT)[0]
    backend = _ScriptedBackend(["no action at all"] * 6)
    episode = await run_jobbench_episode(
        task,
        backend,
        JobBenchAgentConfig(step_limit=10, max_consecutive_format_errors=2),
        _env_config(tmp_path),
        run_id="r-format",
    )
    assert episode.exit_status == "RepeatedFormatError"
    assert episode.steps == 2


# --------------------------------------------------------------------- judge


def test_parse_judge_json_accepts_a_fenced_object() -> None:
    parsed = parse_judge_json('```json\n{"rubric_passed": true}\n```')
    assert parsed == {"rubric_passed": True}


def test_malformed_notebook_outputs_do_not_abort_image_collection(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    notebook = {
        "cells": [
            "not-a-cell",
            {
                "cell_type": "code",
                "source": [],
                "outputs": ["not-an-output", {"data": "not-a-mime-map"}],
            },
        ]
    }
    (output_dir / "malformed.ipynb").write_text(json.dumps(notebook), encoding="utf-8")

    assert collect_image_attachments(output_dir) == []


def test_build_score_is_weighted_and_all_or_nothing_per_item() -> None:
    results = [
        RubricResult(0, "a", 10, True, 10, 2, 2),
        RubricResult(1, "b", 6, False, 0, 1, 0),
    ]
    score = build_score("uid", results, judge_model="m", n_output_files=1)
    assert score.total_score == 10
    assert score.max_score == 16
    assert score.normalized_score == pytest.approx(0.625)
    assert score.pass_rate == pytest.approx(0.5)
    assert score.reward == score.normalized_score
    assert score.graded


@pytest.mark.asyncio
async def test_empty_output_scores_zero_without_calling_the_judge(
    release: Path, tmp_path: Path
) -> None:
    task = load_jobbench_tasks(release, EVAL_SPLIT)[0]
    empty = tmp_path / "output"
    empty.mkdir()
    score = await judge_episode(task, empty, JudgeConfig(api_key="unused"))
    assert score.normalized_score == 0.0
    assert score.reason == "no_output_files"
    # A genuine zero, not a grading failure: the sample stays trainable.
    assert score.graded


@pytest.mark.asyncio
async def test_judge_scores_each_rubric_item(release: Path, tmp_path: Path) -> None:
    task = load_jobbench_tasks(release, EVAL_SPLIT)[0]
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "memo.md").write_text("The rate was 20.2%.\n", encoding="utf-8")

    calls: list[str] = []

    class _StubCompletions:
        async def create(self, **kwargs: Any) -> Any:
            content = kwargs["messages"][1]["content"]
            calls.append(content)
            # Decide on the rubric description alone: the deliverable's text is
            # pasted into every rubric prompt, so matching the whole prompt
            # would pass both items.
            description = content.split("## Criteria to Evaluate")[0]
            passed = "20.2% event rate" in description
            payload = {
                "criteria_results": [
                    {"index": index, "passed": passed, "reasoning": "", "evidence": ""}
                    for index in range(content.count("Criterion "))
                ],
                "rubric_passed": passed,
                "overall_reasoning": "stub",
            }
            message = type("M", (), {"content": json.dumps(payload)})()
            choice = type("C", (), {"message": message})()
            return type("R", (), {"choices": [choice]})()

    class _StubClient:
        chat = type("Chat", (), {"completions": _StubCompletions()})()

    score = await judge_episode(
        task, output_dir, JudgeConfig(api_key="unused"), client=_StubClient()
    )
    assert len(calls) == 2  # one call per rubric item
    assert score.passed_count == 1  # only the item whose text mentions 20.2%
    assert score.total_score == 10
    assert score.normalized_score == pytest.approx(0.625)
    assert score.graded
    # The deliverable's text is what the judge saw.
    assert "The rate was 20.2%." in calls[0]


@pytest.mark.asyncio
async def test_judge_failures_are_counted_not_hidden(
    release: Path, tmp_path: Path
) -> None:
    task = load_jobbench_tasks(release, EVAL_SPLIT)[0]
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "memo.md").write_text("nothing useful", encoding="utf-8")

    class _FailingCompletions:
        async def create(self, **kwargs: Any) -> Any:
            raise RuntimeError("judge endpoint down")

    class _FailingClient:
        chat = type("Chat", (), {"completions": _FailingCompletions()})()

    config = JudgeConfig(api_key="unused", max_retries=1, retry_delay_sec=0.0)
    score = await judge_episode(task, output_dir, config, client=_FailingClient())
    assert score.judge_failures == 2
    assert not score.graded
    assert score.normalized_score == 0.0


def test_judge_require_key_uses_adc_only_for_vertex_bases(monkeypatch):
    from officeqa_rl.jobbench import judge as judge_module

    monkeypatch.setattr(judge_module, "_adc_token", lambda: "adc-token")
    vertex = judge_module.JudgeConfig(
        api_base=(
            "https://aiplatform.googleapis.com/v1/projects/p/locations/global/"
            "endpoints/openapi"
        ),
        api_key="",
    )
    assert vertex.require_key() == "adc-token"

    regional = judge_module.JudgeConfig(
        api_base="https://us-east5-aiplatform.googleapis.com/v1/x", api_key=""
    )
    assert regional.require_key() == "adc-token"

    xai = judge_module.JudgeConfig(api_base="https://api.x.ai/v1", api_key="")
    with pytest.raises(RuntimeError, match="No judge API key"):
        xai.require_key()

    # A static key always wins, Vertex base or not.
    keyed = judge_module.JudgeConfig(
        api_base="https://aiplatform.googleapis.com/v1", api_key="sk-static"
    )
    assert keyed.require_key() == "sk-static"


def test_container_names_stay_unique_for_long_uids():
    from officeqa_rl.jobbench.environment import _CONTAINER_NAME_MAX, _container_name

    # Longest real occupation slug on the main split; the old cap truncated the
    # run-id/suffix tail away, so concurrent samples collided on one name.
    uid = "JB__main__training_and_development_specialists__task2"
    names = {_container_name(uid, "88fd307775c8") for _ in range(64)}
    assert len(names) == 64
    for name in names:
        assert len(name) <= _CONTAINER_NAME_MAX
        assert name.startswith("jobbench-jb-main-training")
        assert name.split("-")[-2] == "88fd3077"

    short = _container_name("JB__easy__statisticians__task1", "abcdef012345")
    assert short.startswith("jobbench-jb-easy-statisticians-task1-abcdef01-")


def test_workspace_names_preserve_unique_tail_for_long_uids():
    from officeqa_rl.jobbench.environment import _WORKSPACE_NAME_MAX, _workspace_name

    uid = (
        "JB__main__sales_representatives_wholesale_and_manufacturing_"
        "technical_and_scientific_products__task1"
    )
    first = _workspace_name(uid, "abcdef012345")
    second = _workspace_name(uid, "9876543210ab")
    assert first != second
    # The run_id survives truncation, but is no longer the final segment: a
    # uuid suffix now follows it, because run_id alone repeats across episodes
    # that share an index (see the collision test below).
    assert "-abcdef012345-" in first
    assert "-9876543210ab-" in second
    assert len(first) <= _WORKSPACE_NAME_MAX


# --- judge retry -----------------------------------------------------------
# A rubric that never grades still counts its weight in the denominator, so a
# judge failure biases the attempt's score DOWN. These pin the retry policy.

def test_retryable_classification_separates_transient_from_permanent():
    from officeqa_rl.jobbench.judge import _is_retryable

    for transient in (
        RuntimeError("HTTP 503: Service Unavailable"),
        RuntimeError("429 rate limit exceeded"),
        TimeoutError("request timed out"),
        ConnectionError("connection reset"),
        RuntimeError("model overloaded, try again"),
    ):
        assert _is_retryable(transient), transient

    for permanent in (
        RuntimeError("HTTP 400: invalid_argument"),
        RuntimeError("HTTP 404: publisher model not found"),
        RuntimeError("HTTP 403: permission denied"),
    ):
        assert not _is_retryable(permanent), permanent

    # A 429 wrapped alongside a permanent-looking code is still worth retrying.
    assert _is_retryable(RuntimeError("HTTP 429 after 403 refresh"))


def test_judge_config_retry_defaults_match_the_vertex_teacher_posture():
    """Three attempts at 2s was too thin for the endpoint the teacher needs
    18 attempts at 20s to survive."""
    from officeqa_rl.jobbench.judge import JudgeConfig

    c = JudgeConfig()
    assert c.max_retries >= 8
    assert c.retry_delay_sec >= 4.0
    assert c.retry_max_delay_sec >= 30.0


def test_rollout_reads_the_retry_knobs_from_args():
    """retry_delay_sec used to be dataclass-only, so the yaml could not set it."""
    from types import SimpleNamespace

    from officeqa_rl.training.jobbench_rollout import jobbench_judge_config_from_args

    cfg = jobbench_judge_config_from_args(
        SimpleNamespace(
            officeqa_jobbench_judge_api_base=(
                "https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi"
            ),
            officeqa_jobbench_judge_max_retries=11,
            officeqa_jobbench_judge_retry_delay_sec=7.5,
            officeqa_jobbench_judge_retry_max_delay_sec=90.0,
        )
    )
    assert cfg.max_retries == 11
    assert cfg.retry_delay_sec == 7.5
    assert cfg.retry_max_delay_sec == 90.0


def test_workspace_name_is_unique_when_run_id_truncates():
    """Two episodes sharing an index must not build the same workspace path.

    ``run_id`` is ``idx<i>-grp<g>-rid<r>-<hash>``, so a short index leaves
    ``run_id[:12]`` as just ``idx34-grp34-`` and the distinguishing hash is
    truncated away. ``_prepare_workspace`` creates the directory with
    ``exist_ok=False``, so a duplicate name kills the episode outright --
    observed 2026-09-04 taking every episode of two consecutive online-SFT
    rounds.
    """
    from officeqa_rl.jobbench.environment import _WORKSPACE_NAME_MAX, _workspace_name

    uid = "JB__main__computer_and_information_research_scientists__task1"
    same_index = [
        "idx34-grp34-ridNone-35d7b461",
        "idx34-grp34-ridNone-c5085e08",
    ]
    names = [_workspace_name(uid, run_id) for run_id in same_index]
    assert names[0] != names[1]
    assert all(len(name) <= _WORKSPACE_NAME_MAX for name in names)
    # The uid must still be recognisable in the path for debugging.
    assert all(name.startswith("JB__main__computer") for name in names)
