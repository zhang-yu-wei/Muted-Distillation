"""Tests for the JobBench rollout hook.

Everything runs without a GPU, Docker, or slime training: a scripted fake
SGLang router serves exact token ids, the sandbox uses the environment's
``local`` runtime (real workspace staging and real shell commands, no
container), and the LLM judge is replaced with a canned score. What stays
real: task loading from a release directory, the mini-swe-agent control flow,
sample building, reward selection, and metadata export.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("slime", reason="requires the vendored slime on PYTHONPATH")

from slime.utils.types import Sample

from officeqa_rl.jobbench.environment import SUBMIT_SENTINEL
from officeqa_rl.jobbench.judge import JobBenchScore, RubricResult, build_score
from officeqa_rl.training import jobbench_metrics, jobbench_rollout, rollout
from tests.sglang_fakes import ByteTokenizer, FakeSGLangServer, sglang_response

WRITE_TURN = (
    "Writing the memo.\n```mswea_bash_command\n"
    'printf "20.2%%\\n" > "$JB_OUTPUT_DIR/memo.md"\n```'
)
SUBMIT_TURN = f"Done.\n```mswea_bash_command\necho {SUBMIT_SENTINEL}\n```"

RUBRICS = {
    "rubrics": [
        {"rubric": "Reports the rate", "weight": 10, "criterion": ["States 20.2%"]},
        {"rubric": "Has a table", "weight": 6, "criterion": ["Table present"]},
    ]
}


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch):
    # jobbench_rollout builds its backend through rollout._create_backend,
    # which resolves the tokenizer via rollout._get_tokenizer at call time.
    monkeypatch.setattr(rollout, "_get_tokenizer", lambda args: ByteTokenizer())


@pytest.fixture
def data_root(tmp_path):
    """A minimal one-task-per-split release the real loader accepts."""
    root = tmp_path / "release"
    for split_dir in ("dataset", "dataset_easy"):
        task_dir = root / split_dir / "statisticians" / "task1"
        folder = task_dir / "task_folder"
        folder.mkdir(parents=True)
        (folder / "TASK_INSTRUCTIONS.txt").write_text("Write a memo.", encoding="utf-8")
        (folder / "cohort.csv").write_text("year,events\n2006,12\n", encoding="utf-8")
        (task_dir / "RUBRICS.json").write_text(json.dumps(RUBRICS), encoding="utf-8")
    return root


# One rubric item passed, one failed: reward 10/16.
CANNED_RESULTS = [
    RubricResult(0, "Reports the rate", 10, True, 10, 1, 1),
    RubricResult(1, "Has a table", 6, False, 0, 1, 0),
]


@pytest.fixture
def canned_judge(monkeypatch):
    """Replace the judge; record the output directory it was asked to grade."""
    graded: list[Path] = []

    async def fake_judge(task, output_dir, config, *, client=None):
        graded.append(Path(output_dir))
        return build_score(
            task.uid,
            list(CANNED_RESULTS),
            judge_model="canned",
            n_output_files=len(list(Path(output_dir).rglob("*"))),
        )

    monkeypatch.setattr(jobbench_rollout, "judge_episode", fake_judge)
    return graded


def make_args(server, tmp_path, **overrides):
    values = dict(
        hf_checkpoint="fake-actor",
        sglang_router_ip=server.host,
        sglang_router_port=server.port,
        rollout_max_context_len=16384,
        eval_max_context_len=16384,
        officeqa_workspace_root=str(tmp_path / "workspaces"),
        officeqa_jobbench_step_limit=6,
        officeqa_jobbench_sandbox_runtime="local",
        officeqa_jobbench_keep_workspaces=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_base_sample(data_root, split="easy", **overrides):
    values = dict(
        index=3,
        group_index=1,
        rollout_id=7,
        prompt="Write a memo.",
        label=f"JB__{split}__statisticians__task1",
        metadata={
            "uid": f"JB__{split}__statisticians__task1",
            "benchmark": "jobbench",
            "split": split,
            "occupation": "statisticians",
            "task_id": "task1",
            "data_root": str(data_root),
        },
    )
    values.update(overrides)
    return Sample(**values)


def scripted_responder(texts):
    queue = list(texts)

    async def responder(payload):
        return sglang_response(queue.pop(0))

    return responder


@pytest.fixture(autouse=True)
def judge_key(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "test-key")


async def test_reward_is_the_normalized_rubric_score(data_root, tmp_path, canned_judge):
    async with FakeSGLangServer(scripted_responder([WRITE_TURN, SUBMIT_TURN])) as server:
        args = make_args(server, tmp_path)
        result = await jobbench_rollout.generate(
            args, make_base_sample(data_root), {"max_new_tokens": 256}
        )

    samples = result if isinstance(result, list) else [result]
    assert all(sample.status == Sample.Status.COMPLETED for sample in samples)
    assert samples[0].reward == pytest.approx(10 / 16)
    metadata = samples[0].metadata
    assert metadata["exit_status"] == "Submitted"
    assert metadata["submitted"] is True
    assert metadata["n_output_files"] == 1
    assert metadata["jobbench_metrics"]["passed_count"] == 1
    assert metadata["jobbench_metrics"]["max_score"] == 16
    assert metadata["episode_output_tokens"] > 0
    assert samples[0].response_length > 0

    # Only the deliverable directory is graded, and the workspace is gone.
    assert len(canned_judge) == 1
    assert canned_judge[0].name == "output"
    assert not canned_judge[0].parent.exists()


async def test_step_limit_truncates_without_a_deliverable(
    data_root, tmp_path, canned_judge
):
    keep_going = "Looking around.\n```mswea_bash_command\nls\n```"
    async with FakeSGLangServer(scripted_responder([keep_going] * 3)) as server:
        args = make_args(server, tmp_path, officeqa_jobbench_step_limit=3)
        result = await jobbench_rollout.generate(
            args, make_base_sample(data_root), {"max_new_tokens": 256}
        )

    samples = result if isinstance(result, list) else [result]
    assert all(sample.status == Sample.Status.TRUNCATED for sample in samples)
    assert samples[0].metadata["exit_status"] == "LimitsExceeded"
    assert samples[0].metadata["n_output_files"] == 0


async def test_judge_failure_aborts_instead_of_scoring_zero(
    data_root, tmp_path, monkeypatch
):
    async def failing_judge(task, output_dir, config, *, client=None):
        score = build_score(
            task.uid,
            [
                RubricResult(
                    0, "Reports the rate", 10, False, 0, 1, 0, error="judge_error: boom"
                )
            ],
            judge_model="canned",
            n_output_files=1,
        )
        assert isinstance(score, JobBenchScore)
        return score

    monkeypatch.setattr(jobbench_rollout, "judge_episode", failing_judge)
    async with FakeSGLangServer(scripted_responder([WRITE_TURN, SUBMIT_TURN])) as server:
        args = make_args(server, tmp_path)
        sample = await jobbench_rollout.generate(
            args, make_base_sample(data_root), {"max_new_tokens": 256}
        )

    assert sample.status == Sample.Status.ABORTED
    assert sample.remove_sample is True
    assert sample.reward == 0.0


async def test_missing_judge_key_aborts_the_sample(
    data_root, tmp_path, canned_judge, monkeypatch
):
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    async with FakeSGLangServer(scripted_responder([WRITE_TURN, SUBMIT_TURN])) as server:
        args = make_args(server, tmp_path)
        sample = await jobbench_rollout.generate(
            args, make_base_sample(data_root), {"max_new_tokens": 256}
        )

    assert sample.status == Sample.Status.ABORTED
    assert not canned_judge  # never reached the judge


async def test_non_jobbench_rows_go_to_the_fallback_hook(data_root, tmp_path):
    forwarded = []

    async def fake_fallback(args, base_sample, sampling_params, evaluation=False):
        forwarded.append(base_sample.metadata.get("benchmark"))
        return base_sample

    jobbench_rollout._FALLBACK_GENERATES["tests.fallback"] = fake_fallback
    sample = Sample(
        index=0,
        group_index=0,
        rollout_id=0,
        prompt="p",
        label="l",
        metadata={"benchmark": "officeqa"},
    )
    args = SimpleNamespace(
        officeqa_jobbench_fallback_generate_path="tests.fallback",
    )
    returned = await jobbench_rollout.generate(args, sample, {})
    assert returned is sample
    assert forwarded == ["officeqa"]


def test_metrics_summarize_episodes(data_root):
    sample = make_base_sample(data_root)
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {
        **sample.metadata,
        "run_id": "r1",
        "submitted": True,
        "steps": 4,
        "n_output_files": 2,
        "jobbench_metrics": {"normalized_score": 0.625, "pass_rate": 0.5},
    }
    metrics = jobbench_metrics._jobbench_metrics([sample])
    assert metrics["jobbench/episodes"] == 1.0
    assert metrics["jobbench/submitted_ratio"] == 1.0
    assert metrics["jobbench/empty_output_ratio"] == 0.0
    assert metrics["jobbench/normalized_score"] == pytest.approx(0.625)
    assert metrics["jobbench/rubric_pass_rate"] == pytest.approx(0.5)
    assert metrics["jobbench/steps"] == 4.0


def test_metrics_ignore_aborted_episodes(data_root):
    sample = make_base_sample(data_root)
    sample.status = Sample.Status.ABORTED
    assert jobbench_metrics._jobbench_metrics([sample]) == {}


def test_judge_config_allows_adc_for_vertex_endpoints(monkeypatch):
    """A Vertex judge authenticates with ADC, so an empty key is not an error."""
    from officeqa_rl.training import jobbench_rollout

    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    vertex = SimpleNamespace(
        officeqa_jobbench_judge_api_base=(
            "https://aiplatform.googleapis.com/v1/projects/p/locations/global/"
            "endpoints/openapi"
        ),
        officeqa_jobbench_judge_model="google/gemini-3.5-flash",
    )
    config = jobbench_rollout.jobbench_judge_config_from_args(vertex)
    assert config.api_key == ""
    assert config.model == "google/gemini-3.5-flash"

    # A key-authenticated endpoint still requires the key.
    keyed = SimpleNamespace(officeqa_jobbench_judge_api_base="https://api.x.ai/v1")
    with pytest.raises(ValueError, match="judge API key"):
        jobbench_rollout.jobbench_judge_config_from_args(keyed)


def _episode_sample(data_root, run_id, *, score, reward_available=True):
    sample = make_base_sample(data_root)
    sample.status = Sample.Status.COMPLETED
    sample.reward = score
    sample.metadata = {
        **sample.metadata,
        "run_id": run_id,
        "submitted": True,
        "steps": 4,
        "n_output_files": 2,
        "reward_available": reward_available,
        "jobbench_metrics": {"normalized_score": score, "pass_rate": score},
    }
    return sample


def test_ungraded_episodes_stay_in_the_averages_and_are_reported_separately(
    data_root,
):
    """Averages cover every episode generated; the ratio says how many were
    grading outages rather than outcomes.

    Keeping the denominator fixed is what makes these comparable across
    rollouts, and because an ungraded episode contributes a reward of exactly
    0.0, the graded-only mean stays recoverable as
    ``reward_mean / (1 - ungraded_ratio)``.
    """
    from officeqa_rl.training import metrics as shared_metrics

    graded = _episode_sample(data_root, "r1", score=0.6)
    ungraded = _episode_sample(data_root, "r2", score=0.0, reward_available=False)
    samples = [graded, ungraded]

    summary = shared_metrics._truncation_metrics(
        samples, max_response_len=1024, max_context_len=16384
    )
    assert summary["episode/reward_mean"] == pytest.approx(0.3)
    assert summary["episode/ungraded_ratio"] == pytest.approx(0.5)
    # The correction the ratio exists to support.
    recovered = summary["episode/reward_mean"] / (1 - summary["episode/ungraded_ratio"])
    assert recovered == pytest.approx(0.6)

    metrics = jobbench_metrics._jobbench_metrics(samples)
    assert metrics["jobbench/episodes"] == 2.0
    assert metrics["jobbench/normalized_score"] == pytest.approx(0.3)
    assert metrics["jobbench/steps"] == 4.0


def test_ungraded_ratio_is_zero_when_every_episode_graded(data_root):
    from officeqa_rl.training import metrics as shared_metrics

    samples = [
        _episode_sample(data_root, "r1", score=0.6),
        _episode_sample(data_root, "r2", score=0.2),
    ]

    summary = shared_metrics._truncation_metrics(
        samples, max_response_len=1024, max_context_len=16384
    )

    assert summary["episode/ungraded_ratio"] == 0.0
    assert summary["episode/reward_mean"] == pytest.approx(0.4)


def test_ungraded_ratio_defaults_to_zero_for_harnesses_without_the_flag(data_root):
    """OfficeQA and SSB1 samples carry no ``reward_available``; the shared hook
    must read that as graded rather than as a failure."""
    from officeqa_rl.training import metrics as shared_metrics

    sample = _episode_sample(data_root, "r1", score=0.6)
    sample.metadata = {k: v for k, v in sample.metadata.items() if k != "reward_available"}

    summary = shared_metrics._truncation_metrics(
        [sample], max_response_len=1024, max_context_len=16384
    )

    assert summary["episode/ungraded_ratio"] == 0.0
    assert summary["episode/reward_mean"] == pytest.approx(0.6)
