"""Tests for the SpreadsheetBench v1 rollout hook.

Everything runs without a GPU, Docker, or slime training: a scripted fake
SGLang router serves exact token ids, a fake SSB1 environment stands in for
the Docker sandbox (which the real ``SpreadsheetBenchV1Environment``
requires), and the replay/recalc grading pipeline is replaced with canned
official-shape metrics. What stays real: task loading from a release
directory, the agent loop, sample building, reward selection, and metadata
export.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("slime", reason="requires the vendored slime on PYTHONPATH")

from slime.utils.types import Sample

from officeqa_rl.spreadsheetbench_v1_scoring import aggregate_v1_metrics
from officeqa_rl.training import rollout, ssb1_metrics, ssb1_rollout
from officeqa_rl.types import StepOutput, Submitted
from tests.sglang_fakes import ByteTokenizer, FakeSGLangServer, sglang_response

WRITE_TURN = "Writing the program.\n```bash\ncat > solution.py <<'EOF'\nprint(1)\nEOF\n```"
SUBMIT_TURN = "Verified.\n```bash\nsubmit\n```"


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch):
    # ssb1_rollout builds its backend through rollout._create_backend, which
    # resolves the tokenizer via rollout._get_tokenizer at call time.
    monkeypatch.setattr(rollout, "_get_tokenizer", lambda args: ByteTokenizer())


@pytest.fixture
def data_root(tmp_path):
    """A minimal one-task release the real loader accepts."""
    root = tmp_path / "release"
    task_dir = root / "spreadsheet" / "T1"
    task_dir.mkdir(parents=True)
    (task_dir / "1_T1_input.xlsx").write_bytes(b"")
    (task_dir / "1_T1_answer.xlsx").write_bytes(b"")
    (root / "dataset.json").write_text(
        json.dumps(
            [
                {
                    "id": "T1",
                    "instruction": "Fill the answer column.",
                    "spreadsheet_path": "spreadsheet/T1",
                    "instruction_type": "Cell-Level Manipulation",
                    "answer_position": "A1:A2",
                }
            ]
        ),
        encoding="utf-8",
    )
    return root


class FakeSSB1Environment:
    """Stands in for the Docker-only SpreadsheetBenchV1Environment: same
    surface the hook and agent use, no containers."""

    def __init__(self, config, task, *, run_id):
        self.config = config
        self.ssb_task = task
        self.run_id = run_id
        self.workspace = Path(config.workspace_root) / f"{task.uid}--{run_id}"

    @property
    def solution_host_path(self):
        return self.workspace / "solution.py"

    async def __aenter__(self):
        self.workspace.mkdir(parents=True)
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def get_task_message(self):
        return {"role": "user", "content": f"Instruction: {self.ssb_task.instruction}"}

    async def execute(self, action):
        command = action.get("command", "") if isinstance(action, dict) else action
        if command.strip().startswith("submit"):
            if self.solution_host_path.is_file():
                raise Submitted("solution.py")
            return StepOutput("Cannot submit yet; missing solution.py.")
        self.solution_host_path.write_text("print(1)\n", encoding="utf-8")
        return StepOutput("Exit code: 0")

    def get_info(self):
        return {"uid": self.ssb_task.uid, "run_id": self.run_id}


# Official-shape metrics with hard != soft so tests can tell which one the
# reward used: case 1 fully correct, case 2 wrong in one of two cells.
CANNED_CASE_METRICS = [
    {"accuracy": 1.0, "cell_accuracy": 1.0, "cells": {"correct": 2, "total": 2},
     "cell_errors": [], "cell_error_count": 0, "error": "", "case_index": 1,
     "replay": {"exit_code": 0, "timed_out": False, "output_produced": True,
                "stdout": "", "stderr": ""}},
    {"accuracy": 0.0, "cell_accuracy": 0.5, "cells": {"correct": 1, "total": 2},
     "cell_errors": ["Value difference at Sheet1!A2"], "cell_error_count": 1,
     "error": "", "case_index": 2,
     "replay": {"exit_code": 0, "timed_out": False, "output_produced": True,
                "stdout": "", "stderr": "boom"}},
]


@pytest.fixture
def canned_scoring(monkeypatch):
    calls = []

    async def fake_score(args, task, solution_path, scratch_root):
        assert solution_path.is_file()
        calls.append(task.task_id)
        return aggregate_v1_metrics(CANNED_CASE_METRICS)

    monkeypatch.setattr(ssb1_rollout, "_score_submission", fake_score)
    monkeypatch.setattr(
        ssb1_rollout, "_TrainingSSB1Environment", FakeSSB1Environment
    )
    return calls


def make_args(server, tmp_path, **overrides):
    values = dict(
        hf_checkpoint="fake-actor",
        sglang_router_ip=server.host,
        sglang_router_port=server.port,
        rollout_max_context_len=8192,
        eval_max_context_len=8192,
        officeqa_workspace_root=str(tmp_path / "workspaces"),
        officeqa_ssb1_step_limit=4,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_base_sample(data_root, **overrides):
    values = dict(
        index=3,
        group_index=1,
        rollout_id=7,
        prompt="Fill the answer column.",
        label="A1:A2",
        metadata={
            "uid": "SSB1__T1",
            "benchmark": "ssb1",
            "task_id": "T1",
            "instruction_type": "Cell-Level Manipulation",
            "answer_position": "A1:A2",
            "case_count": 1,
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


async def test_training_reward_is_soft_restriction(data_root, tmp_path, canned_scoring):
    async with FakeSGLangServer(scripted_responder([WRITE_TURN, SUBMIT_TURN])) as server:
        args = make_args(server, tmp_path)
        sample = await ssb1_rollout.generate(
            args, make_base_sample(data_root), {"max_new_tokens": 512}
        )

    assert isinstance(sample, Sample)
    assert sample.status == Sample.Status.COMPLETED
    assert canned_scoring == ["T1"]
    # soft restriction: 1 of 2 canned cases fully correct.
    assert sample.reward == 0.5
    assert sample.metadata["exit_status"] == "submitted"
    assert sample.metadata["ssb1_reward_metric"] == "soft_restriction"
    scores = sample.metadata["ssb1_metrics"]
    assert scores["hard_restriction"] == 0.0
    assert scores["soft_restriction"] == 0.5
    assert scores["cell_accuracy"] == 0.75
    assert scores["test_case_results"] == [1, 0]
    # Compaction: per-case diagnostics survive, cell error lists do not.
    assert [case["case_index"] for case in scores["cases"]] == [1, 2]
    assert scores["cases"][1]["stderr_tail"] == "boom"
    assert all("cell_errors" not in case for case in scores["cases"])

    # The exported sample trains on exactly the served turns.
    assert sample.response_length > 0
    assert sample.loss_mask is not None
    assert any(mask == 0 for mask in sample.loss_mask)  # observation region


async def test_eval_reward_is_hard_restriction(data_root, tmp_path, canned_scoring):
    async with FakeSGLangServer(scripted_responder([WRITE_TURN, SUBMIT_TURN])) as server:
        args = make_args(server, tmp_path)
        sample = await ssb1_rollout.generate(
            args, make_base_sample(data_root), {"max_new_tokens": 512}, evaluation=True
        )

    # Same canned metrics, but eval scores strict: hard restriction is 0.0
    # while training would have paid 0.5.
    assert sample.reward == 0.0
    assert sample.metadata["ssb1_reward_metric"] == "hard_restriction"
    assert sample.metadata["ssb1_metrics"]["soft_restriction"] == 0.5

    # Eval episodes carry the exact generation cost too, so the metrics hook
    # can log a measured episode/output_tokens_mean instead of leaving the
    # figure to the mean-turn-length x steps estimate.
    # ByteTokenizer emits one token per character, so the episode's exact cost
    # is the two scripted turns' texts.
    assert sample.metadata["episode_output_tokens"] == len(WRITE_TURN) + len(
        SUBMIT_TURN
    )


async def test_step_limit_yields_truncated_zero_reward(
    data_root, tmp_path, canned_scoring
):
    async with FakeSGLangServer(
        scripted_responder([WRITE_TURN, WRITE_TURN])
    ) as server:
        args = make_args(server, tmp_path, officeqa_ssb1_step_limit=2)
        sample = await ssb1_rollout.generate(
            args, make_base_sample(data_root), {"max_new_tokens": 512}
        )

    assert sample.status == Sample.Status.TRUNCATED
    assert sample.reward == 0.0
    assert sample.metadata["exit_status"] == "limits_exceeded"
    assert "ssb1_metrics" not in sample.metadata
    assert canned_scoring == []  # no replay work for unsubmitted episodes


async def test_unknown_task_becomes_removable_abort(data_root, tmp_path, canned_scoring):
    async with FakeSGLangServer(scripted_responder([SUBMIT_TURN])) as server:
        args = make_args(server, tmp_path)
        sample = await ssb1_rollout.generate(
            args,
            make_base_sample(data_root, metadata={
                "uid": "SSB1__missing",
                "benchmark": "ssb1",
                "task_id": "missing",
                "data_root": str(data_root),
            }),
            {"max_new_tokens": 512},
        )

    assert sample.status == Sample.Status.ABORTED
    assert sample.remove_sample is True
    assert sample.metadata["abort_reason"].startswith("invalid_task:")


async def test_non_ssb1_rows_are_forwarded(data_root, tmp_path, monkeypatch):
    forwarded = []

    async def recorder(args, base_sample, sampling_params, evaluation=False):
        forwarded.append((base_sample.metadata.get("uid"), evaluation))
        return base_sample

    monkeypatch.setattr(ssb1_rollout, "_fallback_generate", lambda args: recorder)
    base_sample = Sample(prompt="q", label="42", metadata={"uid": "UID0001"})
    result = await ssb1_rollout.generate(
        SimpleNamespace(), base_sample, {}, evaluation=True
    )

    assert result is base_sample
    assert forwarded == [("UID0001", True)]


def test_default_fallback_is_the_officeqa_hook():
    assert ssb1_rollout._fallback_generate(SimpleNamespace()) is rollout.generate


async def test_task_table_loads_once_per_root(data_root, monkeypatch):
    calls = []
    real_loader = ssb1_rollout.load_spreadsheetbench_v1_tasks

    def counting_loader(root):
        calls.append(root)
        return real_loader(root)

    monkeypatch.setattr(
        ssb1_rollout, "load_spreadsheetbench_v1_tasks", counting_loader
    )
    metadata = {"task_id": "T1", "data_root": str(data_root)}
    task_1, root_1 = await ssb1_rollout._load_task(metadata)
    task_2, root_2 = await ssb1_rollout._load_task(metadata)

    assert len(calls) == 1
    assert task_1 is task_2
    assert task_1.uid == "SSB1__T1"
    assert root_1 == root_2 == data_root.resolve()


def _episode_sample(status=Sample.Status.COMPLETED, **metadata):
    return Sample(prompt="p", metadata=metadata, status=status)


def test_ssb1_metrics_aggregate_episodes():
    samples = [
        _episode_sample(
            benchmark="ssb1", run_id="a", exit_status="submitted",
            ssb1_metrics={"hard_restriction": 1.0, "soft_restriction": 1.0,
                          "cell_accuracy": 1.0},
        ),
        # Sibling fragment of episode "a": counted once.
        _episode_sample(
            benchmark="ssb1", run_id="a", exit_status="submitted",
            ssb1_metrics={"hard_restriction": 1.0, "soft_restriction": 1.0,
                          "cell_accuracy": 1.0},
        ),
        # Unsubmitted episode: scores 0 everywhere but stays in the means.
        _episode_sample(benchmark="ssb1", run_id="b", exit_status="limits_exceeded"),
        # Aborted and non-ssb1 samples are excluded.
        _episode_sample(Sample.Status.ABORTED, benchmark="ssb1", run_id="c"),
        _episode_sample(uid="UID0001"),
    ]
    metrics = ssb1_metrics._ssb1_metrics(samples)
    assert metrics == {
        "ssb1/episodes": 2.0,
        "ssb1/submitted_ratio": 0.5,
        "ssb1/hard_restriction": 0.5,
        "ssb1/soft_restriction": 0.5,
        "ssb1/cell_accuracy": 0.5,
    }


THINK_WRITE_TURN = (
    "<think>Plan the program.</think>\n\nWriting it now.\n"
    "```bash\ncat > solution.py <<'EOD'\nprint(1)\nEOD\n```"
)
THINK_SUBMIT_TURN = "<think>Looks right.</think>\n\n```bash\nsubmit\n```"


async def test_per_turn_aod2op_scores_commands_on_the_real_rollout(
    data_root, tmp_path, canned_scoring
):
    """The OfficeQA hill-climb engine, driven from the SSB1 rollout: the
    teacher scores reasoning-free requests per turn, the trained sample is the
    student's real rollout, and only fenced commands carry loss and scores."""
    from tests.test_vertex_teacher_client import teacher_responder
    from officeqa_rl.training.vertex_teacher_client import get_teacher_client

    async with (
        FakeSGLangServer(
            scripted_responder([THINK_WRITE_TURN, THINK_SUBMIT_TURN])
        ) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = make_args(
            actor,
            tmp_path,
            use_opd=True,
            opd_type="sglang",
            officeqa_opd_teacher_url=f"{teacher.base_url}/generate",
            officeqa_opd_backoff_sec=0.01,
            officeqa_opd_per_turn=True,
            officeqa_opd_teacher_no_think=True,
        )
        result = await ssb1_rollout.generate(
            args, make_base_sample(data_root), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result
    assert sample.status == Sample.Status.COMPLETED
    # Reward selection is untouched by the scoring shape.
    assert sample.reward == 0.5

    # Teacher side: each scored request ablates the current turn's reasoning.
    scored = [
        "".join(chr(t) for t in p["input_ids"]) for p in teacher.generate_payloads
    ]
    assert scored, "the teacher must be asked to score each fenced turn"
    for text in scored:
        assert text.rsplit("<think>", 1)[1].startswith("\n\n</think>")

    # Student side: the real rollout is trained, reasoning intact, loss and
    # teacher scores on the fenced commands only.
    response = sample.tokens[-sample.response_length :]
    text = "".join(chr(t) for t in response)
    assert "Plan the program." in text
    trained = "".join(
        chr(t) for t, m in zip(response, sample.loss_mask, strict=True) if m
    )
    assert "```bash" in trained and "<think>" not in trained
    assert "Writing it now." not in trained
    on_action = [
        lp
        for lp, m in zip(sample.teacher_log_probs, sample.loss_mask, strict=True)
        if m
    ]
    off_action = [
        lp
        for lp, m in zip(sample.teacher_log_probs, sample.loss_mask, strict=True)
        if not m
    ]
    assert any(lp != 0.0 for lp in on_action)
    assert all(lp == 0.0 for lp in off_action)


async def test_per_turn_mode_conflicts_abort_the_sample(
    data_root, tmp_path, canned_scoring
):
    from tests.test_vertex_teacher_client import teacher_responder
    from officeqa_rl.training.vertex_teacher_client import get_teacher_client

    async with (
        FakeSGLangServer(scripted_responder([THINK_SUBMIT_TURN])) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = make_args(
            actor,
            tmp_path,
            use_opd=True,
            opd_type="sglang",
            officeqa_opd_teacher_url=f"{teacher.base_url}/generate",
            officeqa_opd_backoff_sec=0.01,
            officeqa_opd_per_turn=True,
            officeqa_opd_action_only=True,
            officeqa_opd_mask_think=True,
        )
        sample = await ssb1_rollout.generate(
            args, make_base_sample(data_root), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    assert sample.status == Sample.Status.ABORTED
    assert "alternatives" in (sample.metadata or {})["abort_reason"]
    assert teacher.generate_payloads == []
