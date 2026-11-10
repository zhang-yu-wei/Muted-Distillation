"""Per-turn OPD (the hill-climb arms) driven from the JobBench rollout.

JobBench reaches the same engine as OfficeQA and SpreadsheetBench
(:mod:`officeqa_rl.training.opd_rollout`), so these cover the three things
that are specific to this harness rather than the engine:

* mini-swe-agent fences commands as ``mswea_bash_command``, not ``bash`` —
  ``_fence_span`` must still find the command, or every turn would look
  actionless and the arms would train nothing;
* the episode is judged, not replayed, so the judge-failure paths have to get
  the scorer right — released when the sample is dropped, and allowed to
  finish when the trajectory is kept for distillation;
* the per-sample teacher attachment must not also run when the per-turn
  scorer did, or a turn's log-probs would be written twice.
"""

import pytest
from slime.utils.types import Sample

from officeqa_rl.jobbench.environment import SUBMIT_SENTINEL
from officeqa_rl.training import jobbench_rollout
from tests.sglang_fakes import FakeSGLangServer
from tests.test_jobbench_rollout import (  # noqa: F401  (fixtures)
    canned_judge,
    data_root,
    fake_tokenizer,
    judge_key,
    make_args,
    make_base_sample,
    scripted_responder,
)

# The chat template opens ``<think>`` in the prompt, so a real turn closes it
# before the visible region. The fakes echo raw text, so the block is written
# out here.
THINK_WRITE_TURN = (
    "<think>Plan the memo.</think>\n\nWriting the memo.\n"
    '```mswea_bash_command\nprintf "20.2%%\\n" > "$JB_OUTPUT_DIR/memo.md"\n```'
)
THINK_SUBMIT_TURN = (
    f"<think>Looks right.</think>\n\nDone.\n"
    f"```mswea_bash_command\necho {SUBMIT_SENTINEL}\n```"
)


def test_fence_span_accepts_the_mini_swe_agent_tag():
    """The engine's fence parser must accept ``mswea_bash_command``.

    Guards the assumption the whole port rests on: the language class in
    ``_FENCED_COMMAND_RE`` is ``[A-Za-z0-9_-]*``, so the tag matches. If this
    ever tightened to ``bash``, every JobBench turn would read as actionless —
    silently, since a turn with no fence is dropped rather than raised.
    """
    from tests.sglang_fakes import ByteTokenizer
    from officeqa_rl.training.opd_rollout import _fence_span

    tokenizer = ByteTokenizer()
    ids = [ord(c) for c in THINK_SUBMIT_TURN]
    assert _fence_span(tokenizer, ids) is not None


async def test_per_turn_aod2op_scores_commands_on_the_real_rollout(
    data_root, tmp_path, canned_judge
):
    """Teacher scores reasoning-free requests per turn; the trained sample is
    the student's real rollout, with loss and scores on the fenced command."""
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
        result = await jobbench_rollout.generate(
            args, make_base_sample(data_root), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result
    assert sample.status == Sample.Status.COMPLETED
    # The judge still sets the reward; the scoring shape does not touch it.
    assert sample.reward == pytest.approx(10 / 16)
    assert sample.metadata["opd_path"]["scorer"] is True
    assert sample.metadata["opd_path"]["per_turn"] is True

    # Teacher side: each scored request ablates that turn's reasoning.
    scored = [
        "".join(chr(t) for t in p["input_ids"]) for p in teacher.generate_payloads
    ]
    assert scored, "the teacher must be asked to score each fenced turn"
    for text in scored:
        assert text.rsplit("<think>", 1)[1].startswith("\n\n</think>")

    # Student side: real rollout trained, reasoning intact, loss on commands.
    response = sample.tokens[-sample.response_length :]
    text = "".join(chr(t) for t in response)
    assert "Plan the memo." in text
    trained = "".join(
        chr(t) for t, m in zip(response, sample.loss_mask, strict=True) if m
    )
    assert "mswea_bash_command" in trained
    assert "<think>" not in trained
    assert "Writing the memo." not in trained
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


async def test_per_turn_mode_conflicts_abort_before_the_teacher_is_called(
    data_root, tmp_path, canned_judge
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
        sample = await jobbench_rollout.generate(
            args, make_base_sample(data_root), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    assert sample.status == Sample.Status.ABORTED
    assert "alternatives" in (sample.metadata or {})["abort_reason"]
    assert teacher.generate_payloads == []


async def test_eval_rollouts_skip_the_per_turn_scorer(
    data_root, tmp_path, canned_judge
):
    """Eval must not spend teacher calls, exactly as on the other two hooks."""
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
        result = await jobbench_rollout.generate(
            args,
            make_base_sample(data_root),
            {"max_new_tokens": 512},
            evaluation=True,
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result
    assert sample.metadata["opd_path"]["scorer"] is False
    assert sample.metadata["opd_path"]["evaluation"] is True
    assert teacher.generate_payloads == []


async def test_judge_failure_keeps_the_trajectory_for_distillation(
    data_root, tmp_path, monkeypatch
):
    """A grading outage must not discard a usable distillation trajectory.

    The teacher's per-token log-probs are independent of the LLM judge, so an
    episode the judge could not grade still carries a full OPD signal. Aborting
    it would throw away every turn of teacher supervision over an outage on a
    different service. The sample is kept and trained; only its task reward is
    withheld, via ``reward_available: False`` (see
    ``slime/ray/rollout.py::_normalize_rewards_per_rollout``).
    """
    from tests.test_vertex_teacher_client import teacher_responder
    from officeqa_rl.jobbench.judge import RubricResult, build_score
    from officeqa_rl.training.vertex_teacher_client import get_teacher_client

    async def failing_judge(task, output_dir, config, *, client=None):
        return build_score(
            task.uid,
            [RubricResult(0, "Reports the rate", 10, False, 0, 1, 0, error="judge_error: boom")],
            judge_model="canned",
            n_output_files=1,
        )

    monkeypatch.setattr(jobbench_rollout, "judge_episode", failing_judge)

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
        result = await jobbench_rollout.generate(
            args, make_base_sample(data_root), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    samples = result if isinstance(result, list) else [result]
    sample = samples[0]

    # Kept, not aborted: the loss mask is live and the KL term will train it.
    assert sample.status is not Sample.Status.ABORTED
    assert sample.remove_sample is False
    assert any(sample.loss_mask)

    # The teacher was allowed to finish rather than being cancelled.
    assert teacher.generate_payloads, "teacher scoring must not be cancelled"
    assert any(lp != 0.0 for lp in sample.teacher_log_probs)

    # The reward is a placeholder and is flagged as unmeasured, so reward
    # centering excludes it and it never becomes an advantage.
    assert sample.reward == 0.0
    assert sample.metadata["reward_available"] is False
    assert "judge failed on" in sample.metadata["grading_failure"]


async def test_judge_failure_still_aborts_without_a_teacher(
    data_root, tmp_path, monkeypatch
):
    """Without OPD the task reward is the entire signal, so nothing survives a
    grading outage and the sample must still be dropped rather than trained on
    a zero that reads as a policy failure."""
    from officeqa_rl.jobbench.judge import RubricResult, build_score

    async def failing_judge(task, output_dir, config, *, client=None):
        return build_score(
            task.uid,
            [RubricResult(0, "Reports the rate", 10, False, 0, 1, 0, error="judge_error: boom")],
            judge_model="canned",
            n_output_files=1,
        )

    monkeypatch.setattr(jobbench_rollout, "judge_episode", failing_judge)
    async with FakeSGLangServer(
        scripted_responder([THINK_WRITE_TURN, THINK_SUBMIT_TURN])
    ) as actor:
        args = make_args(actor, tmp_path)
        sample = await jobbench_rollout.generate(
            args, make_base_sample(data_root), {"max_new_tokens": 512}
        )

    assert sample.status == Sample.Status.ABORTED
    assert sample.remove_sample is True
