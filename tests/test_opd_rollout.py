import asyncio
"""Tests for the per-turn OPD teacher-scoring rollout hook.

Reuses the actor/teacher fakes and fixtures from ``test_slime_rollout`` and
``test_vertex_teacher_client``: the actor fake serves scripted turns, the
teacher fake scores -0.25 * global position per token, both over plain http.
"""

import json

import pytest
from aiohttp import web

pytest.importorskip("slime", reason="requires the vendored slime on PYTHONPATH")

from slime.utils.types import Sample

from officeqa_rl.training import opd_rollout
from officeqa_rl.training.opd_rollout import (
    _EMPTY_THINK,
    _GEMMA_DIALECT,
    _QWEN_DIALECT,
    _action_only_plan,
    _fill_teacher_log_probs,
    _tail_span,
    _think_dialect,
)
from tests.sglang_fakes import ByteTokenizer
from officeqa_rl.training.vertex_teacher_client import (
    TeacherScoringError,
    get_teacher_client,
)
from tests.sglang_fakes import FakeSGLangServer, sglang_response
from tests.test_slime_rollout import (  # noqa: F401  (fixtures register on import)
    SEARCH_TURN,
    SUBMIT_TURN,
    corpus_dir,
    fake_tokenizer,
    make_args,
    make_base_sample,
    scripted_responder,
)
from tests.test_vertex_teacher_client import opd_args, teacher_responder, teacher_scores


async def test_per_turn_scoring_matches_sample_layout(corpus_dir, tmp_path):
    async with (
        FakeSGLangServer(scripted_responder([SEARCH_TURN, SUBMIT_TURN])) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(actor, teacher, corpus_dir, tmp_path)
        sample = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    assert isinstance(sample, Sample)
    assert sample.status == Sample.Status.COMPLETED
    assert sample.reward == 1.0

    # One teacher request per agent turn, each scoring exactly that turn's
    # output under its exact served prompt.
    assert len(teacher.generate_payloads) == 2
    for actor_payload, teacher_payload in zip(
        actor.generate_payloads, teacher.generate_payloads, strict=True
    ):
        prompt_ids = actor_payload["input_ids"]
        assert teacher_payload["input_ids"][: len(prompt_ids)] == prompt_ids
        assert teacher_payload["logprob_start_len"] == len(prompt_ids) - 1

    # The assembled teacher_log_probs cover the response region: turn spans
    # carry the teacher's scores (-0.25 * global position within that turn's
    # request), everything else (observations) is 0.0.
    response_length = sample.response_length
    assert len(sample.teacher_log_probs) == response_length
    expected = [0.0] * response_length
    position = 0
    for teacher_payload in teacher.generate_payloads:
        start = teacher_payload["logprob_start_len"]
        output_length = len(teacher_payload["input_ids"]) - start - 1
        # Find this turn's trained span: the next run of loss_mask == 1.
        while sample.loss_mask[position] == 0:
            position += 1
        expected[position : position + output_length] = [
            -0.25 * p for p in range(start + 1, start + 1 + output_length)
        ]
        position += output_length
    assert sample.teacher_log_probs == pytest.approx(expected)
    # Observation positions really are zero (and masked).
    assert any(mask == 0 for mask in sample.loss_mask)
    for mask, log_prob in zip(sample.loss_mask, sample.teacher_log_probs, strict=True):
        if mask == 0:
            assert log_prob == 0.0


async def test_delegates_to_default_path_when_opd_off_or_eval(corpus_dir, tmp_path):
    async with (
        FakeSGLangServer(scripted_responder([SUBMIT_TURN, SUBMIT_TURN])) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        # OPD flags absent: behaves exactly like the default hook.
        args = make_args(actor, corpus_dir, tmp_path)
        sample = await opd_rollout.generate(args, make_base_sample(), {"max_new_tokens": 512})
        assert sample.teacher_log_probs is None

        # OPD on but evaluation: no teacher traffic.
        args = opd_args(actor, teacher, corpus_dir, tmp_path)
        sample = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}, evaluation=True
        )
        assert sample.teacher_log_probs is None

    assert teacher.generate_payloads == []


async def test_teacher_failure_is_fatal_and_logged(corpus_dir, tmp_path):
    async def broken(payload):
        return web.Response(status=500, text="teacher down")

    failure_log = tmp_path / "failures" / "opd.jsonl"
    async with (
        FakeSGLangServer(scripted_responder([SUBMIT_TURN])) as actor,
        FakeSGLangServer(broken) as teacher,
    ):
        args = opd_args(
            actor,
            teacher,
            corpus_dir,
            tmp_path,
            officeqa_opd_attempts=2,
            officeqa_opd_failure_log=str(failure_log),
        )
        with pytest.raises(TeacherScoringError, match="after 2 attempts"):
            await opd_rollout.generate(args, make_base_sample(), {"max_new_tokens": 512})
        await get_teacher_client(args).close()

    records = [json.loads(line) for line in failure_log.read_text().splitlines()]
    assert len(records) == 1
    assert "HTTP 500" in records[0]["error"]


def _sample(tokens, loss_mask, response_length):
    return Sample(
        prompt="p",
        tokens=tokens,
        loss_mask=loss_mask,
        response_length=response_length,
    )


def test_fill_maps_turns_in_order_and_zeroes_observations():
    # prompt [1,2]; turn A output [10,11]; obs [20,21,22]; turn B output [12].
    sample = _sample(
        tokens=[1, 2, 10, 11, 20, 21, 22, 12],
        loss_mask=[1, 1, 0, 0, 0, 1],
        response_length=6,
    )
    _fill_teacher_log_probs(
        [sample],
        [([10, 11], [-1.0, -2.0], None), ([12], [-3.0], None)],
    )
    assert sample.teacher_log_probs == [-1.0, -2.0, 0.0, 0.0, 0.0, -3.0]


def test_fill_skips_realigned_turns():
    # Turn [10, 11] was REALIGNed away: it never appears as a trained span.
    sample = _sample(
        tokens=[1, 2, 12, 13],
        loss_mask=[1, 1],
        response_length=2,
    )
    _fill_teacher_log_probs(
        [sample],
        [([10, 11], [-1.0, -2.0], None), ([12, 13], [-3.0, -4.0], None)],
    )
    assert sample.teacher_log_probs == [-3.0, -4.0]


def test_fill_handles_truncated_tail_and_multiple_samples():
    # Sample 1 trains turn A; sample 2 trains turn B but is truncated
    # mid-span by max_sample_tokens.
    sample_1 = _sample(tokens=[1, 10, 11], loss_mask=[1, 1], response_length=2)
    sample_2 = _sample(tokens=[1, 2, 12, 13], loss_mask=[1, 1], response_length=2)
    _fill_teacher_log_probs(
        [sample_1, sample_2],
        [([10, 11], [-1.0, -2.0], None), ([12, 13, 14], [-3.0, -4.0, -5.0], None)],
    )
    assert sample_1.teacher_log_probs == [-1.0, -2.0]
    assert sample_2.teacher_log_probs == [-3.0, -4.0]


def test_fill_raises_on_misalignment():
    sample = _sample(tokens=[1, 10, 11], loss_mask=[1, 1], response_length=2)
    with pytest.raises(ValueError, match="cannot align"):
        _fill_teacher_log_probs([sample], [([99, 98], [-1.0, -2.0], None)])


# ------------------------------------------- action-only distillation (AOD)

THINK_SEARCH_TURN = (
    "<think>Peek at the file.</think>\n\nChecking.\n"
    "```bash\ncat officeqa_corpus/example.txt\n```"
)
THINK_SUBMIT_TURN = '<think>Answer is 42.</think>\n\n```bash\nsubmit "42"\n```'
NO_FENCE_TURN = "<think>Hmm.</think>\n\nI am lost."
# Opens deliberation and stops inside it: no </think>, so no visible region
# at all. The command-only request drops such a turn entirely.
UNCLOSED_TURN = "<think>Let me reconsider the layout once more"


def _ids(text):
    return [ord(ch) for ch in text]


PROMPT_NO_THINK = _ids("<assistant>")
# Qwen3.5 thinking mode: the generation prompt opens the block itself, so the
# turn output carries only the closing tag.
PROMPT_OPEN_THINK = _ids("<assistant><think>\n")


def test_score_plan_splits_think_and_fence():
    text = THINK_SEARCH_TURN
    plan = _action_only_plan(ByteTokenizer(), PROMPT_NO_THINK, _ids(text))
    visible_from = text.index("</think>") + len("</think>")
    fence_lo = text.index("```bash")
    assert plan.tail_start == visible_from
    assert plan.prefix_ids == tuple(_ids(_EMPTY_THINK))
    assert plan.spans == ((fence_lo, len(text)),)


def test_score_plan_completes_a_prompt_opened_think_block():
    # The prompt already wrote "<think>\n"; inserting the whole empty block
    # would nest a second opener, so only the remainder belongs in the turn.
    text = '\n\n```bash\nsubmit "42"\n```'
    plan = _action_only_plan(ByteTokenizer(), PROMPT_OPEN_THINK, _ids(text))
    assert plan.prefix_ids == tuple(_ids("\n</think>"))
    rendered = "<think>\n" + "\n</think>" + text
    assert rendered.startswith(_EMPTY_THINK)
    assert rendered.count("<think>") == 1


def test_score_plan_adds_nothing_when_prompt_closed_the_block():
    text = '```bash\nsubmit "42"\n```'
    prompt = _ids("<assistant>" + _EMPTY_THINK + "\n\n")
    plan = _action_only_plan(ByteTokenizer(), prompt, _ids(text))
    assert plan.prefix_ids == ()


def test_score_plan_without_think_scores_from_turn_start():
    text = '```bash\nsubmit "42"\n```'
    plan = _action_only_plan(ByteTokenizer(), PROMPT_NO_THINK, _ids(text))
    assert plan.tail_start == 0
    assert plan.spans == ((0, len(text)),)


@pytest.mark.parametrize(
    "text",
    [
        NO_FENCE_TURN,  # no fenced command
        "<think>ok</think>\n```bash\na\n```\n```bash\nb\n```",  # two fences
        "<think>\n```bash\nls\n```\n</think>\nno action",  # fence only in think
    ],
)
def test_score_plan_discards_unusable_turns(text):
    plan = _action_only_plan(ByteTokenizer(), PROMPT_NO_THINK, _ids(text))
    assert plan.tail_start is None


async def test_action_only_builds_reasoning_free_turn_samples(corpus_dir, tmp_path):
    turns = [THINK_SEARCH_TURN, NO_FENCE_TURN, THINK_SUBMIT_TURN]
    async with (
        FakeSGLangServer(scripted_responder(turns)) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor, teacher, corpus_dir, tmp_path, officeqa_opd_action_only=True
        )
        samples = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    # The fence-less turn contributes nothing: two samples, two requests.
    assert isinstance(samples, list) and len(samples) == 2
    assert len(teacher.generate_payloads) == 2

    empty_think = _ids(_EMPTY_THINK)
    for sample, teacher_payload, actor_index, text in (
        (samples[0], teacher.generate_payloads[0], 0, THINK_SEARCH_TURN),
        (samples[1], teacher.generate_payloads[1], 2, THINK_SUBMIT_TURN),
    ):
        assert sample.reward == 1.0
        prompt_ids = actor.generate_payloads[actor_index]["input_ids"]
        visible_from = text.index("</think>") + len("</think>")
        tail = text[visible_from:]

        # The student trains on the SAME reasoning-free sequence the teacher
        # scored: context + empty think as prompt, the visible tail as the
        # response region.
        assert sample.tokens == prompt_ids + empty_think + _ids(tail)
        assert sample.response_length == len(tail)
        assert teacher_payload["input_ids"] == sample.tokens
        assert (
            teacher_payload["logprob_start_len"]
            == len(prompt_ids) + len(empty_think) - 1
        )

        # Loss mask keeps exactly the fenced command block.
        fence_lo = tail.index("```")
        expected_mask = [1 if fence_lo <= i < len(tail) else 0 for i in range(len(tail))]
        assert sample.loss_mask == expected_mask

        # Teacher log-probs cover the whole response region at the request's
        # score positions (-0.25 * global request position).
        start = teacher_payload["logprob_start_len"]
        expected_scores = [-0.25 * p for p in range(start + 1, start + 1 + len(tail))]
        assert sample.teacher_log_probs == pytest.approx(expected_scores)

        # The sample no longer holds the turn's reasoning, so metrics read
        # the full generated length from metadata instead.
        assert sample.metadata["episode_output_tokens"] == sum(len(t) for t in turns)


async def test_mask_think_keeps_full_turn_but_trains_only_the_command(
    corpus_dir, tmp_path
):
    turns = [THINK_SEARCH_TURN, NO_FENCE_TURN, THINK_SUBMIT_TURN]
    async with (
        FakeSGLangServer(scripted_responder(turns)) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor, teacher, corpus_dir, tmp_path, officeqa_opd_mask_think=True
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result

    # The teacher saw every turn in full, reasoning included — one request per
    # turn over the served prompt, exactly as without the flag.
    assert len(teacher.generate_payloads) == 3
    for actor_payload, teacher_payload in zip(
        actor.generate_payloads, teacher.generate_payloads, strict=True
    ):
        prompt_ids = actor_payload["input_ids"]
        assert teacher_payload["input_ids"][: len(prompt_ids)] == prompt_ids
        assert teacher_payload["logprob_start_len"] == len(prompt_ids) - 1

    # The sample still carries the reasoning tokens, and teacher log-probs
    # still cover them; only the loss mask narrowed to the fenced commands.
    response = sample.tokens[-sample.response_length :]
    text = "".join(chr(t) for t in response)
    assert "<think>" in text
    trained = "".join(
        chr(t) for t, m in zip(response, sample.loss_mask, strict=True) if m
    )
    assert trained.count("```bash") == 2 and "<think>" not in trained
    assert "I am lost" not in trained  # fence-less turn keeps no loss
    scored_think = [
        lp
        for lp, m, t in zip(sample.teacher_log_probs, sample.loss_mask, response, strict=True)
        if not m and chr(t) != "\n"
    ]
    assert any(lp != 0.0 for lp in scored_think)


async def test_mask_think_tail_trains_the_whole_visible_tail(corpus_dir, tmp_path):
    """-maskthink-tail: the full-turn request of mask-think, the visible-tail
    loss of the -tail variants."""
    turns = [THINK_SEARCH_TURN, NO_FENCE_TURN, THINK_SUBMIT_TURN]
    async with (
        FakeSGLangServer(scripted_responder(turns)) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor, teacher, corpus_dir, tmp_path,
            officeqa_opd_mask_think=True, officeqa_opd_tail_loss=True,
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result
    # Same full-turn teacher request as plain mask-think.
    assert len(teacher.generate_payloads) == 3
    for actor_payload, teacher_payload in zip(
        actor.generate_payloads, teacher.generate_payloads, strict=True
    ):
        prompt_ids = actor_payload["input_ids"]
        assert teacher_payload["input_ids"][: len(prompt_ids)] == prompt_ids

    response = sample.tokens[-sample.response_length :]
    trained = "".join(
        chr(t) for t, m in zip(response, sample.loss_mask, strict=True) if m
    )
    assert "<think>" not in trained and "</think>" not in trained
    assert trained.count("```bash") == 2
    # The fence-less turn's visible text now carries loss too (tail, not command).
    assert "I am lost" in trained
    # Every trained turn's loss starts right after its think block: nothing
    # between </think> and the fence is left out.
    text = "".join(chr(t) for t in response)
    for visible in (t.split("</think>", 1)[1] for t in turns if "</think>" in t):
        assert visible.strip() in trained


async def test_train_turns_caps_training_but_not_evaluation(corpus_dir, tmp_path):
    """The horizon curriculum must cut the training episode without changing
    what evaluation measures."""
    turns = [SEARCH_TURN, SEARCH_TURN, SUBMIT_TURN]
    async with (
        FakeSGLangServer(scripted_responder(list(turns))) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor, teacher, corpus_dir, tmp_path, officeqa_opd_train_turns=2
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result
    # Two turns generated and scored, then the episode stopped: the third
    # scripted turn (the submission) was never reached.
    assert len(actor.generate_payloads) == 2
    assert len(teacher.generate_payloads) == 2
    assert sample.metadata["steps"] == 2
    assert sample.metadata["exit_status"] == "limits_exceeded"
    assert sample.status == Sample.Status.TRUNCATED
    assert sample.reward == 0.0

    # Evaluation ignores the cap and runs the real task to submission.
    async with (
        FakeSGLangServer(scripted_responder(list(turns))) as eval_actor,
        FakeSGLangServer(teacher_responder) as eval_teacher,
    ):
        args = opd_args(
            eval_actor, eval_teacher, corpus_dir, tmp_path,
            officeqa_opd_train_turns=2,
        )
        eval_sample = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}, evaluation=True
        )

    assert len(eval_actor.generate_payloads) == 3
    assert eval_sample.reward == 1.0
    assert eval_teacher.generate_payloads == []


async def test_aborted_sample_carries_tensorizable_fields(corpus_dir, tmp_path):
    """slime keeps removed samples in the batch and tensorizes every per-token
    field, so a None among them fails the whole step (observed: aod2 died at
    step 8 on one aborted episode)."""
    async with (
        FakeSGLangServer(scripted_responder([NO_FENCE_TURN, NO_FENCE_TURN])) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor,
            teacher,
            corpus_dir,
            tmp_path,
            officeqa_opd_action_only=True,
            officeqa_step_limit=2,
        )
        sample = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    assert sample.status == Sample.Status.ABORTED and sample.remove_sample
    for field in ("loss_mask", "rollout_log_probs", "teacher_log_probs"):
        values = getattr(sample, field)
        assert values is not None, f"{field} is None — tensorization would fail"
        assert len(values) == sample.response_length
        assert all(v is not None for v in values)
    # slime left-pads the loss mask by prompt_length - 1, so a tokenless abort
    # kills the step with "narrow(): length must be non-negative" (observed:
    # mask-think died at step 92 on one no_action_turns abort).
    prompt_length = len(sample.tokens or []) - sample.response_length
    assert prompt_length >= 1, "aborted sample must keep at least one prompt token"


async def test_action_only_aborts_without_any_fenced_turn(corpus_dir, tmp_path):
    async with (
        FakeSGLangServer(scripted_responder([NO_FENCE_TURN, NO_FENCE_TURN])) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor,
            teacher,
            corpus_dir,
            tmp_path,
            officeqa_opd_action_only=True,
            officeqa_step_limit=2,
        )
        sample = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    assert isinstance(sample, Sample)
    assert sample.status == Sample.Status.ABORTED
    assert teacher.generate_payloads == []


async def test_teacher_no_think_scores_without_reasoning_but_trains_the_rollout(
    corpus_dir, tmp_path
):
    """The deliberately mismatched pairing: the teacher's request drops the
    turn's reasoning (as under AOD), while the trained sequence is the one the
    student actually generated, reasoning ahead of the action."""
    turns = [THINK_SEARCH_TURN, NO_FENCE_TURN, THINK_SUBMIT_TURN]
    async with (
        FakeSGLangServer(scripted_responder(turns)) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor, teacher, corpus_dir, tmp_path, officeqa_opd_teacher_no_think=True
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result

    # Teacher side: the scored sequence carries the empty think block, not the
    # student's reasoning — the same request AOD would have sent.
    scored = ["".join(chr(t) for t in p["input_ids"]) for p in teacher.generate_payloads]
    assert scored, "the teacher must still be asked to score"
    for text in scored:
        # Only the CURRENT turn's reasoning is ablated; earlier turns stay in
        # the history, so check the tail the teacher is asked to score.
        assert text.rsplit("<think>", 1)[1].startswith("\n\n</think>")
    assert "Peek at the file." not in scored[0], "turn 1 has no history to hide in"

    # Student side: the trained sample is the real rollout, reasoning intact.
    response = sample.tokens[-sample.response_length :]
    text = "".join(chr(t) for t in response)
    assert "Peek at the file." in text, "the student's own reasoning must survive"

    # Loss stays on the fenced commands only.
    trained = "".join(
        chr(t) for t, m in zip(response, sample.loss_mask, strict=True) if m
    )
    assert trained.count("```bash") == 2 and "<think>" not in trained
    assert "I am lost" not in trained
    # Visible prose of a fence-bearing turn is teacher context, never trained:
    # the think-to-prose migration this arm shows on SSB1 is not the loss
    # reaching the prose.
    assert "Checking." in text and "Checking." not in trained

    # Teacher log-probs land on the trained positions and nowhere else.
    on_action = [
        lp for lp, m in zip(sample.teacher_log_probs, sample.loss_mask, strict=True) if m
    ]
    off_action = [
        lp
        for lp, m in zip(sample.teacher_log_probs, sample.loss_mask, strict=True)
        if not m
    ]
    assert any(lp != 0.0 for lp in on_action)
    assert all(lp == 0.0 for lp in off_action)


async def test_teacher_no_think_rejects_combining_with_the_other_modes(
    corpus_dir, tmp_path
):
    async with (
        FakeSGLangServer(scripted_responder([THINK_SUBMIT_TURN])) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor,
            teacher,
            corpus_dir,
            tmp_path,
            officeqa_opd_teacher_no_think=True,
            officeqa_opd_action_only=True,
        )
        sample = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    assert sample.status == Sample.Status.ABORTED
    assert "alternatives" in (sample.metadata or {})["abort_reason"]
    assert teacher.generate_payloads == []


async def test_tail_loss_aod_trains_the_whole_visible_tail(corpus_dir, tmp_path):
    """AOD + tail loss: prose joins the trained span, and a fence-less turn
    contributes a sample instead of being dropped."""
    turns = [THINK_SEARCH_TURN, NO_FENCE_TURN, THINK_SUBMIT_TURN]
    async with (
        FakeSGLangServer(scripted_responder(turns)) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor,
            teacher,
            corpus_dir,
            tmp_path,
            officeqa_opd_action_only=True,
            officeqa_opd_tail_loss=True,
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    samples = result if isinstance(result, list) else [result]
    # The fence-less turn is now scorable, so every turn gets a teacher
    # request and its own single-turn sample.
    assert len(teacher.generate_payloads) == 3
    assert len(samples) == 3

    trained = ""
    for sample in samples:
        response = sample.tokens[-sample.response_length :]
        trained += "".join(
            chr(t) for t, m in zip(response, sample.loss_mask, strict=True) if m
        )
    assert "Checking." in trained, "prose must carry loss"
    assert "I am lost." in trained, "the fence-less turn must carry loss"
    assert trained.count("```bash") == 2
    assert "<think>" not in trained and "Peek at the file." not in trained


async def test_tail_loss_teacher_no_think_scores_and_trains_prose(
    corpus_dir, tmp_path
):
    """aod2op + tail loss: real rollout trained, mask covers the whole
    visible tail, thinking stays maskless and unscored."""
    turns = [THINK_SEARCH_TURN, NO_FENCE_TURN, THINK_SUBMIT_TURN]
    async with (
        FakeSGLangServer(scripted_responder(turns)) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor,
            teacher,
            corpus_dir,
            tmp_path,
            officeqa_opd_teacher_no_think=True,
            officeqa_opd_tail_loss=True,
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result
    response = sample.tokens[-sample.response_length :]
    text = "".join(chr(t) for t in response)
    assert "Peek at the file." in text, "the real rollout keeps its reasoning"

    trained = "".join(
        chr(t) for t, m in zip(response, sample.loss_mask, strict=True) if m
    )
    assert "Checking." in trained and "I am lost." in trained
    assert trained.count("```bash") == 2
    assert "<think>" not in trained
    # Teacher scores cover exactly the masked (visible-tail) positions.
    on = [lp for lp, m in zip(sample.teacher_log_probs, sample.loss_mask, strict=True) if m]
    off = [lp for lp, m in zip(sample.teacher_log_probs, sample.loss_mask, strict=True) if not m]
    assert any(lp != 0.0 for lp in on)
    assert all(lp == 0.0 for lp in off)


def _marked_runs(sample) -> list[float]:
    """Sum of the marker values in each contiguous penalised span.

    Marked tokens are contiguous within a turn and turns are separated by
    unmarked observation tokens, so a run is one penalised turn.
    """
    runs, current = [], 0.0
    for lp, m in zip(sample.teacher_log_probs, sample.loss_mask, strict=True):
        if m and lp > 0:
            current += lp
        elif current:
            runs.append(current)
            current = 0.0
    if current:
        runs.append(current)
    return runs


async def test_noaction_penalty_window_is_length_independent(
    corpus_dir, tmp_path
):
    """A short actionless turn pays exactly what a long one pays.

    Both earlier builds coupled the charge to the turn's length and the policy
    escaped along that axis: a constant per-token value made the total grow
    with length, so it truncated (1,947 -> 514 tokens); spreading a constant
    total made the per-token value fall as 1/length, so it inflated
    (498 -> 3,396 tokens). A fixed window at a fixed value is charged the same
    either way. The two turns here differ several-fold in length and must pay
    identically, in both token count and total.
    """
    long_turn = "<think>" + "Let me reconsider the layout once more. " * 12
    short_turn = "<think>No.</think>\n\nstuck"
    turns = [long_turn, short_turn, THINK_SUBMIT_TURN]
    window = 16
    async with (
        FakeSGLangServer(scripted_responder(turns)) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor, teacher, corpus_dir, tmp_path,
            officeqa_opd_teacher_no_think=True, officeqa_opd_tail_loss=True,
            officeqa_opd_pure=True, officeqa_opd_noaction_penalty=0.25,
            officeqa_opd_noaction_window=window,
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result
    spans = list(_marked_span_lengths(sample))
    assert len(spans) == 2, f"both actionless turns penalised, got {len(spans)}"
    # Same token count and same total, though the turns differ in length.
    for span in spans:
        assert len(span) == window
        assert all(value == pytest.approx(0.25) for value in span)
        assert sum(span) == pytest.approx(window * 0.25)


def _marked_span_lengths(sample):
    """Each contiguous penalised span, as a list of its marker values."""
    current: list[float] = []
    for lp, m in zip(sample.teacher_log_probs, sample.loss_mask, strict=True):
        if m and lp > 0:
            current.append(lp)
        elif current:
            yield current
            current = []
    if current:
        yield current


async def test_noaction_penalty_covers_both_kinds_of_actionless_turn(
    corpus_dir, tmp_path
):
    """Every turn that issued no command is penalised over its whole length.

    Two kinds qualify and both must be caught: NO_FENCE_TURN closes its think
    block but writes no command (otherwise trained against the teacher), and
    UNCLOSED_TURN never closes it (otherwise dropped from the objective).
    """
    turns = [UNCLOSED_TURN, NO_FENCE_TURN, THINK_SUBMIT_TURN]
    async with (
        FakeSGLangServer(scripted_responder(turns)) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor,
            teacher,
            corpus_dir,
            tmp_path,
            officeqa_opd_teacher_no_think=True,
            officeqa_opd_tail_loss=True,
            officeqa_opd_pure=True,
            officeqa_opd_noaction_penalty=0.25,
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result
    response = sample.tokens[-sample.response_length :]
    marked = "".join(
        chr(t)
        for t, lp, m in zip(response, sample.teacher_log_probs, sample.loss_mask,
                            strict=True)
        if m and lp > 0
    )
    # Both actionless turns are penalised, in full.
    assert "reconsider the layout" in marked
    assert "I am lost." in marked
    # The acting turn keeps real teacher scores and is never marked.
    assert "submit" not in marked
    scored = [
        lp
        for lp, m in zip(sample.teacher_log_probs, sample.loss_mask, strict=True)
        if m and lp <= 0
    ]
    assert scored, "the turn that acted still trains on teacher scores"
    # With no window the whole turn is marked, every token at full magnitude.
    assert all(lp == pytest.approx(0.25) for lp in sample.teacher_log_probs if lp > 0)


async def test_noaction_metric_is_written_even_with_the_penalty_off(
    corpus_dir, tmp_path
):
    """The metric records what the function received, so a missing key means
    it never ran and penalty=0.0 means the value did not arrive."""
    turns = [UNCLOSED_TURN, NO_FENCE_TURN, THINK_SUBMIT_TURN]
    async with (
        FakeSGLangServer(scripted_responder(turns)) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor, teacher, corpus_dir, tmp_path,
            officeqa_opd_teacher_no_think=True, officeqa_opd_tail_loss=True,
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result
    metric = (sample.metadata or {})["noaction"]
    assert metric["penalty"] == 0.0
    assert metric["penalised"] == 0
    assert metric["penalised_tokens"] == 0
    assert metric["window"] == 0
    assert metric["walked"] >= 1, "the walk still reports what it saw"


async def test_noaction_penalty_off_leaves_scoring_untouched(corpus_dir, tmp_path):
    """Default-off must reproduce the previous behaviour exactly."""
    turns = [UNCLOSED_TURN, NO_FENCE_TURN, THINK_SUBMIT_TURN]
    async with (
        FakeSGLangServer(scripted_responder(turns)) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor, teacher, corpus_dir, tmp_path,
            officeqa_opd_teacher_no_think=True, officeqa_opd_tail_loss=True,
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result
    assert all(lp <= 0.0 for lp in sample.teacher_log_probs)
    response = sample.tokens[-sample.response_length :]
    trained = "".join(
        chr(t) for t, m in zip(response, sample.loss_mask, strict=True) if m
    )
    assert "reconsider the layout" not in trained, "unclosed turn stays dropped"
    assert "I am lost." in trained, "fence-less prose still trains on scores"


async def test_noaction_penalty_requires_the_on_policy_base_and_pure(
    corpus_dir, tmp_path
):
    for extra, expected in (
        ({"officeqa_opd_pure": True}, "officeqa_opd_teacher_no_think"),
        ({"officeqa_opd_teacher_no_think": True}, "officeqa_opd_pure"),
    ):
        async with (
            FakeSGLangServer(scripted_responder([THINK_SUBMIT_TURN])) as actor,
            FakeSGLangServer(teacher_responder) as teacher,
        ):
            args = opd_args(
                actor, teacher, corpus_dir, tmp_path,
                officeqa_opd_noaction_penalty=0.25, **extra,
            )
            sample = await opd_rollout.generate(
                args, make_base_sample(), {"max_new_tokens": 512}
            )
            await get_teacher_client(args).close()
        assert sample.status == Sample.Status.ABORTED
        assert expected in (sample.metadata or {})["abort_reason"]


async def test_tail_loss_requires_a_command_only_mode(corpus_dir, tmp_path):
    async with (
        FakeSGLangServer(scripted_responder([THINK_SUBMIT_TURN])) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor, teacher, corpus_dir, tmp_path, officeqa_opd_tail_loss=True
        )
        sample = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    assert sample.status == Sample.Status.ABORTED
    assert "officeqa_opd_mask_think" in (sample.metadata or {})["abort_reason"]
    assert teacher.generate_payloads == []


async def test_prose_as_think_ablates_prose_from_teacher_and_sample(
    corpus_dir, tmp_path
):
    """AOD + prose-as-think: the kept tail is exactly the fenced command, so
    prose vanishes from both the teacher request and the trained sequence,
    with the canonical newlines bridging </think> to the fence."""
    turns = [THINK_SEARCH_TURN, NO_FENCE_TURN, THINK_SUBMIT_TURN]
    async with (
        FakeSGLangServer(scripted_responder(turns)) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor,
            teacher,
            corpus_dir,
            tmp_path,
            officeqa_opd_action_only=True,
            officeqa_opd_prose_as_think=True,
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    samples = result if isinstance(result, list) else [result]
    # Fence-less turn stays discarded, exactly as in plain AOD.
    assert len(teacher.generate_payloads) == 2
    assert len(samples) == 2

    for payload in teacher.generate_payloads:
        scored = "".join(chr(t) for t in payload["input_ids"])
        current = scored.rsplit("<think>", 1)[1]
        assert current.startswith("\n\n</think>\n\n```bash"), (
            "the command must follow the canonical empty think directly"
        )
        assert "Checking." not in current and "I am lost." not in current

    for sample in samples:
        response = sample.tokens[-sample.response_length :]
        text = "".join(chr(t) for t in response)
        assert text.startswith("```bash") and text.rstrip().endswith("```")
        assert "Checking." not in text
        assert all(m == 1 for m in sample.loss_mask), (
            "the trimmed response IS the command; all of it carries loss"
        )


async def test_prose_as_think_teacher_no_think_trains_only_commands(
    corpus_dir, tmp_path
):
    """aod2op + prose-as-think: real rollout trained; the mask stays on the
    commands (same as plain aod2op) while the teacher request drops prose."""
    turns = [THINK_SEARCH_TURN, THINK_SUBMIT_TURN]
    async with (
        FakeSGLangServer(scripted_responder(turns)) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor,
            teacher,
            corpus_dir,
            tmp_path,
            officeqa_opd_teacher_no_think=True,
            officeqa_opd_prose_as_think=True,
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result
    response = sample.tokens[-sample.response_length :]
    text = "".join(chr(t) for t in response)
    assert "Peek at the file." in text and "Checking." in text, (
        "the real rollout keeps reasoning and prose"
    )
    trained = "".join(
        chr(t) for t, m in zip(response, sample.loss_mask, strict=True) if m
    )
    assert trained.count("```bash") == 2
    assert "Checking." not in trained and "<think>" not in trained
    for payload in teacher.generate_payloads:
        scored = "".join(chr(t) for t in payload["input_ids"])
        assert "Checking." not in scored.rsplit("<think>", 1)[1]
    on = [lp for lp, m in zip(sample.teacher_log_probs, sample.loss_mask, strict=True) if m]
    off = [lp for lp, m in zip(sample.teacher_log_probs, sample.loss_mask, strict=True) if not m]
    assert any(lp != 0.0 for lp in on)
    assert all(lp == 0.0 for lp in off)


async def test_prose_as_think_and_tail_loss_are_mutually_exclusive(
    corpus_dir, tmp_path
):
    async with (
        FakeSGLangServer(scripted_responder([THINK_SUBMIT_TURN])) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor,
            teacher,
            corpus_dir,
            tmp_path,
            officeqa_opd_action_only=True,
            officeqa_opd_tail_loss=True,
            officeqa_opd_prose_as_think=True,
        )
        sample = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    assert sample.status == Sample.Status.ABORTED
    assert "opposite treatments" in (sample.metadata or {})["abort_reason"]
    assert teacher.generate_payloads == []


def _thinking_teacher_responder(think_text):
    """Answer generation requests with a think block, scoring requests as usual.

    The two request kinds share the /generate route; generation carries a
    positive ``max_new_tokens`` while scoring pins it to 0.
    """

    async def responder(payload):
        params = payload.get("sampling_params") or {}
        if int(params.get("max_new_tokens") or 0) > 0:
            return sglang_response(think_text + "</think>")
        return teacher_scores(payload)

    return responder


async def test_teacher_think_conditions_the_judge_on_its_own_reasoning(
    corpus_dir, tmp_path
):
    """The judge deliberates first: the teacher generates its own think block
    and scores the student's command behind it, while the trained sample stays
    the student's real rollout."""
    turns = [THINK_SEARCH_TURN, THINK_SUBMIT_TURN]
    async with (
        FakeSGLangServer(scripted_responder(turns)) as actor,
        FakeSGLangServer(_thinking_teacher_responder("Teacher plans.")) as teacher,
    ):
        args = opd_args(
            actor,
            teacher,
            corpus_dir,
            tmp_path,
            officeqa_opd_teacher_no_think=True,
            officeqa_opd_teacher_think=True,
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result
    assert sample.status == Sample.Status.COMPLETED

    generation = [
        p
        for p in teacher.generate_payloads
        if int((p.get("sampling_params") or {}).get("max_new_tokens") or 0) > 0
    ]
    scoring = [
        p
        for p in teacher.generate_payloads
        if int((p.get("sampling_params") or {}).get("max_new_tokens") or 0) == 0
    ]
    assert generation and scoring

    # Generation: greedy by default, capped, stopped at the think close, and
    # fired from the same context with a freshly opened block.
    for payload in generation:
        params = payload["sampling_params"]
        assert params["temperature"] == 0.0
        assert params["max_new_tokens"] == 512
        assert params["stop"] == ["</think>"]
        assert "".join(chr(t) for t in payload["input_ids"]).endswith("<think>\n")

    # Scoring: the teacher's own deliberation fills the think slot, and the
    # student's visible tail follows it unchanged.
    scored = ["".join(chr(t) for t in p["input_ids"]) for p in scoring]
    for text in scored:
        tail = text.rsplit("<think>", 1)[1]
        assert tail.startswith("\nTeacher plans.\n</think>\n\n")

    # Student side: real rollout trained, loss and teacher scores on the
    # fenced commands only.
    response = sample.tokens[-sample.response_length :]
    text = "".join(chr(t) for t in response)
    assert "Peek at the file." in text
    trained = "".join(
        chr(t) for t, m in zip(response, sample.loss_mask, strict=True) if m
    )
    assert trained.count("```bash") == 2 and "<think>" not in trained
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


async def test_teacher_think_falls_back_to_empty_think_when_generation_fails(
    corpus_dir, tmp_path
):
    """A teacher that cannot generate (scoring-shaped reply) degrades the turn
    to the empty-think request instead of killing the episode."""
    async with (
        FakeSGLangServer(scripted_responder([THINK_SUBMIT_TURN])) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor,
            teacher,
            corpus_dir,
            tmp_path,
            officeqa_opd_teacher_no_think=True,
            officeqa_opd_teacher_think=True,
            officeqa_opd_attempts=1,
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result
    assert sample.status == Sample.Status.COMPLETED
    scoring = [
        p
        for p in teacher.generate_payloads
        if int((p.get("sampling_params") or {}).get("max_new_tokens") or 0) == 0
    ]
    assert scoring, "the fallback must still score the turn"
    for payload in scoring:
        tail = "".join(chr(t) for t in payload["input_ids"]).rsplit("<think>", 1)[1]
        assert tail.startswith("\n\n</think>")
    on_action = [
        lp
        for lp, m in zip(sample.teacher_log_probs, sample.loss_mask, strict=True)
        if m
    ]
    assert any(lp != 0.0 for lp in on_action)


async def test_teacher_think_requires_the_on_policy_base(corpus_dir, tmp_path):
    async with (
        FakeSGLangServer(scripted_responder([THINK_SUBMIT_TURN])) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor,
            teacher,
            corpus_dir,
            tmp_path,
            officeqa_opd_action_only=True,
            officeqa_opd_teacher_think=True,
        )
        sample = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    assert sample.status == Sample.Status.ABORTED
    assert "on-policy base" in (sample.metadata or {})["abort_reason"]
    assert teacher.generate_payloads == []


async def test_teacher_think_prestarts_during_student_generation(
    corpus_dir, tmp_path
):
    """The deliberation is fired from the turn-prompt observer — concurrent
    with the student's own generation — and the scorer consumes that task
    instead of generating a second time."""
    turns = [THINK_SEARCH_TURN, THINK_SUBMIT_TURN]
    async with (
        FakeSGLangServer(scripted_responder(turns)) as actor,
        FakeSGLangServer(_thinking_teacher_responder("Teacher plans.")) as teacher,
    ):
        args = opd_args(
            actor,
            teacher,
            corpus_dir,
            tmp_path,
            officeqa_opd_teacher_no_think=True,
            officeqa_opd_teacher_think=True,
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result
    assert sample.status == Sample.Status.COMPLETED

    generation = [
        p
        for p in teacher.generate_payloads
        if int((p.get("sampling_params") or {}).get("max_new_tokens") or 0) > 0
    ]
    # One deliberation per dispatched student turn — the prestart is consumed
    # by scoring, never regenerated.
    assert len(generation) == len(turns)


async def test_teacher_think_reaps_hung_prestarts_for_unscored_turns(
    corpus_dir, tmp_path
):
    """A deliberation prestarted for a turn that never becomes scorable (no
    fence) may still be in flight when the episode ends; reaping it must not
    leak the cancellation into the episode result (CancelledError is a
    BaseException — the original cleanup missed it)."""
    hang = asyncio.Event()

    async def responder(payload):
        params = payload.get("sampling_params") or {}
        if int(params.get("max_new_tokens") or 0) > 0:
            text = "".join(chr(t) for t in payload["input_ids"])
            if "Peek at the file." in text and "I am lost" not in text:
                await hang.wait()  # the fence-less turn's deliberation
            return sglang_response("Teacher plans.</think>")
        return teacher_scores(payload)

    turns = [THINK_SEARCH_TURN, NO_FENCE_TURN, THINK_SUBMIT_TURN]
    try:
        async with (
            FakeSGLangServer(scripted_responder(turns)) as actor,
            FakeSGLangServer(responder) as teacher,
        ):
            args = opd_args(
                actor,
                teacher,
                corpus_dir,
                tmp_path,
                officeqa_opd_teacher_no_think=True,
                officeqa_opd_teacher_think=True,
            )
            result = await opd_rollout.generate(
                args, make_base_sample(), {"max_new_tokens": 512}
            )
            await get_teacher_client(args).close()
    finally:
        hang.set()

    sample = result[0] if isinstance(result, list) else result
    assert sample.status == Sample.Status.COMPLETED
    on_action = [
        lp
        for lp, m in zip(sample.teacher_log_probs, sample.loss_mask, strict=True)
        if m
    ]
    assert any(lp != 0.0 for lp in on_action)


# --- mask-to-think (think distillation) --------------------------------------


async def test_mask_to_think_trains_only_the_think_region(corpus_dir, tmp_path):
    """Teacher scores the full real rollout; loss narrows to the think region
    (content through ``</think>`` inclusive). The mirror image of mask-think:
    commands and prose keep informing the teacher but carry no gradient, and a
    fence-less turn with a closed think KEEPS its loss."""
    turns = [THINK_SEARCH_TURN, NO_FENCE_TURN, THINK_SUBMIT_TURN]
    async with (
        FakeSGLangServer(scripted_responder(turns)) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor, teacher, corpus_dir, tmp_path, officeqa_opd_mask_to_think=True
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    samples = result if isinstance(result, list) else [result]
    trained = "".join(
        chr(t)
        for sample in samples
        for t, m in zip(
            sample.tokens[-sample.response_length :], sample.loss_mask, strict=True
        )
        if m
    )
    assert "Peek at the file.</think>" in trained
    assert "Hmm.</think>" in trained, "fence-less turn must keep its think loss"
    assert "Answer is 42.</think>" in trained
    assert trained.count("</think>") == 3
    assert "<think>" not in trained, "the template opener is not deliberation"
    assert "Checking." not in trained and "I am lost" not in trained
    assert "```bash" not in trained, "actions are untrained in this mode"

    # Teacher log-probs cover the whole turn (full-turn scoring), so trained
    # positions carry real scores.
    for sample in samples:
        on = [
            lp
            for lp, m in zip(sample.teacher_log_probs, sample.loss_mask, strict=True)
            if m
        ]
        if on:
            assert any(lp != 0.0 for lp in on)


async def test_mask_to_think_first_and_last_windows(corpus_dir, tmp_path):
    """first/last are POSITIONAL over the whole turn response, not over the
    think region: first lands on the opening deliberation (the template opens
    <think> in the prompt), last on whatever ends the turn — normally the
    fenced command. ByteTokenizer gives one char per token."""
    turns = [THINK_SEARCH_TURN, NO_FENCE_TURN, THINK_SUBMIT_TURN]
    for span, tokens, expected in (
        # responses: "<think>Peek at the file.</think>\n\nChecking.\n```bash\ncat ...\n```",
        #            "<think>Hmm.</think>\n\nI am lost.",
        #            "<think>Answer is 42.</think>\n\n```bash\nsubmit \"42\"\n```"
        ("first", 10, "<think>Pee" "<think>Hmm" "<think>Ans"),
        # last 6 of each: "...example.txt\n```" / "...I am lost." / '...42"\n```'
        ("last", 6, "xt\n```" + " lost." + '2"\n```'),
    ):
        async with (
            FakeSGLangServer(scripted_responder(turns)) as actor,
            FakeSGLangServer(teacher_responder) as teacher,
        ):
            args = opd_args(
                actor,
                teacher,
                corpus_dir,
                tmp_path,
                officeqa_opd_mask_to_think=True,
                officeqa_opd_think_span=span,
                officeqa_opd_think_span_tokens=tokens,
            )
            result = await opd_rollout.generate(
                args, make_base_sample(), {"max_new_tokens": 512}
            )
            await get_teacher_client(args).close()

        samples = result if isinstance(result, list) else [result]
        trained = "".join(
            chr(t)
            for sample in samples
            for t, m in zip(
                sample.tokens[-sample.response_length :], sample.loss_mask, strict=True
            )
            if m
        )
        assert trained == expected, (span, tokens, repr(trained))


def test_mask_to_think_validation():
    from types import SimpleNamespace

    from officeqa_rl.training.opd_rollout import parse_per_turn_modes

    with pytest.raises(ValueError, match="alternatives"):
        parse_per_turn_modes(
            SimpleNamespace(
                officeqa_opd_mask_to_think=True, officeqa_opd_mask_think=True
            )
        )
    with pytest.raises(ValueError, match="think_span must be one of"):
        parse_per_turn_modes(
            SimpleNamespace(
                officeqa_opd_mask_to_think=True, officeqa_opd_think_span="middle"
            )
        )
    with pytest.raises(ValueError, match="think_span_tokens > 0"):
        parse_per_turn_modes(
            SimpleNamespace(
                officeqa_opd_mask_to_think=True, officeqa_opd_think_span="first"
            )
        )
    with pytest.raises(ValueError, match="do not compose"):
        parse_per_turn_modes(
            SimpleNamespace(
                officeqa_opd_mask_to_think=True, officeqa_opd_tail_loss=True
            )
        )
    # spectator knobs are ignored when the mode is off
    modes = parse_per_turn_modes(SimpleNamespace(officeqa_opd_mask_think=True))
    assert not modes.mask_to_think


def test_distill_span_edges():
    from officeqa_rl.training.opd_rollout import _distill_span

    tok = ByteTokenizer()
    ids = _ids("no close tag here ```bash\nls\n```")
    assert _distill_span(tok, ids) is None, "span=all drops an unclosed think"
    # the positional windows never parse think, so they keep such a turn
    (lo, hi), = _distill_span(tok, ids, "first", 5)
    assert "".join(chr(t) for t in ids[lo:hi]) == "no cl"
    (lo, hi), = _distill_span(tok, ids, "last", 3)
    assert "".join(chr(t) for t in ids[lo:hi]) == "```"

    ids = _ids("<think></think>after")
    (lo, hi), = _distill_span(tok, ids)
    assert "".join(chr(t) for t in ids[lo:hi]) == "</think>", (
        "an empty think still trains the close decision"
    )

    # windows clamp to the response length
    ids = _ids("<think>ab</think>x")
    (lo, hi), = _distill_span(tok, ids, "first", 99)
    assert (lo, hi) == (0, len(ids))
    (lo, hi), = _distill_span(tok, ids, "last", 99)
    assert (lo, hi) == (0, len(ids))
    assert _distill_span(tok, [], "first", 4) is None


# ------------------------------------------------- gemma-4 think dialect

# Gemma-4 renders reasoning as a channel, not a tag pair, and its canonical
# empty block has no interior blank line. The dialect is read off the chat
# template, so the fake only needs that attribute to flip every marker.
GEMMA_TEMPLATE_SNIPPET = (
    "{%- if thinking_text -%}<|channel>thought\n{{ thinking_text }}<channel|>"
    "{%- endif -%}"
)
GEMMA_EMPTY = "<|channel>thought\n<channel|>"
GEMMA_TURN = (
    "<|channel>thought\nPeek at the file.<channel|>\nChecking.\n"
    "```bash\ncat officeqa_corpus/example.txt\n```"
)


class GemmaByteTokenizer(ByteTokenizer):
    chat_template = GEMMA_TEMPLATE_SNIPPET


def test_think_dialect_detected_from_chat_template():
    assert _think_dialect(GemmaByteTokenizer()) is _GEMMA_DIALECT
    assert _think_dialect(ByteTokenizer()) is _QWEN_DIALECT


def test_gemma_plan_splits_channel_and_fence():
    # Thinking enabled: the generation prompt ends at "<|turn>model\n" with no
    # channel marker, so the empty-think insert must be the full gemma block.
    tok = GemmaByteTokenizer()
    prompt = _ids("<|turn>model\n")
    plan = _action_only_plan(tok, prompt, _ids(GEMMA_TURN))
    visible_from = GEMMA_TURN.index("<channel|>") + len("<channel|>")
    fence_lo = GEMMA_TURN.index("```bash")
    assert plan.tail_start == visible_from
    assert plan.prefix_ids == tuple(_ids(GEMMA_EMPTY))
    assert plan.spans == ((fence_lo, len(GEMMA_TURN)),)


def test_gemma_plan_adds_nothing_when_template_precloses_the_channel():
    # Thinking disabled: the template itself writes the pre-closed block, so
    # inserting anything would duplicate it.
    tok = GemmaByteTokenizer()
    prompt = _ids("<|turn>model\n" + GEMMA_EMPTY)
    plan = _action_only_plan(tok, prompt, _ids("```bash\nsubmit \"42\"\n```"))
    assert plan.prefix_ids == ()


def test_gemma_tail_span_keeps_whole_visible_region():
    tok = GemmaByteTokenizer()
    found = _tail_span(tok, _ids(GEMMA_TURN))
    assert found is not None
    tail_start, spans = found
    visible_from = GEMMA_TURN.index("<channel|>") + len("<channel|>")
    assert tail_start == visible_from
    assert spans == ((visible_from, len(GEMMA_TURN)),)


def test_gemma_unclosed_channel_discards_the_turn():
    tok = GemmaByteTokenizer()
    assert _tail_span(tok, _ids("<|channel>thought\nstill musing")) is None


# ---------------------------------------------------------------------------
# teacher-think + tail loss: the whole visible tail follows the teacher's
# deliberation and carries the loss
# ---------------------------------------------------------------------------


async def test_teacher_think_with_tail_loss_scores_and_trains_the_whole_visible_tail(
    corpus_dir, tmp_path
):
    turns = [THINK_SEARCH_TURN, NO_FENCE_TURN, THINK_SUBMIT_TURN]
    async with (
        FakeSGLangServer(scripted_responder(turns)) as actor,
        FakeSGLangServer(_thinking_teacher_responder("Teacher plans.")) as teacher,
    ):
        args = opd_args(
            actor, teacher, corpus_dir, tmp_path,
            officeqa_opd_teacher_no_think=True,
            officeqa_opd_teacher_think=True,
            officeqa_opd_tail_loss=True,
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result
    assert sample.status == Sample.Status.COMPLETED
    scoring = [
        p for p in teacher.generate_payloads
        if int((p.get("sampling_params") or {}).get("max_new_tokens") or 0) == 0
    ]
    generation = [
        p for p in teacher.generate_payloads
        if int((p.get("sampling_params") or {}).get("max_new_tokens") or 0) > 0
    ]
    # Every turn is scorable under tail loss (the fence-less one included),
    # and each got a teacher deliberation before its scoring request.
    assert len(scoring) == 3 and len(generation) == 3
    scored = ["".join(chr(t) for t in p["input_ids"]) for p in scoring]
    # The request is context + teacher's think + the student's WHOLE visible
    # tail: prose and command, not just the fence.
    assert any("\nTeacher plans.\n</think>\n\nChecking.\n```bash" in t for t in scored)
    assert any("\nTeacher plans.\n</think>\n\nI am lost." in t for t in scored)
    assert not any("Peek at the file." in t.rsplit("<think>", 1)[1] for t in scored), \
        "the student's own reasoning never reaches the judge"

    response = sample.tokens[-sample.response_length :]
    text = "".join(chr(t) for t in response)
    assert "Peek at the file." in text, "the real rollout keeps its reasoning"
    trained = "".join(chr(t) for t, m in zip(response, sample.loss_mask, strict=True) if m)
    assert "Checking." in trained and "I am lost." in trained
    assert trained.count("```bash") == 2 and "<think>" not in trained
    on = [lp for lp, m in zip(sample.teacher_log_probs, sample.loss_mask, strict=True) if m]
    off = [lp for lp, m in zip(sample.teacher_log_probs, sample.loss_mask, strict=True) if not m]
    assert any(lp != 0.0 for lp in on) and all(lp == 0.0 for lp in off)


async def test_teacher_think_still_rejects_prose_as_think(corpus_dir, tmp_path):
    async with (
        FakeSGLangServer(scripted_responder([THINK_SUBMIT_TURN])) as actor,
        FakeSGLangServer(_thinking_teacher_responder("Teacher plans.")) as teacher,
    ):
        args = opd_args(
            actor, teacher, corpus_dir, tmp_path,
            officeqa_opd_teacher_no_think=True, officeqa_opd_teacher_think=True,
            officeqa_opd_prose_as_think=True,
        )
        sample = await opd_rollout.generate(args, make_base_sample(), {"max_new_tokens": 512})
        await get_teacher_client(args).close()
    assert sample.status == Sample.Status.ABORTED
    assert "prose-as-think" in (sample.metadata or {})["abort_reason"]
    assert teacher.generate_payloads == []


# ---------------------------------------------------------------------------
# tail-think: the tail loss pulled back into the last N tokens of the thought;
# the teacher reads the ending inside its own think block behind an omission
# note (student sequence minus the head of the thought)
# ---------------------------------------------------------------------------

OMITTED = opd_rollout._TAIL_THINK_OMITTED


def test_tail_think_plan_extends_into_the_thought_and_keeps_the_sequence():
    # THINK_SEARCH_TURN opens its own block here (no <think> in the prompt), so
    # the thought body starts after "<think>"; N=8 keeps its last 8 characters
    # ("he file." of "Peek at the file.").
    tok = ByteTokenizer()
    ids = _ids(THINK_SEARCH_TURN)
    body_from = len("<think>")
    close_lo = THINK_SEARCH_TURN.index("</think>")
    close_hi = close_lo + len("</think>")
    plan = opd_rollout._action_only_plan(
        tok, PROMPT_NO_THINK, ids, tail_loss=True, tail_think_tokens=8
    )
    assert plan.tail_start == close_lo - 8 >= body_from
    assert plan.spans == ((close_lo - 8, close_lo), (close_hi, len(ids)))
    assert plan.tail_think
    # The request is the student's own sequence from the ending on, behind a
    # prefix that re-opens the block and flags the omission.
    request = "".join(chr(t) for t in plan.request_ids(ids))
    assert request == "he file." + "</think>" + THINK_SEARCH_TURN[close_hi:]
    assert plan.prefix_ids == tuple(_ids("<think>\n" + OMITTED))
    # Scores stay contiguous over the request; the marker is scored but not trained.
    assert plan.score_index(close_lo - 8) == 0
    assert plan.score_index(close_hi) == 16


def test_tail_think_adds_no_opener_when_the_prompt_opened_the_block():
    tok = ByteTokenizer()
    ids = _ids(THINK_SEARCH_TURN)
    plan = opd_rollout._action_only_plan(
        tok, _ids("<assistant><think>\n"), ids, tail_loss=True, tail_think_tokens=8
    )
    assert plan.prefix_ids == tuple(_ids(OMITTED))
    assert plan.tail_think


def test_tail_think_never_reaches_past_the_thought_start():
    tok = ByteTokenizer()
    ids = _ids(THINK_SEARCH_TURN)
    plan = opd_rollout._action_only_plan(
        tok, PROMPT_NO_THINK, ids, tail_loss=True, tail_think_tokens=1000
    )
    assert plan.tail_start == len("<think>")
    trained = "".join(chr(t) for t in ids[plan.spans[0][0] : plan.spans[0][1]])
    assert trained == "Peek at the file."


def test_tail_think_leaves_an_empty_thought_alone():
    tok = ByteTokenizer()
    turn = '<think></think>\n\n```bash\nsubmit "42"\n```'
    ids = _ids(turn)
    plain = opd_rollout._action_only_plan(tok, PROMPT_NO_THINK, ids, tail_loss=True)
    plan = opd_rollout._action_only_plan(
        tok, PROMPT_NO_THINK, ids, tail_loss=True, tail_think_tokens=8
    )
    assert plan == plain  # empty-block prefix, visible tail only
    assert not plan.tail_think


def test_tail_think_is_off_by_default_and_for_the_fence_only_plan():
    tok = ByteTokenizer()
    ids = _ids(THINK_SEARCH_TURN)
    assert not opd_rollout._action_only_plan(tok, PROMPT_NO_THINK, ids, tail_loss=True).tail_think
    plan = opd_rollout._action_only_plan(
        tok, PROMPT_NO_THINK, ids, tail_loss=False, tail_think_tokens=8
    )
    assert not plan.tail_think
    assert plan.spans == opd_rollout._action_only_plan(tok, PROMPT_NO_THINK, ids).spans


def test_gemma_tail_think_plan_opens_the_channel():
    tok = GemmaByteTokenizer()
    ids = _ids(GEMMA_TURN)
    body_from = len("<|channel>thought\n")
    close_lo = GEMMA_TURN.index("<channel|>")
    plan = opd_rollout._action_only_plan(
        tok, _ids("<|turn>model\n"), ids, tail_loss=True, tail_think_tokens=8
    )
    assert plan.tail_start == close_lo - 8 >= body_from
    assert plan.prefix_ids == tuple(_ids("<|channel>thought\n" + OMITTED))
    request = "".join(chr(t) for t in plan.request_ids(ids))
    assert request == GEMMA_TURN[close_lo - 8:]
    assert request.startswith("he file.<channel|>")


async def test_tail_think_trains_the_thought_ending_on_the_real_rollout(
    corpus_dir, tmp_path
):
    """aod2op + tail + tailthink8: the real rollout is trained, the mask covers
    the last 8 thought characters plus the visible tail, the close marker and
    the rest of the thought carry no loss, and the teacher scored the ending
    inside its own think block behind the omission note."""
    turns = [THINK_SEARCH_TURN, THINK_SUBMIT_TURN]
    async with (
        FakeSGLangServer(scripted_responder(turns)) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor, teacher, corpus_dir, tmp_path,
            officeqa_opd_teacher_no_think=True, officeqa_opd_tail_loss=True,
            officeqa_opd_tail_think_tokens=8,
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result
    response = sample.tokens[-sample.response_length :]
    text = "".join(chr(t) for t in response)
    assert "Peek at the file." in text, "the real rollout keeps its reasoning"
    trained = "".join(
        chr(t) for t, m in zip(response, sample.loss_mask, strict=True) if m
    )
    assert "he file." in trained and "Checking." in trained
    assert "r is 42." in trained  # last 8 characters of the second thought
    assert "Peek" not in trained and "Answer" not in trained
    assert "</think>" not in trained and "<think>" not in trained
    on = [lp for lp, m in zip(sample.teacher_log_probs, sample.loss_mask, strict=True) if m]
    off = [lp for lp, m in zip(sample.teacher_log_probs, sample.loss_mask, strict=True) if not m]
    assert any(lp != 0.0 for lp in on)
    assert all(lp == 0.0 for lp in off)
    assert (sample.metadata or {})["tail_think"] == {"turns": 2, "tokens": 16}
    assert len(teacher.generate_payloads) == 2
    scored = "".join(chr(t) for t in teacher.generate_payloads[0]["input_ids"])
    assert "<think>\n" + OMITTED + "he file.</think>\n\nChecking." in scored
    assert _EMPTY_THINK not in scored


async def test_tail_think_requires_tail_loss(corpus_dir, tmp_path):
    async with (
        FakeSGLangServer(scripted_responder([THINK_SUBMIT_TURN])) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor, teacher, corpus_dir, tmp_path,
            officeqa_opd_teacher_no_think=True, officeqa_opd_tail_think_tokens=8,
        )
        sample = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()
    assert sample.status == Sample.Status.ABORTED
    assert "tail_think_tokens" in (sample.metadata or {})["abort_reason"]
    assert teacher.generate_payloads == []


async def test_tail_think_requires_the_on_policy_base(corpus_dir, tmp_path):
    async with (
        FakeSGLangServer(scripted_responder([THINK_SUBMIT_TURN])) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor, teacher, corpus_dir, tmp_path,
            officeqa_opd_action_only=True, officeqa_opd_tail_loss=True,
            officeqa_opd_tail_think_tokens=8,
        )
        sample = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()
    assert sample.status == Sample.Status.ABORTED
    assert "on-policy" in (sample.metadata or {})["abort_reason"]
    assert teacher.generate_payloads == []


async def test_tail_think_refuses_teacher_think(corpus_dir, tmp_path):
    async with (
        FakeSGLangServer(scripted_responder([THINK_SUBMIT_TURN])) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor, teacher, corpus_dir, tmp_path,
            officeqa_opd_teacher_no_think=True, officeqa_opd_tail_loss=True,
            officeqa_opd_tail_think_tokens=8, officeqa_opd_teacher_think=True,
        )
        sample = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()
    assert sample.status == Sample.Status.ABORTED
    assert "teacher_think" in (sample.metadata or {})["abort_reason"]
    assert teacher.generate_payloads == []


# --- fence delimiters out of the loss (officeqa_opd_mask_fence_delims) -------
#
# The opening ``` is where q38 disagrees most with the student: it scores the
# fence ~1.6 nats below the student (it would rather keep writing prose), so
# the tail loss pushes the student off the fence and onto its own <tool_call>
# opener. The delimiters are environment syntax; masking them keeps the
# command body and the prose trained and the request unchanged.


def test_fence_delimiter_ranges_cover_exactly_the_opener_and_closer():
    from officeqa_rl.training.opd_rollout import _fence_delimiter_ranges

    tok = ByteTokenizer()
    output = _ids("thought</think>\n\nLook:\n\n```bash\nls -la\n```")
    (open_lo, open_hi), (close_lo, close_hi) = _fence_delimiter_ranges(tok, output)
    text = "thought</think>\n\nLook:\n\n```bash\nls -la\n```"
    assert text[open_lo:open_hi] == "```bash\n"
    assert text[close_lo:close_hi] == "\n```"
    assert text[open_hi:close_lo] == "ls -la"
    # Same fence rule as _fence_span: none, several, or one inside the thought -> None.
    assert _fence_delimiter_ranges(tok, _ids("no fence here")) is None
    assert _fence_delimiter_ranges(tok, _ids("</think>\n```bash\na\n```\n```bash\nb\n```")) is None
    assert _fence_delimiter_ranges(tok, _ids("```bash\nx\n```</think>\nprose")) is None


def test_action_only_plan_masks_delimiters_but_keeps_the_request_and_spans():
    from officeqa_rl.training.opd_rollout import _action_only_plan

    tok = ByteTokenizer()
    prompt = _ids("<user>q<assistant><think>\n")
    output = _ids("t</think>\n\nSee:\n\n```bash\nls\n```")
    plain = _action_only_plan(tok, prompt, output, tail_loss=True)
    masked = _action_only_plan(tok, prompt, output, tail_loss=True, mask_fence_delims=True)
    assert masked.tail_start == plain.tail_start and masked.spans == plain.spans
    assert masked.request_ids(output) == plain.request_ids(output)
    assert plain.masked == () and len(masked.masked) == 2
    kept = "".join(chr(output[i]) for lo, hi in masked.spans for i in range(lo, hi) if not masked.is_masked(i))
    assert kept == "\n\nSee:\n\nls"
    # Fence-only mode: the trained span shrinks to the command body.
    fence_only = _action_only_plan(tok, prompt, output, mask_fence_delims=True)
    kept = "".join(chr(output[i]) for lo, hi in fence_only.spans for i in range(lo, hi) if not fence_only.is_masked(i))
    assert kept == "ls"


async def test_mask_fence_delims_drops_the_delimiters_from_the_trained_tail(
    corpus_dir, tmp_path
):
    turns = [THINK_SEARCH_TURN, NO_FENCE_TURN, THINK_SUBMIT_TURN]
    async with (
        FakeSGLangServer(scripted_responder(turns)) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor,
            teacher,
            corpus_dir,
            tmp_path,
            officeqa_opd_teacher_no_think=True,
            officeqa_opd_tail_loss=True,
            officeqa_opd_mask_fence_delims=True,
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result
    response = sample.tokens[-sample.response_length :]
    trained = "".join(
        chr(t) for t, m in zip(response, sample.loss_mask, strict=True) if m
    )
    assert "Checking." in trained and "I am lost." in trained  # prose still trained
    assert "```" not in trained  # no delimiter carries loss
    assert sample.metadata["fence_delims_masked"] > 0
    # Masked positions carry no loss and, like every other unmasked token,
    # an inert 0.0 teacher log-prob; the request itself is unchanged.
    text = "".join(chr(t) for t in response)
    first_fence = text.index("```bash")
    assert sample.loss_mask[first_fence] == 0
    assert sample.teacher_log_probs[first_fence] == 0.0


async def test_mask_fence_delims_requires_a_fence_planned_mode(corpus_dir, tmp_path):
    async with (
        FakeSGLangServer(scripted_responder([THINK_SUBMIT_TURN])) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor, teacher, corpus_dir, tmp_path, officeqa_opd_mask_fence_delims=True
        )
        sample = await opd_rollout.generate(args, make_base_sample(), {"max_new_tokens": 512})
        await get_teacher_client(args).close()
    assert sample.status == Sample.Status.ABORTED
    assert "mask_fence_delims" in (sample.metadata or {})["abort_reason"]


# --- fixed teacher think text (officeqa_opd_teacher_think_text) -------------
#
# The empty-think (-aod2op) request leaves the teacher's think block empty. A
# fixed sentence there, right before the scored tail, is the placement that
# measurably moves q38 toward opening the fence (a system block at the start
# of the context does not). The student never sees it; only the request grows.


def test_teacher_think_text_fills_the_think_block_and_keeps_the_spans():
    from officeqa_rl.training.opd_rollout import _action_only_plan, _think_prefix_text

    tok = ByteTokenizer()
    prompt = _ids("<user>q<assistant><think>\n")
    output = _ids("t</think>\n\nSee:\n\n```bash\nls\n```")
    plain = _action_only_plan(tok, prompt, output, tail_loss=True)
    fixed = _action_only_plan(tok, prompt, output, tail_loss=True, teacher_think_text="One sentence.")
    assert fixed.tail_start == plain.tail_start and fixed.spans == plain.spans
    assert "".join(chr(t) for t in fixed.prefix_ids) == _think_prefix_text(tok, prompt, "One sentence.")
    assert "".join(chr(t) for t in fixed.prefix_ids) == "One sentence.\n</think>"
    assert "".join(chr(t) for t in plain.prefix_ids) == "\n</think>"
    # Composes with the delimiter mask; the request is context + prefix + tail.
    both = _action_only_plan(tok, prompt, output, tail_loss=True, teacher_think_text="One sentence.", mask_fence_delims=True)
    assert both.prefix_ids == fixed.prefix_ids and len(both.masked) == 2


async def test_teacher_think_text_reaches_every_teacher_request(corpus_dir, tmp_path):
    turns = [THINK_SEARCH_TURN, THINK_SUBMIT_TURN]
    async with (
        FakeSGLangServer(scripted_responder(turns)) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(
            actor,
            teacher,
            corpus_dir,
            tmp_path,
            officeqa_opd_teacher_no_think=True,
            officeqa_opd_tail_loss=True,
            officeqa_opd_teacher_think_text="One sentence.",
        )
        result = await opd_rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}
        )
        await get_teacher_client(args).close()

    sample = result[0] if isinstance(result, list) else result
    assert teacher.generate_payloads, "the teacher was asked to score"
    for actor_payload, teacher_payload in zip(
        actor.generate_payloads, teacher.generate_payloads, strict=True
    ):
        prompt_ids = actor_payload["input_ids"]
        sent = "".join(chr(t) for t in teacher_payload["input_ids"])
        prompt = "".join(chr(t) for t in prompt_ids)
        assert sent.startswith(prompt)
        # The fixed text sits in the teacher's think block (opened by the
        # template on Qwen, opened here when the template did not),
        # immediately before the student's visible tail.
        from officeqa_rl.training.opd_rollout import _think_prefix_text

        expected_prefix = _think_prefix_text(ByteTokenizer(), prompt_ids, "One sentence.")
        assert "One sentence.\n</think>" in expected_prefix
        assert sent[len(prompt):].startswith(expected_prefix)
        assert teacher_payload["logprob_start_len"] == len(prompt_ids) + len(expected_prefix) - 1
    # The student's own sample never carries the text.
    assert "One sentence." not in "".join(chr(t) for t in sample.tokens)


async def test_teacher_think_text_requires_the_empty_think_request(corpus_dir, tmp_path):
    for extra, needle in (
        ({}, "teacher_think_text"),
        ({"officeqa_opd_teacher_no_think": True, "officeqa_opd_teacher_think": True}, "pick one"),
    ):
        async with (
            FakeSGLangServer(scripted_responder([THINK_SUBMIT_TURN])) as actor,
            FakeSGLangServer(teacher_responder) as teacher,
        ):
            args = opd_args(
                actor, teacher, corpus_dir, tmp_path,
                officeqa_opd_teacher_think_text="One sentence.", **extra,
            )
            sample = await opd_rollout.generate(args, make_base_sample(), {"max_new_tokens": 512})
            await get_teacher_client(args).close()
        assert sample.status == Sample.Status.ABORTED
        assert needle in (sample.metadata or {})["abort_reason"]
