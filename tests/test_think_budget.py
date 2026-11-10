"""Think-budget generation: a hard cap on the thought channel, enforced by the
SGLang backend in two /generate phases, with the forced close loss-masked.

The fake tokenizer is one token per character, so ``{`` and ``}`` stand in for
the channel-open and channel-close tokens and every assertion is against the
exact token sequence the trainer would see.
"""

import pytest

pytest.importorskip("slime", reason="requires the vendored slime on PYTHONPATH")

from slime.utils.types import Sample

from officeqa_rl.training import metrics as officeqa_metrics
from officeqa_rl.training.rollout import _forced_close_turns, build_training_samples
from officeqa_rl.training.slime_sglang_backend import SlimeSGLangBackend
from tests.sglang_fakes import ByteTokenizer, FakeSGLangServer, sglang_response

OPEN, CLOSE = ord("{"), ord("}")
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]


def ids(text: str) -> list[int]:
    return [ord(ch) for ch in text]


class Script:
    """Serve canned /generate responses in order and keep every payload."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.payloads = []

    async def __call__(self, payload):
        self.payloads.append(payload)
        assert self.responses, "unexpected extra /generate request"
        return self.responses.pop(0)


def make_backend(server, **overrides) -> SlimeSGLangBackend:
    values = dict(
        base_url=server.base_url, tokenizer=ByteTokenizer(), max_context_len=4096,
        think_budget=8, think_open_token_id=OPEN, think_close_token_id=CLOSE,
    )
    values.update(overrides)
    return SlimeSGLangBackend(**values)


def base_sample() -> Sample:
    return Sample(index=0, group_index=0, rollout_id=None, prompt="q", label="42")


async def test_forced_close_splits_records_and_masks_only_the_close():
    # Phase 1 hits the 8-token budget with the channel still open.
    script = Script(sglang_response("{think12", finish_reason="length"), sglang_response("act"))
    async with FakeSGLangServer(script) as server:
        backend = make_backend(server)
        turn = await backend.generate(MESSAGES, {"max_new_tokens": 50})

    p1, p2 = script.payloads
    assert p1["sampling_params"]["max_new_tokens"] == 8
    assert CLOSE in p1["sampling_params"]["stop_token_ids"]
    assert p2["input_ids"] == p1["input_ids"] + ids("{think12") + [CLOSE]
    assert p2["sampling_params"]["max_new_tokens"] == 50 - 8 - 1
    assert "stop_token_ids" not in p2["sampling_params"]

    first, second = backend.session_turns
    assert first.turn.output_ids == ids("{think12")
    assert first.turn.finish_reason == "think_budget"  # not "length": the turn still acted
    assert first.turn.usage == {"think_budget_forced": True}
    assert second.turn.prompt_ids == first.turn.prompt_ids + first.turn.output_ids + [CLOSE]
    assert second.turn.output_ids == ids("act") and second.turn.usage is None
    # What the agent sees is the whole turn.
    assert turn.output_ids == ids("{think12") + [CLOSE] + ids("act")
    assert turn.text == "{think12}act"
    assert backend.forced_close_turns == 1
    assert _forced_close_turns(backend.session_turns) == 1

    # Trajectory export. The two records share prompt_messages, so the
    # manager forks them (the per-turn fan-out every agent turn gets): the
    # thought is trained in one sample and the action in the other, each
    # exactly once, and the forced close is never a trained token.
    samples = build_training_samples(
        session_turns=list(backend.session_turns), base_sample=base_sample(),
        reward=0.0, tokenizer=ByteTokenizer(), extra_metadata={"uid": "u"},
    )
    assert len(samples) == 2
    trained = []
    for sample in samples:
        response = sample.tokens[-sample.response_length:]
        trained.append([t for t, m in zip(response, sample.loss_mask) if m])
        assert not any(t == CLOSE and m for t, m in zip(response, sample.loss_mask))
        assert not sample.metadata.get("truncated")
    assert sorted(trained) == sorted([ids("{think12"), ids("act")])
    # The action sample sees the forced close as context, right before the action.
    action_sample = next(s for s in samples if s.tokens[-3:] == ids("act"))
    assert action_sample.tokens[-4] == CLOSE


async def test_natural_close_resumes_as_one_record():
    # Phase 1 stops on the sampled close (no_stop_trim keeps it in the output).
    script = Script(sglang_response("{t}"), sglang_response("act"))
    async with FakeSGLangServer(script) as server:
        backend = make_backend(server)
        await backend.generate(MESSAGES, {"max_new_tokens": 50})

    p1, p2 = script.payloads
    assert p2["input_ids"] == p1["input_ids"] + ids("{t}")  # no close inserted
    assert p2["sampling_params"]["max_new_tokens"] == 50 - 3
    [record] = backend.session_turns
    assert record.turn.output_ids == ids("{t}act") and record.turn.usage is None
    assert backend.forced_close_turns == 0


async def test_turn_ending_before_any_close_makes_one_request():
    script = Script(sglang_response("act"))
    async with FakeSGLangServer(script) as server:
        backend = make_backend(server)
        await backend.generate(MESSAGES, {"max_new_tokens": 50})
    assert len(script.payloads) == 1
    [record] = backend.session_turns
    assert record.turn.output_ids == ids("act") and record.turn.usage is None


async def test_cap_inside_visible_text_resumes_without_forcing():
    # Budget reached but no channel was ever opened: nothing to close.
    script = Script(sglang_response("visible1", finish_reason="length"), sglang_response("more"))
    async with FakeSGLangServer(script) as server:
        backend = make_backend(server)
        await backend.generate(MESSAGES, {"max_new_tokens": 50})
    p1, p2 = script.payloads
    assert p2["input_ids"] == p1["input_ids"] + ids("visible1")
    [record] = backend.session_turns
    assert record.turn.output_ids == ids("visible1more") and record.turn.usage is None
    assert backend.forced_close_turns == 0


async def test_forced_close_with_no_room_left_for_an_action():
    script = Script(sglang_response("{think12", finish_reason="length"))
    async with FakeSGLangServer(script) as server:
        backend = make_backend(server)
        turn = await backend.generate(MESSAGES, {"max_new_tokens": 8})
    assert len(script.payloads) == 1
    [first] = backend.session_turns
    assert first.turn.usage == {"think_budget_forced": True}
    assert turn.output_ids == ids("{think12") + [CLOSE]
    assert turn.finish_reason == "length"


async def test_budget_off_is_the_single_shot_path():
    script = Script(sglang_response("{think}act"))
    async with FakeSGLangServer(script) as server:
        backend = make_backend(
            server, think_budget=0, think_open_token_id=None, think_close_token_id=None
        )
        turn = await backend.generate(MESSAGES, {"max_new_tokens": 50})
    [payload] = script.payloads
    assert payload["sampling_params"]["max_new_tokens"] == 50
    assert "stop_token_ids" not in payload["sampling_params"]
    assert len(backend.session_turns) == 1
    assert turn.output_ids == ids("{think}act")


def test_budget_requires_the_channel_marker_ids():
    with pytest.raises(ValueError, match="think_budget requires"):
        SlimeSGLangBackend(
            base_url="http://x", tokenizer=ByteTokenizer(), max_context_len=10, think_budget=4
        )


def test_metrics_report_forced_close_share():
    def episode_sample(run_id, forced, steps):
        sample = base_sample()
        sample.status = Sample.Status.COMPLETED
        sample.reward = 0.0
        sample.metadata = {
            "run_id": run_id, "steps": steps, "exit_status": "submitted",
            "episode_output_tokens": 10, "forced_close_turns": forced,
        }
        return sample

    metrics = officeqa_metrics._truncation_metrics(
        [episode_sample("a", 2, 10), episode_sample("b", 0, 10)],
        max_response_len=8192, max_context_len=65536,
    )
    assert metrics["episode/forced_close_turns_mean"] == 1.0
    assert metrics["episode/forced_close_share"] == pytest.approx(2 / 20)

    # Budget off: the metadata key is absent and so are the metrics.
    off = episode_sample("c", None, 10)
    del off.metadata["forced_close_turns"]
    metrics = officeqa_metrics._truncation_metrics(
        [off], max_response_len=8192, max_context_len=65536
    )
    assert "episode/forced_close_share" not in metrics
    assert "episode/forced_close_turns_mean" not in metrics


# --- Qwen shape: the TEMPLATE opens the channel ------------------------------
# Qwen3.5's chat template ends the generation prompt with ``<think>\n``, so the
# model starts inside the thought and its output carries only the close. The
# open marker is the prompt's second-to-last token (a newline follows it) and
# never appears in the output. The original budget check looked for the open
# marker in the output alone and so never fired on Qwen (2026-09-09).


class QwenByteTokenizer(ByteTokenizer):
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        return super().apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tokenize=tokenize
        ) + [OPEN, ord("\n")]


def test_qwen_prompt_ends_with_open_marker_then_newline():
    # Pins the shape the fix must handle: last token is NOT the marker.
    prompt = QwenByteTokenizer().apply_chat_template(
        MESSAGES, add_generation_prompt=True, tokenize=True
    )
    assert prompt[-2:] == [OPEN, ord("\n")]


async def test_qwen_channel_opened_by_prompt_is_force_closed_at_the_budget():
    # Phase 1 hits the 8-token budget still inside the thought; the output
    # holds no open marker because the prompt supplied it.
    script = Script(sglang_response("think123", finish_reason="length"), sglang_response("act"))
    async with FakeSGLangServer(script) as server:
        backend = make_backend(server, tokenizer=QwenByteTokenizer())
        turn = await backend.generate(MESSAGES, {"max_new_tokens": 50})

    p1, p2 = script.payloads
    assert OPEN not in ids("think123")  # the shape under test
    assert p2["input_ids"] == p1["input_ids"] + ids("think123") + [CLOSE]
    assert p2["sampling_params"]["max_new_tokens"] == 50 - 8 - 1

    first, second = backend.session_turns
    assert first.turn.finish_reason == "think_budget"
    assert first.turn.usage == {"think_budget_forced": True}
    assert second.turn.prompt_ids == first.turn.prompt_ids + ids("think123") + [CLOSE]
    assert turn.output_ids == ids("think123") + [CLOSE] + ids("act")
    assert backend.forced_close_turns == 1
    assert _forced_close_turns(backend.session_turns) == 1


async def test_qwen_natural_close_within_budget_resumes_as_one_record():
    script = Script(sglang_response("t}"), sglang_response("act"))
    async with FakeSGLangServer(script) as server:
        backend = make_backend(server, tokenizer=QwenByteTokenizer())
        await backend.generate(MESSAGES, {"max_new_tokens": 50})
    p1, p2 = script.payloads
    assert p2["input_ids"] == p1["input_ids"] + ids("t}")  # no close inserted
    [record] = backend.session_turns
    assert record.turn.output_ids == ids("t}act") and record.turn.usage is None
    assert backend.forced_close_turns == 0


async def test_qwen_cap_after_a_natural_close_is_visible_text_not_forced():
    # Thought closed early, then the model kept writing visible text past the
    # budget in phase 2: the close came from the model, nothing is forced.
    script = Script(sglang_response("t}"), sglang_response("visible-more", finish_reason="length"))
    async with FakeSGLangServer(script) as server:
        backend = make_backend(server, tokenizer=QwenByteTokenizer())
        turn = await backend.generate(MESSAGES, {"max_new_tokens": 50})
    [record] = backend.session_turns
    assert record.turn.output_ids == ids("t}visible-more")
    assert turn.finish_reason == "length"
    assert backend.forced_close_turns == 0


# --- single-record forced close (span-planned modes) ------------------------
#
# The two-record split above is right for a loss over the whole output. Under
# the fence/tail planners (-aod2op, -tail, -maskthink) it is not: they look
# for </think> in a record's OUTPUT, and the action record's close sits in its
# prompt, so the record has no visible region and is discarded (14 of 18
# forced closes in one measured step). With think_budget_single_record the
# backend records the whole turn once; the planners cut the trained span
# after the close they find, so the thought and the forced close carry no loss.


async def test_qwen_single_record_forced_close_records_the_whole_turn_once():
    script = Script(sglang_response("think123", finish_reason="length"), sglang_response("act"))
    async with FakeSGLangServer(script) as server:
        backend = make_backend(server, tokenizer=QwenByteTokenizer(), think_budget_single_record=True)
        turn = await backend.generate(MESSAGES, {"max_new_tokens": 50})
    p1, p2 = script.payloads
    assert p2["input_ids"] == p1["input_ids"] + ids("think123") + [CLOSE]  # generation unchanged
    [record] = backend.session_turns
    assert record.turn.prompt_ids == p1["input_ids"]
    assert record.turn.output_ids == ids("think123") + [CLOSE] + ids("act")
    assert record.turn.output_log_probs[len("think123")] == 0.0  # the forced close was never sampled
    assert record.turn.finish_reason == "stop"  # phase 2's reason, not think_budget
    assert record.turn.usage == {"think_budget_forced": True}
    assert turn.output_ids == record.turn.output_ids
    assert backend.forced_close_turns == 1
    assert _forced_close_turns(backend.session_turns) == 1
    samples = build_training_samples(
        session_turns=list(backend.session_turns), base_sample=base_sample(),
        reward=0.0, tokenizer=QwenByteTokenizer(), extra_metadata={"uid": "u"},
    )
    [sample] = samples
    assert sample.tokens[-sample.response_length:] == record.turn.output_ids


def test_tail_planner_excludes_the_forced_close_from_the_trained_span():
    # What makes the single record safe: the tail span starts AFTER the close.
    from officeqa_rl.training.opd_rollout import _fence_span, _tail_span

    tok = ByteTokenizer()
    output = ids("thought</think>\n\n```bash\nls\n```")
    tail_start, spans = _tail_span(tok, output)
    assert tail_start == len("thought</think>")
    assert spans == ((tail_start, len(output)),)
    fence_start, fence_spans = _fence_span(tok, output)
    assert fence_start == tail_start and fence_spans[0][0] >= tail_start
    # And why two records fail: an output whose close sits in the prompt.
    assert _tail_span(tok, ids("\n\n```bash\nls\n```")) is None


async def test_qwen_single_record_with_no_room_left_ends_the_turn_on_length():
    script = Script(sglang_response("think123", finish_reason="length"))
    async with FakeSGLangServer(script) as server:
        backend = make_backend(server, tokenizer=QwenByteTokenizer(), think_budget_single_record=True)
        turn = await backend.generate(MESSAGES, {"max_new_tokens": 9})
    [record] = backend.session_turns
    assert record.turn.output_ids == ids("think123") + [CLOSE]
    assert record.turn.finish_reason == "length" and record.turn.usage == {"think_budget_forced": True}
    assert turn.output_ids == record.turn.output_ids


async def test_single_record_off_keeps_the_two_record_split():
    script = Script(sglang_response("think123", finish_reason="length"), sglang_response("act"))
    async with FakeSGLangServer(script) as server:
        backend = make_backend(server, tokenizer=QwenByteTokenizer())
        await backend.generate(MESSAGES, {"max_new_tokens": 50})
    assert len(backend.session_turns) == 2 and backend.think_budget_single_record is False

