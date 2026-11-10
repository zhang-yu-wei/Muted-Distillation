"""Exact-token and loss-mask tests for the trajectory export used by the
slime rollout hook (``build_training_samples``).

These tests hand-craft ``SessionTurn`` records with known token ids, so every
assertion is against the complete expected token sequence and mask, not just
lengths.
"""

import pytest

pytest.importorskip("slime", reason="requires the vendored slime on PYTHONPATH")

from slime.utils.types import Sample

from officeqa_rl.training.rollout import build_training_samples
from officeqa_rl.training.slime_sglang_backend import SessionTurn
from officeqa_rl.types import GeneratedTurn


class IdTokenizer:
    """Decode token ids to a readable, exactly checkable string."""

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return " ".join(str(i) for i in ids)


SYSTEM = {"role": "system", "content": "s"}
QUESTION = {"role": "user", "content": "q"}
ASSISTANT_1 = {"role": "assistant", "content": "a1"}
OBSERVATION = {"role": "user", "content": "o1"}


def session_turn(prompt_messages, prompt_ids, output_ids, log_probs, *, text, finish="stop"):
    return SessionTurn(
        prompt_messages=prompt_messages,
        turn=GeneratedTurn(
            text=text,
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            output_log_probs=log_probs,
            finish_reason=finish,
        ),
    )


def two_clean_turns():
    """Turn 2's prompt exactly extends turn 1's prompt + sampled output."""
    turn_1 = session_turn(
        [SYSTEM, QUESTION], [1, 2, 3], [10, 11], [-0.1, -0.2], text="a1"
    )
    turn_2 = session_turn(
        [SYSTEM, QUESTION, ASSISTANT_1, OBSERVATION],
        [1, 2, 3, 10, 11, 4, 5],
        [12],
        [-0.3],
        text="a2",
    )
    return [turn_1, turn_2]


def make_base_sample(**overrides):
    values = dict(index=0, group_index=0, rollout_id=None, prompt="q", label="42")
    values.update(overrides)
    return Sample(**values)


def test_two_turn_exact_tokens_masks_and_log_probs():
    samples = build_training_samples(
        session_turns=two_clean_turns(),
        base_sample=make_base_sample(),
        reward=1.0,
        tokenizer=IdTokenizer(),
        extra_metadata={"uid": "UID0001"},
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.tokens == [1, 2, 3, 10, 11, 4, 5, 12]
    assert sample.response_length == 5
    # Model outputs train (1); the observation prompt tail does not (0).
    assert sample.loss_mask == [1, 1, 0, 0, 1]
    assert sample.rollout_log_probs == [-0.1, -0.2, 0.0, 0.0, -0.3]
    assert sample.reward == 1.0
    assert sample.rollout_id == 0  # falls back to base_sample.index
    assert sample.metadata["uid"] == "UID0001"
    assert sample.metadata["truncated"] is False
    assert sample.response == "10 11 4 5 12"


def test_sample_contract_invariants_hold():
    samples = build_training_samples(
        session_turns=two_clean_turns(),
        base_sample=make_base_sample(),
        reward=0.0,
        tokenizer=IdTokenizer(),
    )

    sample = samples[0]
    assert len(sample.loss_mask) == sample.response_length
    assert len(sample.rollout_log_probs) == sample.response_length
    assert len(sample.tokens) >= sample.response_length
    assert sample.reward is not None
    assert isinstance(sample.status, Sample.Status)


def test_fork_siblings_share_rollout_id_and_each_carries_full_reward():
    """Re-tokenization drift too large to absorb splits the trajectory; the
    chosen fan-out reward semantics assign the full terminal reward to every
    sibling (slime averages within a shared rollout_id)."""
    turn_1 = session_turn(
        [SYSTEM, QUESTION], [1, 2, 3], [10, 11], [-0.1, -0.2], text="a1"
    )
    drifted_turn_2 = session_turn(
        [SYSTEM, QUESTION, ASSISTANT_1, OBSERVATION],
        [9, 9, 9],  # shares no prefix with the held [1, 2, 3, 10, 11]
        [12],
        [-0.3],
        text="a2",
    )

    samples = build_training_samples(
        session_turns=[turn_1, drifted_turn_2],
        base_sample=make_base_sample(rollout_id=7),
        reward=1.0,
        tokenizer=IdTokenizer(),
    )

    assert len(samples) == 2
    first, second = samples
    assert first.tokens == [1, 2, 3, 10, 11]
    assert first.response_length == 2
    assert first.loss_mask == [1, 1]
    assert first.rollout_log_probs == [-0.1, -0.2]
    assert second.tokens == [9, 9, 9, 12]
    assert second.response_length == 1
    assert second.loss_mask == [1]
    assert second.rollout_log_probs == [-0.3]
    assert [s.reward for s in samples] == [1.0, 1.0]
    assert [s.rollout_id for s in samples] == [7, 7]


def test_final_length_turn_marks_truncated_and_is_masked():
    turn_1 = session_turn(
        [SYSTEM, QUESTION], [1, 2, 3], [10, 11], [-0.1, -0.2], text="a1"
    )
    exhausted_turn = session_turn(
        [SYSTEM, QUESTION, ASSISTANT_1, OBSERVATION],
        [1, 2, 3, 10, 11, 4, 5],
        [],
        [],
        text="",
        finish="length",
    )

    samples = build_training_samples(
        session_turns=[turn_1, exhausted_turn],
        base_sample=make_base_sample(),
        reward=0.0,
        tokenizer=IdTokenizer(),
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.metadata["truncated"] is True
    assert sample.tokens == [1, 2, 3, 10, 11, 4, 5]
    assert sample.loss_mask == [1, 1, 0, 0]
    assert sample.rollout_log_probs == [-0.1, -0.2, 0.0, 0.0]


def test_max_sample_tokens_truncates_tokens_masks_and_log_probs_together():
    samples = build_training_samples(
        session_turns=two_clean_turns(),
        base_sample=make_base_sample(),
        reward=1.0,
        tokenizer=IdTokenizer(),
        max_sample_tokens=6,
    )

    sample = samples[0]
    assert sample.tokens == [1, 2, 3, 10, 11, 4]
    assert sample.response_length == 3
    assert sample.loss_mask == [1, 1, 0]
    assert sample.rollout_log_probs == [-0.1, -0.2, 0.0]
    assert sample.response == "10 11 4"


def test_sessions_without_trained_tokens_export_nothing():
    assert (
        build_training_samples(
            session_turns=[],
            base_sample=make_base_sample(),
            reward=0.0,
            tokenizer=IdTokenizer(),
        )
        == []
    )

    only_length_turn = session_turn(
        [SYSTEM, QUESTION], [1, 2, 3], [], [], text="", finish="length"
    )
    assert (
        build_training_samples(
            session_turns=[only_length_turn],
            base_sample=make_base_sample(),
            reward=0.0,
            tokenizer=IdTokenizer(),
        )
        == []
    )
