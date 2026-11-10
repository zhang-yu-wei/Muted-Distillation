"""Tests for the fan-out-aware dynamic-sampling filter."""

from types import SimpleNamespace

import pytest

pytest.importorskip("slime", reason="requires the vendored slime on PYTHONPATH")

from slime.utils.types import Sample

from officeqa_rl.training.filters import check_execution_reward_nonzero_std

ARGS = SimpleNamespace(reward_key=None)


def _sample(reward: float) -> Sample:
    return Sample(prompt="p", response="r", reward=reward)


def _siblings(reward: float, count: int) -> list[Sample]:
    return [_sample(reward) for _ in range(count)]


def test_mixed_scalar_group_is_kept():
    group = [_sample(1.0), _sample(0.0), _sample(0.0)]
    assert check_execution_reward_nonzero_std(ARGS, group).keep


def test_uniform_scalar_group_is_dropped_with_reason():
    output = check_execution_reward_nonzero_std(ARGS, [_sample(0.0)] * 4)
    assert not output.keep
    assert output.reason == "zero_std_0.0"


def test_nested_group_mixing_scalars_and_sibling_lists_is_kept():
    group = [_siblings(1.0, 3), _sample(0.0), _siblings(0.0, 2)]
    assert check_execution_reward_nonzero_std(ARGS, group).keep


def test_sibling_count_does_not_create_fake_contrast():
    # All executions agree on reward; uneven fan-out must not look like std.
    group = [_siblings(1.0, 50), _sample(1.0), _siblings(1.0, 2)]
    output = check_execution_reward_nonzero_std(ARGS, group)
    assert not output.keep
    assert output.reason == "zero_std_1.0"


def test_contrast_carried_only_by_a_large_sibling_list_is_kept():
    group = [_siblings(0.0, 50), _sample(1.0)]
    assert check_execution_reward_nonzero_std(ARGS, group).keep


def test_singleton_group_is_dropped():
    # std of one value is NaN; a single execution has no contrast.
    assert not check_execution_reward_nonzero_std(ARGS, [_sample(1.0)]).keep
