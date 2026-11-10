from argparse import Namespace

import pytest
import torch

from slime.ray.rollout import _flatten_rollout_samples, _normalize_rewards_per_rollout
from slime.utils.types import Sample


NUM_GPUS = 0


def _sample(*, group=0, rollout_id=None, reward=0.0, status=Sample.Status.COMPLETED):
    return Sample(group_index=group, rollout_id=rollout_id, reward=reward, status=status)


def _normalization_args(*, use_std=True, estimator="grpo"):
    return Namespace(advantage_estimator=estimator, grpo_std_normalization=use_std)


@pytest.mark.unit
def test_flatten_accepts_mixed_scalar_and_fanout_results():
    scalar = _sample()
    one_element_list = [_sample(rollout_id=10)]
    siblings = [_sample(rollout_id=11), _sample(rollout_id=11), _sample(rollout_id=11)]
    aborted_scalar = _sample(status=Sample.Status.ABORTED)

    nested = [[scalar, one_element_list, siblings, aborted_scalar]]

    assert _flatten_rollout_samples(nested) == [scalar, *one_element_list, *siblings, aborted_scalar]


@pytest.mark.unit
@pytest.mark.parametrize("list_first", [False, True])
def test_flatten_inspects_every_element_regardless_of_first_shape(list_first):
    scalar = _sample()
    siblings = [_sample(rollout_id=12), _sample(rollout_id=12)]
    executions = [siblings, scalar] if list_first else [scalar, siblings]

    assert _flatten_rollout_samples([executions]) == ([*siblings, scalar] if list_first else [scalar, *siblings])


@pytest.mark.unit
def test_flatten_rejects_missing_sibling_rollout_id_with_path():
    nested = [[_sample(), [_sample(rollout_id=20), _sample(rollout_id=None)]]]

    with pytest.raises(ValueError, match=r"data\[0\]\[1\].*rollout_id is unset"):
        _flatten_rollout_samples(nested)


@pytest.mark.unit
def test_flatten_rejects_inconsistent_sibling_rollout_ids_with_path():
    nested = [[[_sample(rollout_id=20), _sample(rollout_id=21)]]]

    with pytest.raises(ValueError, match=r"data\[0\]\[0\].*must share rollout_id"):
        _flatten_rollout_samples(nested)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("nested", "error", "message"),
    [
        ([[]], ValueError, r"empty list.*data\[0\]"),
        ([[object()]], TypeError, r"data\[0\]\[0\].*object"),
    ],
)
def test_flatten_rejects_empty_and_unexpected_nodes_with_path(nested, error, message):
    with pytest.raises(error, match=message):
        _flatten_rollout_samples(nested)


def _fanout_group(success_siblings):
    return [
        *[_sample(group=100, rollout_id=1, reward=1.0) for _ in range(success_siblings)],
        _sample(group=100, rollout_id=2, reward=0.0),
        _sample(group=100, rollout_id=3, reward=0.0),
        _sample(group=100, rollout_id=4, reward=0.0),
    ]


def _advantages_by_rollout(samples, advantages):
    by_rollout = {}
    for sample, advantage in zip(samples, advantages, strict=True):
        by_rollout.setdefault(sample.rollout_id, []).append(advantage)
    for rollout_id, values in by_rollout.items():
        assert values == pytest.approx([values[0]] * len(values)), f"rollout {rollout_id} siblings disagree"
    return {rollout_id: values[0] for rollout_id, values in by_rollout.items()}


@pytest.mark.unit
def test_reward_normalization_is_invariant_to_sibling_count():
    three_siblings = _fanout_group(3)
    fifty_siblings = _fanout_group(50)

    three_advantages = _normalize_rewards_per_rollout(
        _normalization_args(), three_siblings, [sample.reward for sample in three_siblings]
    )
    fifty_advantages = _normalize_rewards_per_rollout(
        _normalization_args(), fifty_siblings, [sample.reward for sample in fifty_siblings]
    )

    expected = {1: 1.5, 2: -0.5, 3: -0.5, 4: -0.5}
    assert _advantages_by_rollout(three_siblings, three_advantages) == pytest.approx(expected, abs=1e-5)
    assert _advantages_by_rollout(fifty_siblings, fifty_advantages) == pytest.approx(expected, abs=1e-5)


@pytest.mark.unit
def test_reward_normalization_uses_prompt_local_rollout_statistics():
    samples = [
        _sample(group=1, rollout_id=1, reward=1.0),
        _sample(group=2, rollout_id=3, reward=0.0),
        _sample(group=1, rollout_id=2, reward=0.0),
        _sample(group=2, rollout_id=4, reward=0.0),
        _sample(group=1, rollout_id=1, reward=1.0),
    ]

    advantages = _normalize_rewards_per_rollout(
        _normalization_args(use_std=False), samples, [sample.reward for sample in samples]
    )

    assert advantages == pytest.approx([0.5, 0.0, -0.5, 0.0, 0.5])


@pytest.mark.unit
def test_dr_grpo_never_divides_centered_rewards_by_group_std():
    samples = [
        _sample(group=1, rollout_id=1, reward=1.0),
        _sample(group=1, rollout_id=2, reward=0.0),
        _sample(group=1, rollout_id=3, reward=0.0),
        _sample(group=1, rollout_id=4, reward=0.0),
    ]

    advantages = _normalize_rewards_per_rollout(
        _normalization_args(estimator="dr_grpo", use_std=True),
        samples,
        [sample.reward for sample in samples],
    )

    assert advantages == pytest.approx([0.75, -0.25, -0.25, -0.25])


@pytest.mark.unit
def test_reinforce_plus_plus_baseline_centers_once_per_rollout_without_local_std():
    samples = _fanout_group(3)
    args = Namespace(advantage_estimator="reinforce_plus_plus_baseline")

    advantages = _normalize_rewards_per_rollout(args, samples, [sample.reward for sample in samples])

    assert _advantages_by_rollout(samples, advantages) == pytest.approx(
        {1: 0.75, 2: -0.25, 3: -0.25, 4: -0.25}
    )


@pytest.mark.unit
def test_reward_normalization_rejects_disagreeing_sibling_rewards():
    samples = [_sample(group=1, rollout_id=7, reward=1.0), _sample(group=1, rollout_id=7, reward=0.0)]

    with pytest.raises(ValueError, match=r"rollout_id=7.*disagree on reward"):
        _normalize_rewards_per_rollout(_normalization_args(), samples, [sample.reward for sample in samples])


@pytest.mark.unit
def test_reward_normalization_requires_group_index():
    sample = _sample(group=None, reward=1.0)

    with pytest.raises(ValueError, match=r"group_index.*position.*0"):
        _normalize_rewards_per_rollout(_normalization_args(), [sample], [sample.reward])


@pytest.mark.unit
def test_reward_normalization_matches_uniform_one_sample_per_rollout_behavior():
    raw_rewards = [1.0, 0.0, 0.5, 4.0, 2.0, 3.0]
    samples = [
        _sample(group=group, reward=reward)
        for group, reward in zip([10, 10, 10, 11, 11, 11], raw_rewards, strict=True)
    ]

    actual = _normalize_rewards_per_rollout(_normalization_args(), samples, raw_rewards)

    expected = torch.tensor(raw_rewards).reshape(2, 3)
    expected = expected - expected.mean(dim=-1, keepdim=True)
    expected = expected / (expected.std(dim=-1, keepdim=True) + 1e-6)
    assert actual == pytest.approx(expected.flatten().tolist())


@pytest.mark.unit
def test_single_rollout_group_has_zero_advantage():
    sample = _sample(group=1, reward=1.0)

    assert _normalize_rewards_per_rollout(_normalization_args(), [sample], [sample.reward]) == [0.0]


def _unscored(*, group=0, rollout_id=None, reward=0.0):
    """An execution kept for a per-token objective but with no measured reward.

    Mirrors what officeqa_rl's JobBench rollout emits when the LLM judge fails
    under OPD: the trajectory still trains on the teacher KL, so it is not
    aborted, but its placeholder reward is not a measurement.
    """
    sample = _sample(group=group, rollout_id=rollout_id, reward=reward)
    sample.metadata = {"reward_available": False}
    return sample


@pytest.mark.unit
def test_unscored_execution_is_excluded_from_the_group_baseline():
    # Two measured executions at 0.2 and 0.4 -> baseline 0.3. A third execution
    # whose judge failed carries a placeholder 0.0; counting it would drop the
    # baseline to 0.2 and turn the 0.2 execution's advantage from negative to
    # exactly zero, rewarding it for a grading outage.
    samples = [
        _sample(rollout_id=0, reward=0.2),
        _sample(rollout_id=1, reward=0.4),
        _unscored(rollout_id=2),
    ]
    rewards = [0.2, 0.4, 0.0]

    normalized = _normalize_rewards_per_rollout(
        _normalization_args(use_std=False, estimator="dr_grpo"), samples, rewards
    )

    assert normalized[0] == pytest.approx(-0.1)
    assert normalized[1] == pytest.approx(0.1)
    assert normalized[2] == 0.0


@pytest.mark.unit
def test_unscored_execution_takes_exactly_zero_advantage_with_std_normalization():
    samples = [
        _sample(rollout_id=0, reward=0.0),
        _sample(rollout_id=1, reward=1.0),
        _unscored(rollout_id=2),
    ]

    normalized = _normalize_rewards_per_rollout(
        _normalization_args(use_std=True), samples, [0.0, 1.0, 0.0]
    )

    assert normalized[2] == 0.0
    assert normalized[0] < 0.0 < normalized[1]


@pytest.mark.unit
def test_unscored_siblings_do_not_shift_their_own_group():
    # Every sibling of one fan-out execution carries the same flag, and the
    # remaining measured executions must be centered exactly as if the unscored
    # execution had never been generated.
    unscored = [_unscored(rollout_id=2), _unscored(rollout_id=2)]
    samples = [_sample(rollout_id=0, reward=0.2), _sample(rollout_id=1, reward=0.4), *unscored]

    with_unscored = _normalize_rewards_per_rollout(
        _normalization_args(use_std=False, estimator="dr_grpo"), samples, [0.2, 0.4, 0.0, 0.0]
    )
    without_unscored = _normalize_rewards_per_rollout(
        _normalization_args(use_std=False, estimator="dr_grpo"), samples[:2], [0.2, 0.4]
    )

    assert with_unscored[:2] == pytest.approx(without_unscored)


@pytest.mark.unit
def test_group_with_no_measured_execution_yields_all_zero_advantages():
    samples = [_unscored(rollout_id=0), _unscored(rollout_id=1, reward=0.9)]

    normalized = _normalize_rewards_per_rollout(
        _normalization_args(use_std=True), samples, [0.0, 0.9]
    )

    assert normalized == [0.0, 0.0]
