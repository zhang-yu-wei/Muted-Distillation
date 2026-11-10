"""Dynamic-sampling filters for the OfficeQA fan-out rollout.

slime's stock ``check_reward_nonzero_std`` reads ``get_reward_value`` off
every element of the prompt group, assuming one flat ``Sample`` per rollout
execution. Our rollout hook returns ``Sample | list[Sample]`` per execution
(one sibling fragment per contiguous token chain), so a group element may be
a list — the stock filter would crash on it, and iterating siblings would
enter each execution's reward once per fragment. This filter takes exactly
one reward per execution before testing for within-group contrast.
"""

import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput


def check_execution_reward_nonzero_std(args, samples, **kwargs) -> DynamicFilterOutput:
    """Keep a prompt group only if its rollout executions disagree on reward.

    ``samples`` is one prompt group with one entry per rollout execution,
    each a ``Sample`` or a non-empty ``list[Sample]`` of sibling fragments
    that share one reward (slime's flattener validates the sharing later).
    A group whose executions all scored the same has zero GRPO advantage on
    every sample, so training on it only spends compute.
    """
    rewards = []
    for execution in samples:
        first = execution[0] if isinstance(execution, list) else execution
        rewards.append(first.get_reward_value(args))
    # A single execution has no within-group contrast (and a singleton sample
    # standard deviation is undefined).
    keep = len(rewards) > 1 and bool(torch.tensor(rewards, dtype=torch.float64).std() > 1e-6)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
