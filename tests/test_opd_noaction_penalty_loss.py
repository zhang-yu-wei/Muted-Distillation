"""Train-side half of the no-action penalty: marker -> fixed advantage.

``_apply_action_scores`` writes the penalty magnitude into
``teacher_log_probs`` as a strictly POSITIVE value, impossible for a real
log-prob. :func:`apply_opd_kl_to_advantages` must read such a token as a
reverse KL of exactly that magnitude — giving it the constant advantage
``-opd_kl_coef * penalty`` regardless of the student's log-prob — and must be
a no-op when the knob is off.
"""
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SLIME = Path(__file__).resolve().parents[1] / "third_party" / "slime"
if str(SLIME) not in sys.path:
    sys.path.insert(0, str(SLIME))
if "megatron" not in sys.modules:  # the function itself needs no Megatron
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    core.mpu = types.SimpleNamespace(get_context_parallel_world_size=lambda: 1)
    megatron.core = core
    sys.modules["megatron"] = megatron
    sys.modules["megatron.core"] = core

apply_opd_kl_to_advantages = pytest.importorskip(
    "slime.backends.megatron_utils.loss"
).apply_opd_kl_to_advantages

STUDENT = torch.tensor([-0.5, -2.0, -0.1, -0.5, -3.0])
TEACHER = torch.tensor([-0.9, -1.0, -0.1, +1.0, +1.0])


def _args(**kwargs):
    fields = dict(
        opd_type="sglang",
        opd_kl_coef=1.0,
        officeqa_opd_pure=True,
        officeqa_opd_contrast=False,
        officeqa_opd_noaction_penalty=0.0,
    )
    fields.update(kwargs)
    return Namespace(**fields)


def _run(args, marker=1.0):
    """The knob only GATES the branch; the magnitude rides in the marker."""
    teacher = TEACHER.clone()
    teacher[3:] = marker
    advantages = [torch.zeros(5)]
    data = {"teacher_log_probs": [teacher]}
    apply_opd_kl_to_advantages(args, data, advantages, [STUDENT.clone()])
    return advantages[0], data["opd_reverse_kl"][0]


def test_marked_tokens_take_a_constant_advantage():
    advantages, reverse_kl = _run(_args(officeqa_opd_noaction_penalty=1.0))
    # Unmarked tokens keep the ordinary reverse KL.
    assert torch.allclose(reverse_kl[:3], STUDENT[:3] - TEACHER[:3])
    assert torch.allclose(advantages[:3], -(STUDENT[:3] - TEACHER[:3]))
    # Marked tokens: advantage is exactly -coef * penalty and does NOT depend
    # on the student log-prob, though these two differ sharply.
    assert torch.allclose(reverse_kl[3:], torch.tensor([1.0, 1.0]))
    assert torch.allclose(advantages[3:], torch.tensor([-1.0, -1.0]))


def test_penalty_scales_with_coefficient_and_marker():
    args = _args(opd_kl_coef=0.5, officeqa_opd_noaction_penalty=0.25)
    advantages, reverse_kl = _run(args, marker=0.25)
    assert torch.allclose(reverse_kl[3:], torch.tensor([0.25, 0.25]))
    assert torch.allclose(advantages[3:], torch.tensor([-0.125, -0.125]))


def test_knob_off_is_the_previous_behaviour():
    """With the knob off the branch must not run, proving no existing arm can
    be altered by this change."""
    advantages, reverse_kl = _run(_args())
    assert torch.allclose(reverse_kl, STUDENT - TEACHER)
    assert torch.allclose(advantages, -(STUDENT - TEACHER))
