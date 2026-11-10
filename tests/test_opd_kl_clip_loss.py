"""Symmetric clip on the per-token reverse KL (officeqa_opd_kl_clip).

``k_hat = clamp(k, -clip, +clip)`` bounds every token's advantage by
``opd_kl_coef * clip``. The clip acts on the RAW student-minus-teacher
difference, before the no-action marker substitution, so that mechanism's
calibrated magnitude is never capped; the logged ``opd_reverse_kl`` stays
unclipped so it remains comparable across runs; and each side's bind rate is
exported as a 0/1 per-token mask. With the knob at 0 nothing changes.
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

# k = student - teacher:  [-0.4, +5.0, 0.0, -3.0, +0.5]
STUDENT = torch.tensor([-0.5, -1.0, -0.1, -4.0, -0.5])
TEACHER = torch.tensor([-0.1, -6.0, -0.1, -1.0, -1.0])
K = STUDENT - TEACHER


def _args(**kwargs):
    fields = dict(
        opd_type="sglang",
        opd_kl_coef=1.0,
        officeqa_opd_pure=True,
        officeqa_opd_contrast=False,
        officeqa_opd_noaction_penalty=0.0,
        officeqa_opd_kl_clip=0.0,
    )
    fields.update(kwargs)
    return Namespace(**fields)


def _run(args, teacher=None, base=None):
    teacher = TEACHER.clone() if teacher is None else teacher
    advantages = [torch.zeros(5) if base is None else base.clone()]
    data = {"teacher_log_probs": [teacher]}
    apply_opd_kl_to_advantages(args, data, advantages, [STUDENT.clone()])
    return advantages[0], data


def test_knob_off_is_the_previous_behaviour():
    adv, data = _run(_args())
    assert torch.allclose(data["opd_reverse_kl"][0], K)
    assert torch.allclose(adv, -K)
    assert "opd_kl_clip_hi" not in data and "opd_kl_clip_lo" not in data


def test_clip_bounds_the_advantage_on_both_sides():
    adv, data = _run(_args(officeqa_opd_kl_clip=1.6))
    expected = -torch.clamp(K, -1.6, 1.6)
    assert torch.allclose(adv, expected)
    assert adv.abs().max() <= 1.6 + 1e-6
    # Tokens inside the band are untouched bit-for-bit.
    inside = K.abs() <= 1.6
    assert torch.equal(adv[inside], (-K)[inside])


def test_logged_reverse_kl_stays_unclipped_and_bind_masks_are_exported():
    _, data = _run(_args(officeqa_opd_kl_clip=1.6))
    assert torch.allclose(data["opd_reverse_kl"][0], K)
    assert torch.equal(data["opd_kl_clip_hi"][0], torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(data["opd_kl_clip_lo"][0], torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]))


def test_clip_scales_with_coefficient_and_keeps_base_advantage():
    base = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    adv, _ = _run(_args(officeqa_opd_pure=False, opd_kl_coef=0.5, officeqa_opd_kl_clip=2.0), base=base)
    assert torch.allclose(adv, base - 0.5 * torch.clamp(K, -2.0, 2.0))


def test_noaction_markers_are_never_capped():
    """A positive marker in teacher_log_probs is a calibrated penalty, not a
    log-prob; the clip must not reach it, in the advantage or in the log."""
    teacher = TEACHER.clone()
    teacher[1] = 4.0  # marker larger than the clip
    adv, data = _run(_args(officeqa_opd_kl_clip=1.6, officeqa_opd_noaction_penalty=1.0), teacher=teacher)
    assert adv[1] == pytest.approx(-4.0)
    assert data["opd_reverse_kl"][0][1] == pytest.approx(4.0)
    # The other tokens are still clipped as usual.
    assert adv[3] == pytest.approx(1.6)


def test_residual_mode_trains_only_on_what_the_clip_removes():
    """Diagnostic mode: advantage = -(k - clamp(k)); zero inside the band,
    and clip + residual reconstruct the unclipped advantage exactly."""
    adv_clip, _ = _run(_args(officeqa_opd_kl_clip=1.6))
    adv_res, data = _run(_args(officeqa_opd_kl_clip=1.6, officeqa_opd_kl_clip_residual=True))
    inside = K.abs() <= 1.6
    assert torch.all(adv_res[inside] == 0)
    assert torch.allclose(adv_clip + adv_res, -K)
    # Logged value is still the raw reverse KL.
    assert torch.allclose(data["opd_reverse_kl"][0], K)
