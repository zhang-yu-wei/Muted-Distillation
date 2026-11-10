"""Poisoned-pool test of the fla GDN stack at Qwen3.5-4B geometry.

Replicates Qwen3_5GatedDeltaNet.forward's kernel calls (ShortConvolution,
chunk_gated_delta_rule with qk l2norm, FusedRMSNormGated) on varlen input
including slime's trailing pad segment, with a NaN-poisoned CUDA caching
allocator. Any output/dgrad region a kernel fails to write shows as nonfinite.
"""

import torch
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

NUM_K, NUM_V, DK, DV = 16, 32, 128, 128

def poison():
    junk = [torch.full((32 * 1024 * 1024,), float("nan"), device="cuda") for _ in range(16)]
    del junk

def report(name, t):
    n = (~torch.isfinite(t)).sum().item()
    print(f"  {name}: shape={tuple(t.shape)} nonfinite={n}", flush=True)
    return n

torch.manual_seed(0)
total_bad = 0
for L, pad in [(6516, 140), (4805, 315), (6697, 471)]:
    T = L + pad
    cu = torch.tensor([0, L, T], dtype=torch.int32, device="cuda")
    print(f"--- L={L} pad={pad} cu={cu.tolist()}")

    conv_dim = NUM_K * DK * 2 + NUM_V * DV
    conv = ShortConvolution(hidden_size=conv_dim, kernel_size=4, bias=False).cuda().to(torch.bfloat16)
    norm = FusedRMSNormGated(DV, eps=1e-6, activation="silu").cuda().to(torch.bfloat16)

    poison()
    mixed_qkv = torch.randn(1, T, conv_dim, dtype=torch.bfloat16, device="cuda").requires_grad_()
    poison()
    conv_out, _ = conv(x=mixed_qkv, cu_seqlens=cu)

    q, k, v = torch.split(conv_out, [NUM_K * DK, NUM_K * DK, NUM_V * DV], dim=-1)
    q = q.reshape(1, T, NUM_K, DK).repeat_interleave(NUM_V // NUM_K, dim=2)
    k = k.reshape(1, T, NUM_K, DK).repeat_interleave(NUM_V // NUM_K, dim=2)
    v = v.reshape(1, T, NUM_V, DV)

    poison()
    b = torch.randn(1, T, NUM_V, dtype=torch.bfloat16, device="cuda").requires_grad_()
    a = torch.randn(1, T, NUM_V, dtype=torch.bfloat16, device="cuda").requires_grad_()
    beta = b.sigmoid()
    g = -torch.rand(NUM_V, device="cuda").exp().log1p() * torch.nn.functional.softplus(a.float() + 1.0)

    poison()
    core, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None,
                                     output_final_state=False, use_qk_l2norm_in_kernel=True,
                                     cu_seqlens=cu)
    fwd_bad = report("fwd core_attn_out", core)

    poison()
    z = torch.randn(1, T, NUM_V, DV, dtype=torch.bfloat16, device="cuda").requires_grad_()
    out = norm(core.reshape(-1, DV), z.reshape(-1, DV))
    fwd_bad += report("fwd norm_out", out)

    poison()
    out.backward(torch.randn_like(out))
    bwd_bad = report("d mixed_qkv", mixed_qkv.grad)
    bwd_bad += report("d b", b.grad)
    bwd_bad += report("d a", a.grad)
    bwd_bad += report("d z", z.grad)
    total_bad += fwd_bad + bwd_bad

print("RESULT:", "NONFINITE-REPRODUCED" if total_bad else "clean")
