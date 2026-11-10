"""Acceptance test: cuDNN fused-attention THD backward must fully write dk/dv.

Poisons the CUDA caching allocator with NaNs, then runs TE DotProductAttention
fwd+bwd at the Qwen3.5-4B TP-shard geometries. Any unwritten region of the
dgrad buffers shows up as non-finite values. BAD = the cuDNN bug that NaN'd
qwen3.5 tau-bench training on the host env (cuDNN 9.10.2) is present.
"""

import torch
import transformer_engine
from transformer_engine.pytorch.attention import DotProductAttention

print(f"torch {torch.__version__} | cudnn {torch.backends.cudnn.version()} | TE {transformer_engine.__version__}")

def poison():
    junk = [torch.full((32 * 1024 * 1024,), float("nan"), device="cuda") for _ in range(16)]
    del junk  # blocks stay in the caching allocator pool, full of NaN

torch.manual_seed(0)
any_bad = False
for H, HKV, label in ((4, 1, "TP4-shard"), (8, 2, "TP2-shard"), (16, 4, "TP1-full")):
    L, pad = 6516, 140
    T = L + pad
    cu = torch.tensor([0, L, T], dtype=torch.int32, device="cuda")
    attn = DotProductAttention(num_attention_heads=H, kv_channels=256, num_gqa_groups=HKV,
                               attn_mask_type="padding_causal", qkv_format="thd",
                               softmax_scale=None, attention_dropout=0.0).cuda()
    poison()
    q = torch.randn(T, H, 256, dtype=torch.bfloat16, device="cuda").requires_grad_()
    k = torch.randn(T, HKV, 256, dtype=torch.bfloat16, device="cuda").requires_grad_()
    v = torch.randn(T, HKV, 256, dtype=torch.bfloat16, device="cuda").requires_grad_()
    poison()
    out = attn(q, k, v, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=L, max_seqlen_kv=L)
    poison()
    out.backward(torch.randn_like(out))
    bad = {n: (~torch.isfinite(t.grad)).sum().item() for n, t in (("dq", q), ("dk", k), ("dv", v))}
    status = "BAD" if any(bad.values()) else "ok"
    any_bad |= status == "BAD"
    print(f"{label} (h={H}, hkv={HKV}): nonfinite {bad} -> {status}")

print("RESULT:", "CUDNN-BUG-PRESENT" if any_bad else "clean")
