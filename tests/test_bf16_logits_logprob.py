"""The log-prob kernel on bf16 logits straight from the model.

Megatron's Float16Module upcasts the model's [T, V] logits to fp32 before slime
sees them, and the backward materialises an fp32 gradient of the same size: at
16k tokens x 248k vocab that is 15 GiB each, and the pair is what OOM'd the
128k-context runs. slime now asks for the bf16 output and lets the kernel
upcast one chunk at a time, fold the rollout temperature in after the cast, and
recompute its fp32 softmax in backward instead of keeping it. These tests pin
the numerics of that path to the old fp32 one.
"""

from __future__ import annotations

import torch

from slime.utils.ppo_utils import _VocabParallelLogProbEntropy, calculate_log_probs_and_entropy

T, V = 37, 53


def _inputs(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    logits = (torch.randn(T, V, generator=gen) * 4).to(torch.bfloat16)
    tokens = torch.randint(0, V, (T,), generator=gen)
    keep = torch.rand(T, V, generator=gen) > 0.3
    return logits, tokens, keep


def _run(logits, tokens, keep, *, chunk_size=-1, temperature=1.0, recompute=False, with_entropy=True):
    logits = logits.clone().requires_grad_(True)
    log_prob, entropy = calculate_log_probs_and_entropy(
        logits,
        tokens,
        None,
        with_entropy=with_entropy,
        chunk_size=chunk_size,
        log_prob_keep_mask=keep,
        with_entropy_grad=True,
        temperature=temperature,
        recompute_chunks=recompute,
    )
    weights = torch.linspace(0.5, 1.5, T)
    loss = (log_prob.squeeze(-1) * weights).sum()
    if entropy is not None:
        loss = loss + (entropy * weights.flip(0)).sum()
    loss.backward()
    return log_prob.detach(), None if entropy is None else entropy.detach(), logits.grad


def test_bf16_input_matches_the_fp32_upcast_path():
    logits, tokens, keep = _inputs()
    lp16, ent16, grad16 = _run(logits, tokens, keep)
    lp32, ent32, grad32 = _run(logits.float(), tokens, keep)
    assert grad16.dtype == torch.bfloat16 and grad32.dtype == torch.float32
    torch.testing.assert_close(lp16, lp32, atol=0, rtol=0)
    torch.testing.assert_close(ent16, ent32, atol=0, rtol=0)
    # The grad is computed in fp32 and cast once to bf16 -- the same rounding
    # Float16Module's backward applied to the full fp32 gradient.
    torch.testing.assert_close(grad16, grad32.to(torch.bfloat16), atol=0, rtol=0)


def test_temperature_inside_the_kernel_equals_the_old_full_tensor_division():
    logits, tokens, keep = _inputs(1)
    temp = 0.7
    lp_new, ent_new, grad_new = _run(logits.float(), tokens, keep, temperature=temp)
    # Old path: divide the whole fp32 tensor first, temperature 1.0 in the kernel.
    scaled = (logits.float() / temp).requires_grad_(True)
    lp_old, ent_old = _VocabParallelLogProbEntropy.apply(scaled, tokens, keep, None, True, True, 1.0)
    weights = torch.linspace(0.5, 1.5, T)
    ((lp_old.squeeze(-1) * weights).sum() + (ent_old * weights.flip(0)).sum()).backward()
    torch.testing.assert_close(lp_new, lp_old.detach(), atol=0, rtol=0)
    torch.testing.assert_close(ent_new, ent_old.detach(), atol=0, rtol=0)
    torch.testing.assert_close(grad_new, scaled.grad / temp, atol=0, rtol=0)


def test_chunk_recompute_is_invisible_to_values_and_grads():
    logits, tokens, keep = _inputs(2)
    for temp in (1.0, 1.3):
        lp_a, ent_a, grad_a = _run(logits, tokens, keep, chunk_size=8, temperature=temp, recompute=False)
        lp_b, ent_b, grad_b = _run(logits, tokens, keep, chunk_size=8, temperature=temp, recompute=True)
        torch.testing.assert_close(lp_a, lp_b, atol=0, rtol=0)
        torch.testing.assert_close(ent_a, ent_b, atol=0, rtol=0)
        torch.testing.assert_close(grad_a, grad_b, atol=0, rtol=0)
        assert grad_b.dtype == torch.bfloat16 and bool(grad_b.abs().sum() > 0)


def test_recompute_frees_the_kernel_saved_softmax_until_backward():
    logits, tokens, keep = _inputs(3)
    logits = logits.clone().requires_grad_(True)
    log_prob, _ = calculate_log_probs_and_entropy(
        logits, tokens, None, with_entropy=False, chunk_size=8, log_prob_keep_mask=keep, recompute_chunks=True
    )
    # Non-reentrant checkpoint replaces the kernel's saved tensors with a
    # recompute hook; nothing [chunk, V]-sized sits in the graph before backward.
    node = log_prob.grad_fn
    saved = []
    stack = [node]
    while stack:
        n = stack.pop()
        if n is None:
            continue
        for attr in dir(n):
            if attr.startswith("_saved_") and isinstance(getattr(n, attr, None), torch.Tensor):
                saved.append(getattr(n, attr))
        for nxt, _ in n.next_functions:
            stack.append(nxt)
    assert all(t.numel() < T * V for t in saved)
    log_prob.sum().backward()
    assert logits.grad is not None and logits.grad.dtype == torch.bfloat16


def test_no_recompute_without_grad():
    logits, tokens, keep = _inputs(4)
    with torch.no_grad():
        lp, ent = calculate_log_probs_and_entropy(
            logits, tokens, None, with_entropy=True, chunk_size=8, log_prob_keep_mask=keep, recompute_chunks=True
        )
    assert lp.dtype == torch.float32 and ent.dtype == torch.float32 and not lp.requires_grad
