# TE cuDNN fused-attention backward leaves dk/dv partially unwritten (GQA hkv=1, d=256, THD)

**Status:** confirmed, standalone repro (`bugs/te_poison_test.py`), workaround deployed
(`--attention-backend flash`). Not yet filed upstream.

**Impact here:** every Qwen3.5-4B tau-bench training run at TP>1 died at the first
optimizer step with `found NaN in local grad norm for bucket #0` (notes Gotcha 8).

## Summary

`transformer_engine.pytorch.DotProductAttention` with the cuDNN FusedAttention backend
computes a **backward pass that does not write large contiguous regions of the `dk` and
`dv` output buffers** for this configuration:

| parameter | value |
|---|---|
| qkv_format | `thd` (packed varlen) |
| attn_mask_type | `padding_causal` |
| num_attention_heads | 4 |
| num_gqa_groups | **1** |
| kv_channels (head_dim) | **256** |
| dtype | bf16 |
| GPU | H100 (sm90) |

This is exactly the per-rank geometry of Qwen3.5-4B (16 Q heads, 4 KV groups, d=256)
under tensor parallel 4. `dq` is fully written; only `dk`/`dv` are affected.

TE allocates the gradient buffers with `torch.empty` (uninitialized). Whatever the CUDA
caching allocator left in the unwritten regions becomes the "gradient":

- **Reused pool memory** → stale garbage, frequently NaN when reinterpreted as bf16 →
  NaN gradients (our crashes).
- **Fresh `cudaMalloc` pages** → effectively zeros → **silently wrong (zeroed) gradients
  that pass every finiteness check**. This is arguably the worse failure mode: a
  validation run on a fresh process can look perfectly healthy while training with
  corrupted gradients.

## Magnitude

With a NaN-poisoned allocator pool, for a single microbatch of T=6656 tokens
(`cu_seqlens=[0, 6516, 6656]`, i.e. one real 6516-token sequence + a 140-token pad
segment):

```
dk: 417,792 nonfinite of 6656×1×256 = 1,703,936 elements (24.5%)
dv: 417,792 nonfinite (same count)
dq: 0
```

The same counts appear with synthetic random Q/K/V and with real training tensors dumped
from the failing run — the unwritten-region geometry is **data-independent** (which is
why the crash reproduced with any batch).

## Reproduction

30 seconds, single GPU:

```bash
# In the official image (cuDNN 9.17.1, torch 2.11.0+cu129, TE 2.16.1)
sudo docker run --rm --gpus device=0 -v /mnt/slime/bugs:/test:ro slimerl/slime:latest \
  bash -c 'rm -f /usr/lib/x86_64-linux-gnu/libcudnn*; ldconfig; python3 /test/te_poison_test.py'
# (the rm works around the image's cuDNN shadowing bug, see
#  bugs/slime_docker_cudnn_shadowing.md)
```

Expected output:

```
TP4-shard (h=4, hkv=1): nonfinite {'dq': 0, 'dk': 417792, 'dv': 417792} -> BAD
TP2-shard (h=8, hkv=2): nonfinite {'dq': 0, 'dk': 0, 'dv': 0} -> ok
TP1-full (h=16, hkv=4): nonfinite {'dq': 0, 'dk': 0, 'dv': 0} -> ok
RESULT: CUDNN-BUG-PRESENT
```

The script's key ingredient is **poisoning the allocator before every kernel call**:

```python
def poison():
    junk = [torch.full((32*1024*1024,), float("nan"), device="cuda") for _ in range(16)]
    del junk   # blocks stay in the caching pool, full of NaN; do NOT empty_cache()
```

Without this, unwritten regions read as zeros and the bug is invisible — this blind spot
is why the Gotcha-7-era isolation tests ("TE d=256 THD fwd+bwd clean") and the first
real-tensor kernel replay in this investigation both passed while training crashed.

Reproduced on **both** environments we have:

- host venv: torch 2.9.1+cu128, cuDNN **9.10.2**, TE 2.16.1
- `slimerl/slime:latest`: torch 2.11.0+cu129, cuDNN **9.17.1**, TE 2.16.1

so it is not fixed by the cuDNN 9.10→9.17 jump. Whether the defect is in the cuDNN
kernel itself or in how TE 2.16.1 sizes/launches it (e.g. workspace or grid computed for
the wrong head count) is not determined — that needs TE maintainers or a cuDNN frontend
trace (`CUDNN_LOGERR_DBG=1`).

## How this manifested in training (evidence chain, for review)

1. Crash: `found NaN in local grad norm for bucket #0 in backward pass` at the first
   train step, every run, any data (256-sample batches, 8-sample subsets, 2-sample
   pairs, shortest-6 samples — all replayed deterministically via
   `--load-debug-rollout-data`).
2. Forward provably clean: the ref log-prob pass runs the identical forward on the same
   data and produced finite values every time. Backward-only.
3. TP1 clean, TP2/TP4 NaN — the only config axis that mattered (chunked log-probs,
   sequence-parallel, rope fusion, recompute, colocate/torch_memory_saver, microbatch
   count were each toggled and exonerated; fla GDN kernels pass the same poison test —
   `bugs/fla_poison_test.py`).
4. `PYTORCH_NO_CUDA_MEMORY_CACHING=1` made the NaN vanish (fresh pages ≈ zeros);
   `CUDA_LAUNCH_BLOCKING=1` did not (not a stream race) → allocator-content-dependent
   read of unwritten memory.
5. `SLIME_NANHUNT=1` backward hooks placed the first nonfinite gradient at the
   full-attention layers (19/27/31, the every-4th non-GDN layers), arriving at
   `k_layernorm.grad_output` — i.e. carried by **dk** out of core attention. Matches
   dk/dv-only corruption.
6. Poisoned standalone replay of the dumped real q/k/v + grad_output reproduced the
   unwritten regions outside slime/megatron entirely (see Magnitude above).
7. Fix validated end-to-end: with `--attention-backend flash` (flash-attn 2.8.3 + FA3 in
   the docker image) the poison test is clean at every geometry, and both the 8-sample
   and the full 256-sample crashing batches complete the train step with correct
   metrics (pg_loss ≈ 0, ppo_kl = 0 exactly, as expected for step 0 with ratio ≡ 1).

## Open questions (honest gaps)

- **TP2**: the synthetic poison test at (h=8, hkv=2) is clean *at the single length
  tested*, yet the real TP2 replay NaN'd. Either other sequence lengths trigger the same
  kernel defect at hkv=2, or a second (unidentified) trigger exists at TP2. Not further
  characterized because TP2 is not our target config; treat TP2 + fused as unsafe too.
- The exact unwritten-region pattern (which rows/heads/strides) was not mapped; the
  417,792 count is stable across runs, suggesting a deterministic tile/stride miss
  rather than a race.
- NANHUNT also flagged "nonfinite PARAM grad" on various TE linears; those are **false
  positives**: with `gradient_accumulation_fusion`, TE returns an uninitialized dummy
  wgrad tensor (the real wgrad goes directly into `main_grad`). Don't chase them.

## Recommended actions

1. Keep `--attention-backend flash` for any model with head_dim 256 + GQA under TP
   sharding (now set in `run_qwen3.5_4B.sh`).
2. File against TransformerEngine (github.com/NVIDIA/TransformerEngine) with
   `te_poison_test.py`, the version matrix above, and the observation that it
   reproduces on cuDNN 9.10.2 and 9.17.1.
3. Any future kernel validation must poison the pool first — add this to the local
   testing playbook.
