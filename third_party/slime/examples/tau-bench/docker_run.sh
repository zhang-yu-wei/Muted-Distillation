#!/bin/bash
# Launch a command (default: interactive bash) inside the official slime image,
# with the tau-bench training environment wired up.
#
# Two environment fixes are applied at container start:
# 1. The image ships a stale apt cuDNN 9.10.2 in /usr/lib/x86_64-linux-gnu that
#    shadows the pip cuDNN 9.17.1 PyTorch/TE were built against — remove it.
# 2. NVTE_FUSED_ATTN=0: TE's cuDNN fused-attention backward leaves dk/dv
#    partially unwritten for Qwen3.5-4B's TP4 per-rank geometry
#    (4 heads / 1 GQA group / head_dim 256 / THD) — reproduced with cuDNN
#    9.10.2 AND 9.17.1 (see notes/taubench_launch_notes.md Gotcha 8). The
#    flash-attn backend (installed in this image) is clean.
#
# Usage:
#   bash docker_run.sh                                   # interactive shell
#   bash docker_run.sh bash examples/tau-bench/run_qwen3.5_4B.sh
#   REPRO_DATA=/repro/subset_short8.pt REPRO_GBS=8 ... bash docker_run.sh \
#       bash examples/tau-bench/repro_nan_rollout1.sh

set -e

# slimerl/slime:tau = slimerl/slime:latest + tau-bench's deps (litellm, termcolor,
# mistralai, google-generativeai, tenacity) installed WITH openai/pydantic/anthropic/
# numpy pinned so sglang's versions are untouched. Rebuild after pulling a new
# :latest (see notes/taubench_launch_notes.md). Override with SLIME_IMAGE.
IMAGE=${SLIME_IMAGE:-slimerl/slime:tau}
SCRATCH=${SCRATCH:-}

MOUNTS=(
   -v /mnt/slime:/mnt/slime
   -v /mnt/Qwen3.5-4B:/mnt/Qwen3.5-4B
   -v /mnt/Qwen3.5-4B_torch_dist_te:/mnt/Qwen3.5-4B_torch_dist_te
   -v /mnt/Qwen3.5-4B_slime_te:/mnt/Qwen3.5-4B_slime_te
   -v /mnt/tau-bench-data:/mnt/tau-bench-data
   -v /mnt/tau-bench:/mnt/tau-bench
   -v /mnt/Qwen3.5-4B_debug_rollout_data:/mnt/Qwen3.5-4B_debug_rollout_data
   -v /mnt/ray_temp_docker:/mnt/ray_temp
)
# Optional extra mount for replay subsets etc., exposed at /repro
if [ -n "${SCRATCH}" ]; then
   MOUNTS+=(-v "${SCRATCH}:/repro")
fi
mkdir -p /mnt/ray_temp_docker

# NOTE: do NOT select the attention backend via NVTE_* env vars — megatron
# asserts they are unset and owns the choice via --attention-backend. Pass
# REPRO_ATTN_BACKEND=flash (repro) or use --attention-backend flash (runs).
# NB: values must be expanded HERE (in the invoking shell) — `sudo docker`
# runs with a sanitized environment, so value-less `-e VAR` forwards nothing.
ENV_VARS=(
   -e MEGATRON_PATH=/root/Megatron-LM
   -e REPRO_ATTN_BACKEND="${REPRO_ATTN_BACKEND:-}"
   -e GEMINI_API_KEY="${GEMINI_API_KEY:-}"
   -e TAU_USER_MODEL_PROVIDER="${TAU_USER_MODEL_PROVIDER:-gemini}"
   -e TAU_USER_MODEL="${TAU_USER_MODEL:-gemini-3.1-flash-lite}"
   # repro knobs pass through if set
   -e REPRO_DATA="${REPRO_DATA:-}"
   -e REPRO_GBS="${REPRO_GBS:-}"
   -e REPRO_NSPP="${REPRO_NSPP:-}"
   -e REPRO_TP="${REPRO_TP:-}"
   -e REPRO_NO_SP="${REPRO_NO_SP:-}"
   -e REPRO_CHUNK="${REPRO_CHUNK:-}"
   -e REPRO_NO_ROPE_FUSION="${REPRO_NO_ROPE_FUSION:-}"
   -e REPRO_NO_COLOCATE="${REPRO_NO_COLOCATE:-}"
   -e REPRO_NO_RECOMPUTE="${REPRO_NO_RECOMPUTE:-}"
   -e SLIME_NANHUNT="${SLIME_NANHUNT:-0}"
   -e SLIME_ANOMALY="${SLIME_ANOMALY:-0}"
   -e NVTE_DEBUG="${NVTE_DEBUG:-0}"
   -e NVTE_DEBUG_LEVEL="${NVTE_DEBUG_LEVEL:-0}"
)

CMD=("$@")
if [ ${#CMD[@]} -eq 0 ]; then
   CMD=(bash)
fi

TTY_FLAGS=()
if [ -t 0 ]; then
   TTY_FLAGS=(-it)
fi

sudo docker run --rm --gpus all --ipc=host --shm-size=16g \
   --ulimit memlock=-1 --ulimit stack=67108864 \
   --network=host \
   "${MOUNTS[@]}" "${ENV_VARS[@]}" \
   -w /mnt/slime \
   "${TTY_FLAGS[@]}" "${IMAGE}" \
   bash -c 'rm -f /usr/lib/x86_64-linux-gnu/libcudnn*; ldconfig; exec "$@"' -- "${CMD[@]}"
