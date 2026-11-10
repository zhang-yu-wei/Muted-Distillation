#!/bin/bash

# Replay the rollout-1 batch that produced the NaN-grad crash (Gotcha 8 in
# notes/taubench_launch_notes.md) against the exact training config of
# run_qwen3.5_4B.sh, without regenerating data: --load-debug-rollout-data
# forces debug_train_only, so no sglang engines / router / Gemini are needed.
# Usage: bash repro_nan_rollout1.sh
# Optionally: REPRO_DATA=/path/to/other.pt bash repro_nan_rollout1.sh
# to replay a filtered copy of the batch (bisection).

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONUNBUFFERED=1

# Host-venv-only env plumbing; inside the slimerl/slime docker image the CUDA
# stack is already coherent and none of this applies.
if [ -n "${VIRTUAL_ENV:-}" ] && [ -d "${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia" ]; then
    CUDA_LIB_DIRS=$(find "${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia" -type d -name lib | paste -sd: -)
    export LD_LIBRARY_PATH="${CUDA_LIB_DIRS}"
fi
if [ -d /mnt/localssd/cuda-12.8 ]; then
    export CUDA_HOME=/mnt/localssd/cuda-12.8
    export CUDA_PATH=${CUDA_HOME}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export SGLANG_DISABLE_CUDNN_CHECK=1
fi

# Megatron location: host checkout by default, the image's bundled repo in docker.
MEGATRON_PATH=${MEGATRON_PATH:-/mnt/Megatron-LM-slime}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3.5-4B.sh"

# {rollout_id} is format()-ed by slime; with the default REPRO_DATA this
# resolves to the dumped 1.pt since the replay starts at rollout 1.
# NB: cannot use ${VAR:-default} here — bash ends the expansion at the first
# unquoted `}`, which turns {rollout_id}.pt into {rollout_id.pt}.
if [ -z "${REPRO_DATA:-}" ]; then
    REPRO_DATA='/mnt/Qwen3.5-4B_debug_rollout_data/{rollout_id}.pt'
fi

CKPT_ARGS=(
   --hf-checkpoint /mnt/Qwen3.5-4B/
   --ref-load /mnt/Qwen3.5-4B_torch_dist_te/
   --load /mnt/Qwen3.5-4B_slime_te/
   # no --save: this is a crash repro, never persist its optimizer state
)

# Bisection subsets carry fewer samples; the batch sizes must match the
# subset (REPRO_GBS = number of samples in the .pt, a multiple of REPRO_NSPP).
# REPRO_NSPP is the GRPO group size of the subset — sub-group bisection needs
# groups smaller than 8, with mixed rewards so advantages stay nonzero.
REPRO_GBS=${REPRO_GBS:-256}
REPRO_NSPP=${REPRO_NSPP:-8}

ROLLOUT_ARGS=(
   --prompt-data /mnt/tau-bench-data/retail_train_tasks.jsonl
   --input-key index
   --load-debug-rollout-data "${REPRO_DATA}"
   # replay starts at rollout 1 (release ckpt), so stop after that one step
   --num-rollout 2
   --rollout-batch-size $(( REPRO_GBS / REPRO_NSPP ))
   --n-samples-per-prompt "${REPRO_NSPP}"
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size "${REPRO_GBS}"
   --balance-data
)

# Config discriminators for the NaN hunt (each defaults to the crashing
# run's value): REPRO_NO_SP=1 drops --sequence-parallel, REPRO_TP overrides
# tensor parallel size, REPRO_CHUNK=-1 disables log-prob chunking (only safe
# on short subsets: unchunked fp32 log-softmax is ~2GB per 8k-token sample).
REPRO_TP=${REPRO_TP:-4}
REPRO_CHUNK=${REPRO_CHUNK:-2048}

PERF_ARGS=(
   --tensor-model-parallel-size "${REPRO_TP}"
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
   --log-probs-chunk-size "${REPRO_CHUNK}"
)
# REPRO_NO_RECOMPUTE=1: drop activation recompute (suspect: the checkpoint
# backward path; only safe on short subsets memory-wise).
if [ -z "${REPRO_NO_RECOMPUTE:-}" ]; then
   PERF_ARGS+=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1)
fi
if [ -z "${REPRO_NO_SP:-}" ]; then
   PERF_ARGS+=(--sequence-parallel)
fi
# REPRO_NO_ROPE_FUSION=1: disable the TE fused RoPE kernel (suspect: its
# backward with partial rotary dims, rotary-percent 0.25).
if [ -n "${REPRO_NO_ROPE_FUSION:-}" ]; then
   PERF_ARGS+=(--no-rope-fusion)
fi
# REPRO_NO_COLOCATE=1: drop --colocate so the torch_memory_saver
# offload/wake cycle never runs (suspect: stale TE/cuBLAS workspaces after
# page remapping).
COLOCATE_ARGS=(--colocate)
if [ -n "${REPRO_NO_COLOCATE:-}" ]; then
   COLOCATE_ARGS=()
fi

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # "fused" = TE cuDNN fused attention (only option in the host venv, and the
   # backend whose backward leaves dk/dv unwritten at TP4 geometry — Gotcha 8).
   # In the docker image pass REPRO_ATTN_BACKEND=flash.
   --attention-backend "${REPRO_ATTN_BACKEND:-fused}"
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

NUM_GPUS=8
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir /mnt/ray_temp

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}/:${SCRIPT_DIR}:/mnt/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"CUDA_HOME\": \"${CUDA_HOME}\",
    \"CUDA_PATH\": \"${CUDA_HOME}\",
    \"PATH\": \"${PATH}\",
    \"SGLANG_DISABLE_CUDNN_CHECK\": \"1\",
    \"TAU_TOOL_CALL_PARSER\": \"qwen3_coder\",
    \"GEMINI_API_KEY\": \"${GEMINI_API_KEY:-unused-in-replay}\",
    \"NVTE_DEBUG\": \"${NVTE_DEBUG:-0}\",
    \"NVTE_DEBUG_LEVEL\": \"${NVTE_DEBUG_LEVEL:-0}\",
    \"SLIME_NANHUNT\": \"${SLIME_NANHUNT:-0}\",
    \"SLIME_NANHUNT_DUMP\": \"${SLIME_NANHUNT_DUMP:-}\",
    \"SLIME_ANOMALY\": \"${SLIME_ANOMALY:-0}\",
    \"CUDA_LAUNCH_BLOCKING\": \"${REPRO_LAUNCH_BLOCKING:-0}\",
    \"PYTORCH_NO_CUDA_MEMORY_CACHING\": \"${REPRO_NO_MEM_CACHING:-0}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --rollout-num-gpus ${NUM_GPUS} \
   ${COLOCATE_ARGS[@]} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}
