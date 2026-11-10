#!/bin/bash

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

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1

# Host-venv-only env plumbing (LD_LIBRARY_PATH/CUDA_HOME juggling, sglang
# cudnn-check skip). Inside the slimerl/slime docker image (the supported way
# to run this — see docker_run.sh) the CUDA stack is coherent and none of
# this applies.
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

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3.5-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint /mnt/Qwen3.5-4B/
   --ref-load /mnt/Qwen3.5-4B_torch_dist_te/
   --load /mnt/Qwen3.5-4B_slime_te/
   --save /mnt/Qwen3.5-4B_slime_te/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/tau-bench-data/retail_train_tasks.jsonl
   --input-key index
   --rollout-shuffle
   --num-rollout 500
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size 256
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --balance-data
   # Post-mortem for the rollout-1 NaN-grad crash (see Gotcha 8 in
   # notes/taubench_launch_notes.md): dump each rollout so a train-step crash
   # can be replayed offline via --load-debug-rollout-data instead of paying
   # another ~50 min generation. A few MB per rollout.
   --save-debug-rollout-data /mnt/Qwen3.5-4B_debug_rollout_data/{rollout_id}.pt
)

EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data retail-dev /mnt/tau-bench-data/retail_dev_tasks.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 1024
   --eval-top-k 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
   # Qwen3.5's 248320 vocab makes unchunked fp32 log-softmax buffers huge
   # (~9-10GB at TP2 for a 12k-token sample); torch_memory_saver then rejects
   # the alloc to protect its memory margin and backward silently NaNs.
   # Chunking caps the peak regardless of sample length.
   --log-probs-chunk-size 2048
)

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

TB_ARGS=(
   --use-tensorboard
   --tb-project-name slime-tau-bench
   --tb-experiment-name qwen3.5-4B
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   # If gemini API reports concurrency limit error, set this parameter to reduce the concurrency
   # --sglang-server-concurrency 32
   # Tau-bench trajectories synchronize waiting on Gemini user-sim turns, then
   # reconnect to the engines all at once; the connect burst trips the router's
   # circuit breaker, which then fast-fails healthy workers and spams
   # "Success recorded while circuit is open". The failures are transient (the
   # engines are never down) and slime's http_utils retry already handles them,
   # so the breaker only adds churn — same rationale as slime's forced disable
   # in PD-disaggregation mode (slime/ray/rollout.py).
   --router-disable-circuit-breaker
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   # NEVER use "fused" (TE cuDNN) for this model at TP>1: its backward leaves
   # dk/dv partially unwritten at the per-rank geometry (4 heads / 1 GQA group
   # / head_dim 256 / THD), which surfaces as NaN grads — see Gotcha 8 in
   # notes/taubench_launch_notes.md. Reproduced with cuDNN 9.10.2 and 9.17.1.
   # "flash" requires flash-attn, which the slimerl/slime docker image ships.
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_tau.generate
)
# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# If you want more or less GPUs, change this parameter
NUM_GPUS=8
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir /mnt/ray_temp

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}/:${SCRIPT_DIR}:/mnt/slime:/mnt/tau-bench\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"CUDA_HOME\": \"${CUDA_HOME}\",
    \"CUDA_PATH\": \"${CUDA_HOME}\",
    \"PATH\": \"${PATH}\",
    \"SGLANG_DISABLE_CUDNN_CHECK\": \"1\",
    \"TAU_TOOL_CALL_PARSER\": \"qwen3_coder\",
    \"TAU_USER_MODEL_PROVIDER\": \"${TAU_USER_MODEL_PROVIDER:-gemini}\",
    \"TAU_USER_MODEL\": \"${TAU_USER_MODEL:-gemini-3.1-flash-lite}\",
    \"GEMINI_API_KEY\": \"${GEMINI_API_KEY:-}\",
    \"DEEPSEEK_API_KEY\": \"${DEEPSEEK_API_KEY:-}\",
    \"TENSORBOARD_DIR\": \"${TENSORBOARD_DIR:-/mnt/slime/tensorboard_log/slime-tau-bench/qwen3.5-4B}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --rollout-num-gpus ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${TB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
