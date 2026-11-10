#!/bin/bash
# Offline supervised fine-tuning for the OfficeQA Qwen3.5-4B agent.
#
# Usage (inside the slime container):
#   bash scripts/train_officeqa_sft.sh smoke filtered
#   bash scripts/train_officeqa_sft.sh full  filtered
#   bash scripts/train_officeqa_sft.sh full  all
#
# The second argument selects data/sft/officeqa_sft_v0_<policy>.jsonl. The
# filtered set is the primary training set; "all" is an ablation that also
# supervises failed, timed-out, and malformed teacher episodes.

set -euo pipefail

MODE=${1:-smoke}
DATA_POLICY=${2:-filtered}
case "${MODE}" in
smoke|full) ;;
*) echo "usage: $0 [smoke|full] [filtered|all]" >&2; exit 2 ;;
esac
case "${DATA_POLICY}" in
filtered|all) ;;
*) echo "usage: $0 [smoke|full] [filtered|all]" >&2; exit 2 ;;
esac

export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

REPO=${REPO:-/mnt/officeqa-rl}
SLIME_DIR="${REPO}/third_party/slime"
MODEL=${QWEN35_4B_MODEL:-/mnt/Qwen3.5-4B}
REF_LOAD=${REF_LOAD:-/mnt/Qwen3.5-4B_torch_dist}
SFT_DATA=${SFT_DATA:-${REPO}/data/sft/officeqa_sft_v0_${DATA_POLICY}.jsonl}
NUM_GPUS=${NUM_GPUS:-8}

# Both generated datasets contain long, highly variable agent contexts. The
# current "all" set reaches about 160k tokens, so TP=2, CP=4 and 40,960 tokens
# per CP rank cover it on eight GPUs. Override these together for other GPU
# counts or for a length-capped dataset.
TP_SIZE=${TP_SIZE:-2}
CP_SIZE=${CP_SIZE:-4}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-40960}
MODEL_PARALLEL_SIZE=$((TP_SIZE * CP_SIZE))
if [ $((NUM_GPUS % MODEL_PARALLEL_SIZE)) -ne 0 ]; then
   echo "NUM_GPUS=${NUM_GPUS} must be divisible by TP_SIZE*CP_SIZE=${MODEL_PARALLEL_SIZE}" >&2
   exit 2
fi

LR=${LR:-5e-6}
MIN_LR=${MIN_LR:-5e-7}
WARMUP_FRACTION=${WARMUP_FRACTION:-0.05}
NUM_EPOCH=${NUM_EPOCH:-1}
ROLLOUT_BS=${ROLLOUT_BS:-32}
GLOBAL_BS=${GLOBAL_BS:-32}
SAVE_INTERVAL=${SAVE_INTERVAL:-25}
TRAIN_MEM_MARGIN_BYTES=${TRAIN_MEM_MARGIN_BYTES:-1073741824}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-4096}
PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-garbage_collection_threshold:0.7}

EXP_NAME=${EXP_NAME:-qwen3.5-4B-sft-${DATA_POLICY}}
if [ "${MODE}" = "smoke" ]; then
   EXP_NAME="${EXP_NAME}-smoke"
fi
SAVE_DIR=${SAVE_DIR:-${REPO}/results/checkpoints/${EXP_NAME}}

if [ ! -f "${SFT_DATA}" ]; then
   echo "SFT dataset not found: ${SFT_DATA}" >&2
   echo "Build it first with scripts/build_sft_dataset.py (see README.md)." >&2
   exit 1
fi
if [ ! -d "${MODEL}" ]; then
   echo "Qwen3.5 HF checkpoint not found: ${MODEL}" >&2
   exit 1
fi

source "${SLIME_DIR}/scripts/models/qwen3.5-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL}"
   --ref-load "${REF_LOAD}"
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data "${SFT_DATA}"
   --input-key messages
   --metadata-key metadata
   --loss-type sft_loss
   --loss-mask-type qwen3_5
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

if [ "${MODE}" = "smoke" ]; then
   # Exactly one optimizer step, using the same model and masking path as full.
   SFT_ARGS+=(
      --num-rollout 1
      --rollout-batch-size 1
      --global-batch-size 1
      --save-debug-train-data "${REPO}/results/debug_train/${EXP_NAME}_{rollout_id}_{rank}.pt"
   )
else
   CKPT_ARGS+=(
      --load "${SAVE_DIR}"
      --save "${SAVE_DIR}"
      --save-interval "${SAVE_INTERVAL}"
   )
   SFT_ARGS+=(
      --num-epoch "${NUM_EPOCH}"
      --rollout-batch-size "${ROLLOUT_BS}"
      --global-batch-size "${GLOBAL_BS}"
      --rollout-shuffle
   )
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE}"
   --context-parallel-size "${CP_SIZE}"
   --pipeline-model-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --balance-data
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --train-memory-margin-bytes "${TRAIN_MEM_MARGIN_BYTES}"
   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
   --train-env-vars "{\"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_ALLOC_CONF}\"}"
)
if [ "${TP_SIZE}" -gt 1 ]; then
   PERF_ARGS+=(--sequence-parallel)
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR}"
   --lr-decay-style cosine
   --min-lr "${MIN_LR}"
   --lr-warmup-fraction "${WARMUP_FRACTION}"
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# Match the GRPO launcher's tracking convention. Disable with
# OFFICEQA_WANDB=0 or use WANDB_MODE=offline.
OFFICEQA_WANDB=${OFFICEQA_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-officeqa-rl}
WANDB_GROUP=${WANDB_GROUP:-${EXP_NAME}}
WANDB_ARGS=()
if [ "${OFFICEQA_WANDB}" = "1" ]; then
   WANDB_ARGS+=(
      --use-wandb
      --wandb-project "${WANDB_PROJECT}"
      --wandb-group "${WANDB_GROUP}"
   )
   if [ -n "${WANDB_MODE:-}" ]; then
      WANDB_ARGS+=(--wandb-mode "${WANDB_MODE}")
   fi
   if [ -n "${WANDB_API_KEY:-}" ]; then
      WANDB_ARGS+=(--wandb-key "${WANDB_API_KEY}")
   elif [ "${SFT_DRY_RUN:-0}" != "1" ] && \
        [ "${WANDB_MODE:-online}" = "online" ] && \
        ! python3 -c "import wandb, sys; sys.exit(0 if wandb.api.api_key else 1)" 2>/dev/null; then
      echo "W&B is enabled but no credentials were found." >&2
      echo "Set WANDB_API_KEY, WANDB_MODE=offline, or OFFICEQA_WANDB=0." >&2
      exit 1
   fi
fi

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "${NVLINK_COUNT}" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
export no_proxy="127.0.0.1,${MASTER_ADDR}"

RUNTIME_ENV_JSON=$(printf '%s' \
   "{\"env_vars\":{\"PYTHONPATH\":\"/root/Megatron-LM:${REPO}/src:${SLIME_DIR}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"${HAS_NVLINK}\",\"no_proxy\":\"${no_proxy}\",\"MASTER_ADDR\":\"${MASTER_ADDR}\"}}")

TRAIN_CMD=(
   python3 train_async.py
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${NUM_GPUS}"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${SFT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${MISC_ARGS[@]}"
)

echo "SFT mode=${MODE} policy=${DATA_POLICY} data=${SFT_DATA}"
echo "GPUs=${NUM_GPUS} TP=${TP_SIZE} CP=${CP_SIZE} max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}"

# Print the fully expanded training command without touching Ray or GPUs.
if [ "${SFT_DRY_RUN:-0}" = "1" ]; then
   printf 'cd %q && ' "${SLIME_DIR}"
   printf '%q ' "${TRAIN_CMD[@]}"
   printf '\n'
   exit 0
fi

# Slime trains from a Megatron torch_dist checkpoint. Match the GRPO launcher
# by creating it automatically on first use. Keep this after dry-run so merely
# inspecting the command never writes a checkpoint.
if [ ! -d "${REF_LOAD}" ]; then
   echo "Converting ${MODEL} -> ${REF_LOAD} (torch_dist)"
   PYTHONPATH="/root/Megatron-LM:${SLIME_DIR}" \
   python3 "${SLIME_DIR}/tools/convert_hf_to_torch_dist_qwen3_5.py" \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint "${MODEL}" \
      --save "${REF_LOAD}"
fi
if [ "${MODE}" = "smoke" ]; then
   mkdir -p "${REPO}/results/debug_train"
else
   mkdir -p "${SAVE_DIR}"
fi

ray stop --force || true
pkill -9 sglang || true
sleep 2
ray start --head \
   --node-ip-address "${MASTER_ADDR}" \
   --num-gpus "${NUM_GPUS}" \
   --disable-usage-stats \
   --dashboard-host=0.0.0.0 \
   --dashboard-port=8265

cd "${SLIME_DIR}"
ray job submit \
   --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "${TRAIN_CMD[@]}"
