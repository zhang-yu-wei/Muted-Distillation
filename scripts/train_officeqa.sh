#!/bin/bash
# OfficeQA agent GRPO launcher (Tasks 10, 11, and the Phase 5 baseline).
#
# Structure copied from third_party/slime/scripts/run-qwen3.5-27B.sh, reduced
# to a single node; the rollout/data section drives the OfficeQA custom
# generate hook. Run inside the slime training container with the repository
# visible at /mnt/officeqa-rl and the host Docker socket + CLI mounted (the
# sandboxes are created DooD).
#
# Modes:
#   bash scripts/train_officeqa.sh rollout-debug   # Task 10: 1 rollout, no training
#   bash scripts/train_officeqa.sh train-smoke     # Task 11: exactly 1 GRPO iteration
#   bash scripts/train_officeqa.sh full            # GRPO baseline with periodic eval
#   bash scripts/train_officeqa.sh full-async      # fully-async rollout, disjoint GPUs
#
# The full modes take key hyperparameters from the environment (defaults in
# parentheses): LR (1e-6), ROLLOUT_BS (8), N_SAMPLES (8), GLOBAL_BS
# (ROLLOUT_BS*N_SAMPLES), NUM_EPOCH (10), MAX_CTX_LEN (65536), MAX_RESP_LEN
# (8192, per-turn; also used for eval), plus the loss-shape knobs
# ADV_ESTIMATOR (grpo), TEMPERATURE (1.0), KL_LOSS_COEF (0.00, nonzero turns
# on --use-kl-loss), EPS_CLIP (0.2), EPS_CLIP_HIGH (unset), ENTROPY_COEF
# (0.00), DYNAMIC_FILTER (0; 1 enables DAPO-style dynamic sampling, full mode
# only), and OVER_SAMPLING_BS (unset = ROLLOUT_BS, the dynamic-sampling
# refill granularity). Training parallelism: TP_SIZE (2) and CP_SIZE (2); CP=2 is what
# lets a single 64k-token sample fit the 32768 max-tokens-per-gpu budget.
# The experiment name starts from "qwen3.5-4B-<estimator>-<mode>" and only
# appends knobs that differ from the baseline defaults (RESD-style _add), so
# the baseline stays short and every deviation is visible. It names both the
# W&B group and the checkpoint directory; override with EXP_NAME.
#
# AGENT_STEP_LIMIT (any mode, default 50) caps the agent's steps per episode.
# It overrides officeqa_step_limit from configs/training/officeqa_rollout.yaml
# by writing a derived config under results/ — the yaml is applied on top of
# CLI args by slime, so a flag alone cannot override it. Set it empty to use
# the yaml value as-is.
#
# full-async uses train_async.py + slime's fully-async rollout worker: actor
# and SGLang live on disjoint GPUs (ACTOR_GPUS, default 4, rest for rollout)
# with no colocation/offload, and there is no periodic eval (the fully-async
# worker does not support it).
#
# The train modes need the Megatron torch_dist checkpoint; if REF_LOAD does not
# exist yet it is converted from the HF checkpoint automatically.

set -ex

MODE=${1:-rollout-debug}
case "${MODE}" in rollout-debug|train-smoke|full|full-async) ;; *)
   echo "usage: $0 [rollout-debug|train-smoke|full|full-async]" >&2; exit 1 ;;
esac

export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

REPO=${REPO:-/mnt/officeqa-rl}

SLIME_DIR="${REPO}/third_party/slime"
MODEL=${QWEN35_4B_MODEL:-/mnt/Qwen3.5-4B}
REF_LOAD=${REF_LOAD:-/mnt/Qwen3.5-4B_torch_dist}
if [ "${MODE}" = "rollout-debug" ]; then
   NUM_GPUS=${NUM_GPUS:-1}
else
   NUM_GPUS=${NUM_GPUS:-8}
fi

# Key hyperparameters (full modes; the smoke modes hard-code theirs). The
# *_DEF values are the baseline; the experiment name only records knobs that
# differ from them.
LR_DEF=1e-6;             LR=${LR:-${LR_DEF}}
ROLLOUT_BS_DEF=8;        ROLLOUT_BS=${ROLLOUT_BS:-${ROLLOUT_BS_DEF}}
N_SAMPLES_DEF=8;         N_SAMPLES=${N_SAMPLES:-${N_SAMPLES_DEF}}
GLOBAL_BS_DEF=$((ROLLOUT_BS * N_SAMPLES)); GLOBAL_BS=${GLOBAL_BS:-${GLOBAL_BS_DEF}}
NUM_EPOCH_DEF=10;        NUM_EPOCH=${NUM_EPOCH:-${NUM_EPOCH_DEF}}
MAX_CTX_LEN_DEF=65536;   MAX_CTX_LEN=${MAX_CTX_LEN:-${MAX_CTX_LEN_DEF}}
MAX_RESP_LEN_DEF=8192;   MAX_RESP_LEN=${MAX_RESP_LEN:-${MAX_RESP_LEN_DEF}}
# Per-episode agent step cap; explicitly empty (AGENT_STEP_LIMIT=) = yaml value.
AGENT_STEP_LIMIT_DEF=50; AGENT_STEP_LIMIT=${AGENT_STEP_LIMIT-${AGENT_STEP_LIMIT_DEF}}

# Loss shape / exploration knobs (all modes).
ADV_ESTIMATOR_DEF=grpo;  ADV_ESTIMATOR=${ADV_ESTIMATOR:-${ADV_ESTIMATOR_DEF}}  # grpo | gspo | cispo | ...
TEMPERATURE_DEF=1.0;     TEMPERATURE=${TEMPERATURE:-${TEMPERATURE_DEF}}
KL_LOSS_COEF_DEF=0.00;   KL_LOSS_COEF=${KL_LOSS_COEF:-${KL_LOSS_COEF_DEF}}     # nonzero enables --use-kl-loss
EPS_CLIP_DEF=0.2;        EPS_CLIP=${EPS_CLIP:-${EPS_CLIP_DEF}}
EPS_CLIP_HIGH=${EPS_CLIP_HIGH:-}           # e.g. 0.28 for DAPO clip-higher; empty = symmetric
ENTROPY_COEF_DEF=0.00;   ENTROPY_COEF=${ENTROPY_COEF:-${ENTROPY_COEF_DEF}}

# DAPO-style dynamic sampling (full mode only; the fully-async rollout worker
# ignores the filter). DYNAMIC_FILTER=1 drops prompt groups whose executions
# all share one reward — zero GRPO advantage on every sample — and keeps
# sampling replacement prompts until the batch holds ROLLOUT_BS informative
# groups. num_rollout stays num_epoch*dataset//rollout_batch_size, so the run
# does the same number of optimizer steps and simply cycles the dataset more
# often. OVER_SAMPLING_BS (empty = ROLLOUT_BS) sets the refill granularity.
DYNAMIC_FILTER_DEF=0;    DYNAMIC_FILTER=${DYNAMIC_FILTER:-${DYNAMIC_FILTER_DEF}}
OVER_SAMPLING_BS=${OVER_SAMPLING_BS:-}

# Training parallelism. ctx 64k needs CP=2: the dynamic batcher packs up to
# max-tokens-per-gpu * CP tokens per microbatch, so a single 64k-token sample
# only fits the 32768 budget when split across 2 CP ranks.
if [ "${MODE}" = "rollout-debug" ]; then
   TP_SIZE=${TP_SIZE:-1}
   CP_SIZE=${CP_SIZE:-1}
else
   TP_SIZE=${TP_SIZE:-2}
   CP_SIZE=${CP_SIZE:-2}
fi

# Raw-CUDA headroom the torch allocator must leave free (slime default,
# 1 GiB). Don't raise it: on the worst-case microbatch (a ~66k-token sample,
# 33k tokens/CP rank) the backward needs a single 15.28 GiB fp32 logits-grad
# block with ~17.1 GiB free once optimizer state exists — a 2 GiB margin
# missed that by 184 MB (2026-07-15 00:14 run, rollout 1), and 4 GiB killed
# the 23:44 run. The Triton-launch memory the margin once protected is
# handled by the persistent kernel cache instead (scripts/prewarm_fla_kernels.py).
TRAIN_MEM_MARGIN_BYTES=${TRAIN_MEM_MARGIN_BYTES:-1073741824}

# Compute loss log-probs/softmax over token chunks instead of the whole
# microbatch: the fp32 transients for a 32k-token CP shard of the 248320
# vocab are ~15 GiB unchunked (the 23:44 OOM), ~1.9 GiB at 4096.
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-4096}

# Train-actor allocator config (train actors only; SGLang unaffected).
# garbage_collection_threshold makes the torch allocator free cached blocks
# proactively once reserved memory passes that fraction of the card, instead
# of only when an allocation hits the wall: the 04:39 run died asking for the
# recurring 15.16 GiB fp32 logits block with 16.13 GiB cached-but-fragmented.
# expandable_segments is NOT safe here — it bypasses the cudaMalloc hook that
# torch_memory_saver's colocate offload relies on.
PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-garbage_collection_threshold:0.7}

nonzero() { [ "$(awk "BEGIN{print ($1 != 0) ? 1 : 0}")" = "1" ]; }

# _add <tag> <value> [<default>]: append -<tag><value> to EXP_NAME unless the
# value is empty or equals its default. The baseline run therefore gets the
# short name "qwen3.5-4B-grpo-<mode>" and only deviations from it are recorded.
_add() {
   local tag=$1 val=$2 def=${3:-}
   if [ -n "${val}" ] && [ "${val}" != "${def}" ]; then
      EXP_NAME+="-${tag}${val}"
   fi
}

# Experiment name encodes the knobs that deviate from the baseline; it names
# both the W&B group and the checkpoint directory so runs never overwrite
# each other.
if [ -z "${EXP_NAME:-}" ]; then
   EXP_NAME="qwen3.5-4B-fanoutfix-${ADV_ESTIMATOR}-${MODE}"
   case "${MODE}" in
   full|full-async)
      _add lr   "${LR}"               "${LR_DEF}"
      _add rbs  "${ROLLOUT_BS}"       "${ROLLOUT_BS_DEF}"
      _add n    "${N_SAMPLES}"        "${N_SAMPLES_DEF}"
      _add gbs  "${GLOBAL_BS}"        "${GLOBAL_BS_DEF}"
      _add ctx  "${MAX_CTX_LEN}"      "${MAX_CTX_LEN_DEF}"
      _add resp "${MAX_RESP_LEN}"     "${MAX_RESP_LEN_DEF}"
      _add t    "${TEMPERATURE}"      "${TEMPERATURE_DEF}"
      _add kl   "${KL_LOSS_COEF}"     "${KL_LOSS_COEF_DEF}"
      _add clip "${EPS_CLIP}"         "${EPS_CLIP_DEF}"
      _add clih "${EPS_CLIP_HIGH}"
      _add ent  "${ENTROPY_COEF}"     "${ENTROPY_COEF_DEF}"
      _add dyn  "${DYNAMIC_FILTER}"   "${DYNAMIC_FILTER_DEF}"
      _add osbs "${OVER_SAMPLING_BS}"
      _add ep   "${NUM_EPOCH}"        "${NUM_EPOCH_DEF}"
      _add sl   "${AGENT_STEP_LIMIT}" "${AGENT_STEP_LIMIT_DEF}"
      ;;
   esac
fi

# The rollout hook reads its settings from the custom config yaml; slime
# applies that yaml on top of the CLI args, so overriding a single key means
# generating a derived config.
CUSTOM_CONFIG="${REPO}/configs/training/officeqa_rollout.yaml"
if [ -n "${AGENT_STEP_LIMIT}" ]; then
   DERIVED_CONFIG="${REPO}/results/officeqa_rollout_sl${AGENT_STEP_LIMIT}.yaml"
   mkdir -p "${REPO}/results"
   { grep -v '^officeqa_step_limit:' "${CUSTOM_CONFIG}"
     echo "officeqa_step_limit: ${AGENT_STEP_LIMIT}"; } > "${DERIVED_CONFIG}"
   CUSTOM_CONFIG="${DERIVED_CONFIG}"
fi

# rerun hygiene (single node)
ray stop --force || true
pkill -9 sglang || true
sleep 2

source "${SLIME_DIR}/scripts/models/qwen3.5-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL}"
)

ROLLOUT_ARGS=(
   --prompt-data "${REPO}/data/slime/officeqa_full_minus_pro_train.jsonl"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --custom-generate-function-path officeqa_rl.training.rollout.generate
   --custom-config-path "${CUSTOM_CONFIG}"
   # Additive W&B metrics splitting truncated_ratio by cause (step limit vs
   # context vs per-turn cap), sample- and episode-weighted; see
   # officeqa_rl/training/metrics.py. slime's default metrics still log.
   --custom-rollout-log-function-path officeqa_rl.training.metrics.log_rollout_extra_metrics
   --custom-eval-rollout-log-function-path officeqa_rl.training.metrics.log_eval_extra_metrics
   --rollout-max-context-len "${MAX_CTX_LEN}"
   --rollout-max-response-len "${MAX_RESP_LEN}"
   --rollout-temperature "${TEMPERATURE}"
)

EVAL_ARGS=()

if [ "${MODE}" != "rollout-debug" ]; then
   # Train modes need the Megatron-format reference/initial checkpoint.
   if [ ! -d "${REF_LOAD}" ]; then
      echo "Converting ${MODEL} -> ${REF_LOAD} (torch_dist)"
      PYTHONPATH="/root/Megatron-LM:${SLIME_DIR}" \
      python3 "${SLIME_DIR}/tools/convert_hf_to_torch_dist_qwen3_5.py" \
         "${MODEL_ARGS[@]}" \
         --hf-checkpoint "${MODEL}" \
         --save "${REF_LOAD}"
   fi
   CKPT_ARGS+=(--ref-load "${REF_LOAD}")
fi

# Checkpoint destination, named after the experiment (SAVE_DIR overrides).
SAVE_DIR=${SAVE_DIR:-${REPO}/results/checkpoints/${EXP_NAME}}

# Rollout dumps live beside the run's checkpoints so a finished run needs no
# copying: same shared-storage path, one directory per run. DUMP_ROOT overrides.
DUMP_ROOT=${DUMP_ROOT:-${SAVE_DIR}/debug_rollout}
mkdir -p "${DUMP_ROOT}" 2>/dev/null || true

case "${MODE}" in
rollout-debug)
   # Task 10: one prompt, one sample, rollout only; inspect the dump with
   # scripts/inspect_rollout_dump.py before going further.
   ROLLOUT_ARGS+=(
      --num-rollout 1
      --rollout-batch-size 1
      --n-samples-per-prompt 1
      --global-batch-size 1
      --debug-rollout-only
      --save-debug-rollout-data "${DUMP_ROOT}/rollout_{rollout_id}.pt"
   )
   ;;
train-smoke)
   # Task 11: exactly one GRPO iteration; >= 2 samples per prompt so the group
   # has within-group reward contrast. No checkpoint saving.
   ROLLOUT_ARGS+=(
      --num-rollout 1
      --rollout-batch-size 1
      --n-samples-per-prompt 2
      --global-batch-size 2
      --save-debug-rollout-data "${DUMP_ROOT}/rollout_{rollout_id}.pt"
      --save-debug-train-data "${REPO}/results/debug_train/train_{rollout_id}_{rank}.pt"
   )
   ;;
full|full-async)
   # Phase 5 baseline: 113-row easy train split. With the defaults, 8 prompts
   # x 8 samples = 64 concurrent trajectories per rollout and global-batch-size
   # 64 => exactly one on-policy optimizer step per rollout.
   ROLLOUT_ARGS+=(
      --num-epoch "${NUM_EPOCH}"
      --rollout-batch-size "${ROLLOUT_BS}"
      --n-samples-per-prompt "${N_SAMPLES}"
      --global-batch-size "${GLOBAL_BS}"
      --rollout-shuffle
   )
   # Save periodically and auto-resume: slime resumes from --load only when
   # that directory holds latest_checkpointed_iteration.txt (restoring
   # weights, optimizer, RNG, and rollout position); otherwise it falls back
   # to --ref-load and starts from rollout 0. Pointing --load at the save dir
   # makes a relaunch of the same EXP_NAME continue from the last save.
   SAVE_INTERVAL=${SAVE_INTERVAL:-10}
   CKPT_ARGS+=(
      --save "${SAVE_DIR}/"
      --load "${SAVE_DIR}/"
      --save-interval "${SAVE_INTERVAL}"
   )
   # DUMP_ROLLOUTS=1 saves every train and eval rollout (rollout_<id>.pt /
   # rollout_eval_<id>.pt) for scripts/inspect_rollout_dump.py. One .pt per
   # rollout; adds up over a long run, so off by default.

   if [ "${DUMP_ROLLOUTS:-0}" = "1" ]; then
      ROLLOUT_ARGS+=(
         --save-debug-rollout-data "${DUMP_ROOT}/rollout_{rollout_id}.pt"
      )
   fi
   if [ "${DYNAMIC_FILTER}" = "1" ]; then
      if [ "${MODE}" = "full-async" ]; then
         echo "DYNAMIC_FILTER=1 requires MODE=full: the fully-async rollout worker never calls the dynamic filter." >&2
         exit 1
      fi
      ROLLOUT_ARGS+=(
         --dynamic-sampling-filter-path officeqa_rl.training.filters.check_execution_reward_nonzero_std
      )
   fi
   if [ -n "${OVER_SAMPLING_BS}" ]; then
      ROLLOUT_ARGS+=(--over-sampling-batch-size "${OVER_SAMPLING_BS}")
   fi
   if [ "${MODE}" = "full-async" ]; then
      # Fully-async rollout: a background worker keeps the generation queue
      # warm across rollout boundaries. slime's fully-async path does not
      # support periodic eval, so EVAL_ARGS stays empty here.
      ROLLOUT_ARGS+=(
         --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
      )
   else
      # Periodic held-out eval on the all-hard pro split (expect pessimistic
      # eval numbers; that is deliberate).
      EVAL_ARGS+=(
         --eval-interval 5
         --eval-prompt-data officeqa_pro "${REPO}/data/slime/officeqa_pro_eval.jsonl"
         --n-samples-per-eval-prompt 1
         --eval-max-response-len "${MAX_RESP_LEN}"
      )
   fi
   ;;
esac

GRPO_ARGS=(
   --advantage-estimator "${ADV_ESTIMATOR}"
   --kl-loss-coef "${KL_LOSS_COEF}"
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef "${ENTROPY_COEF}"
   --eps-clip "${EPS_CLIP}"
)
if [ -n "${EPS_CLIP_HIGH}" ]; then
   GRPO_ARGS+=(--eps-clip-high "${EPS_CLIP_HIGH}")
fi
# slime only adds the KL term (and computes ref log-probs) under --use-kl-loss.
if nonzero "${KL_LOSS_COEF}"; then
   GRPO_ARGS+=(--use-kl-loss)
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# Keep Adam state on CPU (OPTIMIZER_CPU_OFFLOAD=0 to disable): frees the
# ~12 GiB/GPU fp32 optimizer shard, which raises the reserved-pool ceiling for
# the recurring 15.2 GiB logits-grad block from ~57 to ~69 GiB and makes the
# fragmentation OOMs (rollouts 2-3 of the 2026-07-15 runs) structurally
# impossible. Numerically identical (decoupled_weight_decay/AdamW is already
# the default); costs one CPU step + D2H/H2D per optimizer step, overlapped.
OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD:-1}
if [ "${OPTIMIZER_CPU_OFFLOAD}" = "1" ]; then
   # --use-precision-aware-optimizer is required by the hybrid device
   # optimizer; with the default fp32 dtypes it is numerically identical.
   OPTIMIZER_ARGS+=(
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
      --use-precision-aware-optimizer
   )
fi

# Full activation recompute is required, not optional: the Qwen3.5 plugin runs
# the linear-attention layers duplicated on every TP/CP rank (see
# slime_plugins/models/hf_attention.py), so their activations shard with
# neither TP nor CP — without recompute a 64k-token microbatch OOMs an 80 GiB
# GPU regardless of parallelism (as the 2026-07-14 baseline run did at 32k).
PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE}"
   --context-parallel-size "${CP_SIZE}"
   --pipeline-model-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768
   --train-memory-margin-bytes "${TRAIN_MEM_MARGIN_BYTES}"
   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
   --train-env-vars "{\"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_ALLOC_CONF}\"}"
)
if [ "${ADV_ESTIMATOR}" != "dr_grpo" ]; then
   PERF_ARGS+=(--calculate-per-token-loss)
fi
if [ "${TP_SIZE}" -gt 1 ]; then
   PERF_ARGS+=(--sequence-parallel)
fi

if [ "${MODE}" = "full-async" ]; then
   # Fully-async requires disaggregation: actor and SGLang on disjoint GPUs.
   TRAIN_SCRIPT="train_async.py"
   ACTOR_GPUS=${ACTOR_GPUS:-4}
   ROLLOUT_GPUS=${ROLLOUT_GPUS:-$((NUM_GPUS - ACTOR_GPUS))}
else
   TRAIN_SCRIPT="train.py"
   ACTOR_GPUS=${NUM_GPUS}
   ROLLOUT_GPUS=${NUM_GPUS}
fi

SGLANG_ARGS=(
   --rollout-num-gpus "${ROLLOUT_GPUS}"
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

if [ "${MODE}" = "full-async" ]; then
   # train_async.py asserts against --colocate; disjoint GPUs need no offload.
   TRAIN_FLAGS=()
else
   TRAIN_FLAGS=(--colocate)
   if [ "${MODE}" != "rollout-debug" ]; then
      # Colocated actor + SGLang on the same GPUs: park each side in host
      # memory while the other runs.
      TRAIN_FLAGS+=(--offload)
   fi
fi

# W&B is on by default; disable with OFFICEQA_WANDB=0 or WANDB_MODE=offline.
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
   elif [ "${WANDB_MODE:-online}" = "online" ] && \
        ! python3 -c "import wandb, sys; sys.exit(0 if wandb.api.api_key else 1)" 2>/dev/null; then
      echo "W&B is enabled but no credentials were found. Either:" >&2
      echo "  export WANDB_API_KEY=<key>   (or run 'wandb login' in this container)" >&2
      echo "  WANDB_MODE=offline           (log locally, sync later)" >&2
      echo "  OFFICEQA_WANDB=0             (disable tracking)" >&2
      exit 1
   fi
fi

export no_proxy="localhost,127.0.0.1,0.0.0.0"
ray start --head --num-gpus "${NUM_GPUS}" --disable-usage-stats \
   --dashboard-host=0.0.0.0 --dashboard-port=8265

# Persist the Triton autotune cache on /mnt. Besides head/block dimensions,
# ShortConvolution keys on packed-sequence count and the L2/fused norms key on
# flattened-token buckets. A fresh container otherwise re-benchmarks shapes
# mid-training. Run scripts/prewarm_fla_kernels.py on an idle GPU after wiping
# this directory or changing FLA/Triton/model/TP/CP.
TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${REPO}/results/triton_cache}
mkdir -p "${TRITON_CACHE_DIR}"
PREWARM_MANIFEST="${TRITON_CACHE_DIR}/.officeqa_fla_prewarm_complete.json"
if [ "${MODE}" != "rollout-debug" ] && [ ! -f "${PREWARM_MANIFEST}" ]; then
   echo "WARNING: ${TRITON_CACHE_DIR} has no completed FLA prewarm manifest." >&2
   echo "A missing or partial cache can autotune mid-training and hang/OOM. Run" >&2
   echo "  python3 scripts/prewarm_fla_kernels.py" >&2
   echo "on an idle GPU first (see README: Run ladder)." >&2
fi

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "localhost,127.0.0.1,0.0.0.0",
    "PYTHONPATH": "${REPO}/src:${SLIME_DIR}:/root/Megatron-LM",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "TRITON_CACHE_DIR": "${TRITON_CACHE_DIR}"
  }
}
EOF_JSON
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${SLIME_DIR}/${TRAIN_SCRIPT}" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   "${TRAIN_FLAGS[@]}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
