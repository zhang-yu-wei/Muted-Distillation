#!/bin/bash
# Online teacher-trace SFT for the OfficeQA Qwen3.5-4B agent: a frozen teacher
# (SGLang/vLLM server, plain http or Vertex rawPredict) runs the agentic
# episodes INSIDE the training loop, and each episode trains the student with
# hard-label cross-entropy on the teacher tokens (per-assistant-turn samples,
# qwen3_5 loss masking). train_async.py overlaps rollout N+1 generation with
# training on rollout N, so the teacher endpoints and the training GPUs stay
# busy simultaneously. By default no student inference engines are launched
# (--debug-train-only) and there is no in-loop eval; evaluate checkpoints
# offline with convert_torch_dist_to_hf.py + scripts/eval_vllm_full.sh, or set
# SFT_EVAL_INTERVAL to run the held-out set inside the loop.
#
# Usage (inside the slime container):
#   SFT_TEACHER_URL=http://host:30000/generate bash scripts/train_officeqa_sft_online.sh smoke
#   SFT_TEACHER_URL=...  bash scripts/train_officeqa_sft_online.sh full
#
# Teacher selection. Both SFT_TEACHER_URL and SFT_TEACHER_ENDPOINT_ID accept a
# COMMA-SEPARATED list of endpoints serving the same model, exactly like OPD's
# OPD_TEACHER_URL / OPD_ENDPOINT_ID: episodes spread over the pool with
# per-episode affinity (one episode runs every turn against one endpoint, so
# its prefix cache keeps hitting), and SFT_TEACHER_CONCURRENCY applies PER
# ENDPOINT — a pool of 3 with concurrency 8 runs up to 24 episodes at once,
# never more than 8 against any one deployment.
#   SFT_TEACHER_URL       full URLs (http://host:port/generate, .../v1/completions,
#                         or https://...:rawPredict), OR
#   SFT_TEACHER_PROJECT / SFT_TEACHER_LOCATION / SFT_TEACHER_ENDPOINT_ID
#                         Vertex endpoint id list; rawPredict URLs are derived.
#                         All ids must sit in one region — for a cross-region
#                         pool pass full URLs through SFT_TEACHER_URL instead.
#   SFT_TEACHER_PROTOCOL  sglang (default) or vllm
#   SFT_TEACHER_MODEL     served model id (required by the vllm protocol payload)
#   SFT_TEACHER_CONCURRENCY (16) in-flight episodes per endpoint. The 397B
#                         Vertex deployments tolerate at most 8 each.
#   SFT_TEACHER_TIMEOUT_SEC (600) hard per-request timeout
#   SFT_TEACHER_ATTEMPTS  (3)  retry budget for transient failures per request
#   SFT_TEACHER_QUOTA_ATTEMPTS (10) separate budget for 429/503 capacity
#                         pushback, so a model-server restart cannot exhaust
#                         the transient budget
#   SFT_TEACHER_BACKOFF_SEC (2.0) truncated-exponential backoff base, capped at
#                         60 s and jittered 0.5x-1.5x (same as OPD_BACKOFF_SEC).
#                         Raise attempts/backoff to ride out a redeploy: 18
#                         attempts at base 20 covers roughly 8-25 minutes.
#   SFT_FILTER            correct (default; rejection-sampled SFT) or all
#   SFT_EVAL_INTERVAL     (0 = off) rollouts between in-loop evals on the
#                         held-out OfficeQA-Pro set. Non-zero starts student
#                         SGLang engines that --debug-train-only otherwise
#                         omits, so it changes the GPU footprint — see the
#                         EVAL_ARGS block. SFT_EVAL_SAMPLES (1) and
#                         SGLANG_MEM_FRACTION (0.7) tune it.
#
# Try scripts/smoke_teacher_sft_rollout.py first: it drives the same rollout
# function against the teacher with no Ray/GPU footprint.

set -euo pipefail

MODE=${1:-smoke}
case "${MODE}" in
smoke|full) ;;
*) echo "usage: $0 [smoke|full]" >&2; exit 2 ;;
esac

export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

REPO=${REPO:-/mnt/officeqa-rl}

SLIME_DIR="${REPO}/third_party/slime"
MODEL=${QWEN35_4B_MODEL:-/mnt/Qwen3.5-4B}
REF_LOAD=${REF_LOAD:-/mnt/Qwen3.5-4B_torch_dist}
NUM_GPUS=${NUM_GPUS:-8}

# Teacher pool.
SFT_TEACHER_URL=${SFT_TEACHER_URL:-}
SFT_TEACHER_PROTOCOL=${SFT_TEACHER_PROTOCOL:-sglang}
SFT_TEACHER_MODEL=${SFT_TEACHER_MODEL:-teacher}
SFT_TEACHER_CONCURRENCY=${SFT_TEACHER_CONCURRENCY:-16}
SFT_TEACHER_TIMEOUT_SEC=${SFT_TEACHER_TIMEOUT_SEC:-600}
# Retry budgets, split the way the backend counts them: transient failures
# (connection resets, 5xx that are not capacity) and capacity pushback
# (429/503) have separate budgets, so a model server restart cannot eat the
# transient budget. Backoff is truncated-exponential from this base, capped at
# 60 s and jittered 0.5x-1.5x, as in the OPD teacher client.
SFT_TEACHER_ATTEMPTS=${SFT_TEACHER_ATTEMPTS:-3}
SFT_TEACHER_QUOTA_ATTEMPTS=${SFT_TEACHER_QUOTA_ATTEMPTS:-10}
SFT_TEACHER_BACKOFF_SEC=${SFT_TEACHER_BACKOFF_SEC:-2.0}
SFT_FILTER=${SFT_FILTER:-correct}
if [ -z "${SFT_TEACHER_URL}" ]; then
   if [ -z "${SFT_TEACHER_ENDPOINT_ID:-}" ]; then
      echo "Set SFT_TEACHER_URL or SFT_TEACHER_PROJECT/SFT_TEACHER_LOCATION/SFT_TEACHER_ENDPOINT_ID" >&2
      exit 1
   fi
   SFT_TEACHER_PROJECT=${SFT_TEACHER_PROJECT:?SFT_TEACHER_PROJECT is required with SFT_TEACHER_ENDPOINT_ID}
   SFT_TEACHER_LOCATION=${SFT_TEACHER_LOCATION:?SFT_TEACHER_LOCATION is required with SFT_TEACHER_ENDPOINT_ID}
   urls=()
   IFS=',' read -ra endpoint_ids <<< "${SFT_TEACHER_ENDPOINT_ID}"
   for endpoint_id in "${endpoint_ids[@]}"; do
      urls+=("https://${SFT_TEACHER_LOCATION}-aiplatform.googleapis.com/v1/projects/${SFT_TEACHER_PROJECT}/locations/${SFT_TEACHER_LOCATION}/endpoints/${endpoint_id}:rawPredict")
   done
   SFT_TEACHER_URL=$(IFS=','; echo "${urls[*]}")
fi

# Rollout geometry: every rollout step generates ROLLOUT_BS * N_SAMPLES whole
# teacher episodes; slime schedules GLOBAL_BS episodes per optimizer step.
N_SAMPLES=${N_SAMPLES:-8}
ROLLOUT_BS=${ROLLOUT_BS:-4}
GLOBAL_BS=${GLOBAL_BS:-32}
NUM_EPOCH=${NUM_EPOCH:-8}
TEMPERATURE=${TEMPERATURE:-1.0}
MAX_CTX_LEN=${MAX_CTX_LEN:-65536}
MAX_RESP_LEN=${MAX_RESP_LEN:-8192}
AGENT_STEP_LIMIT=${AGENT_STEP_LIMIT:-}
if [ $(( (ROLLOUT_BS * N_SAMPLES) % GLOBAL_BS )) -ne 0 ]; then
   echo "ROLLOUT_BS*N_SAMPLES=$((ROLLOUT_BS * N_SAMPLES)) must be divisible by GLOBAL_BS=${GLOBAL_BS}" >&2
   exit 2
fi

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
SAVE_INTERVAL=${SAVE_INTERVAL:-10}
TRAIN_MEM_MARGIN_BYTES=${TRAIN_MEM_MARGIN_BYTES:-1073741824}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-4096}
PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-garbage_collection_threshold:0.7}

EXP_NAME=${EXP_NAME:-qwen3.5-4B-sft-online-${SFT_FILTER}}
if [ "${MODE}" = "smoke" ]; then
   EXP_NAME="${EXP_NAME}-smoke"
fi
SAVE_DIR=${SAVE_DIR:-${REPO}/results/checkpoints/${EXP_NAME}}

# Rollout dumps live beside the run's checkpoints so a finished run needs no
# copying: same shared-storage path, one directory per run. DUMP_ROOT overrides.
DUMP_ROOT=${DUMP_ROOT:-${SAVE_DIR}/debug_rollout}
mkdir -p "${DUMP_ROOT}" 2>/dev/null || true
# Raw episode traces (offline-pipeline JSON, one <uid>.s<i>.json per completed
# episode, including filter-rejected ones) — reusable by build_sft_dataset.py.
# Point it at the shared bucket for durability; empty disables persistence.
SFT_TRACE_DIR=${SFT_TRACE_DIR-${REPO}/results/sft-online-traces/${EXP_NAME}}

if [ ! -d "${MODEL}" ]; then
   echo "Qwen3.5 HF checkpoint not found: ${MODEL}" >&2
   exit 1
fi

# The rollout hook reads officeqa_sft_* from the custom config yaml (no CLI
# equivalents). Strip any such keys from the base file so the values below
# win, and key the filename on EXP_NAME so concurrent runs never share one.
CUSTOM_CONFIG="${REPO}/configs/training/officeqa_rollout.yaml"
DERIVED_CONFIG="${REPO}/results/officeqa_rollout_sft_online_${EXP_NAME}.yaml"
if [ ! -f "${CUSTOM_CONFIG}" ]; then
   echo "Base rollout config not found: ${CUSTOM_CONFIG}" >&2
   exit 1
fi
mkdir -p "${REPO}/results"
{
   grep -v -E '^(officeqa_step_limit|officeqa_corpus_dir|officeqa_workspace_root|officeqa_sft_[a-z_]+):' "${CUSTOM_CONFIG}"
   # Derive the environment paths from REPO so the base yaml's /mnt/officeqa-rl
   # defaults never leak into COS runs (same convention as the OPD launchers).
   echo "officeqa_corpus_dir: ${REPO}/data/treasury_bulletins_parsed/transformed"
   echo "officeqa_workspace_root: ${REPO}/results/rollout_workspaces"
   if [ -n "${AGENT_STEP_LIMIT}" ]; then
      echo "officeqa_step_limit: ${AGENT_STEP_LIMIT}"
   fi
   echo "# --- teacher SFT (written by scripts/train_officeqa_sft_online.sh) ---"
   echo "officeqa_sft_teacher_url: \"${SFT_TEACHER_URL}\""
   echo "officeqa_sft_teacher_protocol: \"${SFT_TEACHER_PROTOCOL}\""
   echo "officeqa_sft_teacher_model: \"${SFT_TEACHER_MODEL}\""
   echo "officeqa_sft_teacher_timeout_sec: ${SFT_TEACHER_TIMEOUT_SEC}"
   echo "officeqa_sft_teacher_max_retries: ${SFT_TEACHER_ATTEMPTS}"
   echo "officeqa_sft_teacher_quota_retries: ${SFT_TEACHER_QUOTA_ATTEMPTS}"
   echo "officeqa_sft_teacher_backoff_sec: ${SFT_TEACHER_BACKOFF_SEC}"
   echo "# in-flight episodes PER ENDPOINT (pool total = this x endpoint count)"
   echo "officeqa_sft_concurrency: ${SFT_TEACHER_CONCURRENCY}"
   echo "officeqa_sft_filter: \"${SFT_FILTER}\""
   if [ -n "${SFT_TRACE_DIR}" ]; then
      echo "officeqa_sft_trace_dir: \"${SFT_TRACE_DIR}\""
   fi
} > "${DERIVED_CONFIG}"
CUSTOM_CONFIG="${DERIVED_CONFIG}"

# Preflight: one 1-token generation against every pool member before Ray or
# GPUs. Catches a dead server, missing ADC, a wrong protocol, or (vllm) a
# wrong served model id here instead of as dropped episodes mid-run.
if [ "${SFT_PREFLIGHT:-1}" = "1" ]; then
   PYTHONPATH="${REPO}/src:${SLIME_DIR}" SFT_CONFIG="${CUSTOM_CONFIG}" \
   SFT_MODEL="${MODEL}" python3 - <<'EOF_PREFLIGHT'
import os, sys, yaml
from officeqa_rl.backends.vertex import VertexSGLangGenerationBackend

config = yaml.safe_load(open(os.environ["SFT_CONFIG"])) or {}
urls = [url.strip() for url in str(config["officeqa_sft_teacher_url"]).split(",") if url.strip()]
for index, url in enumerate(urls):
    backend = VertexSGLangGenerationBackend(
        str(config.get("officeqa_sft_teacher_model", "teacher")),
        base_url=url,
        protocol=str(config.get("officeqa_sft_teacher_protocol", "sglang")),
        tokenizer_path=os.environ["SFT_MODEL"],
        max_tokens=1, max_retries=1, quota_max_retries=1, request_timeout_sec=60.0,
    )
    print(f"teacher SFT preflight [{index + 1}/{len(urls)}] -> {url}", flush=True)
    try:
        backend._generate_sync([{"role": "user", "content": "ping"}], {"temperature": 0.0})
    except Exception as error:
        print(f"teacher SFT preflight FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        print("Check the URL/endpoint, ADC (`gcloud auth application-default login`),", file=sys.stderr)
        print("SFT_TEACHER_PROTOCOL, and SFT_TEACHER_MODEL. SFT_PREFLIGHT=0 skips this.", file=sys.stderr)
        sys.exit(1)
print("teacher SFT preflight OK", flush=True)
EOF_PREFLIGHT
fi

source "${SLIME_DIR}/scripts/models/qwen3.5-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL}"
   --ref-load "${REF_LOAD}"
)

SFT_ARGS=(
   --rollout-function-path officeqa_rl.training.teacher_sft_rollout.generate_rollout
   --prompt-data "${REPO}/data/slime/officeqa_full_minus_pro_train.jsonl"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --custom-config-path "${CUSTOM_CONFIG}"
   --custom-rollout-log-function-path officeqa_rl.training.metrics.log_rollout_extra_metrics
   --rollout-max-context-len "${MAX_CTX_LEN}"
   --rollout-max-response-len "${MAX_RESP_LEN}"
   --rollout-temperature "${TEMPERATURE}"
   --loss-type sft_loss
   --loss-mask-type qwen3_5
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
)

# In-loop eval on the held-out OfficeQA-Pro tasks, off by default. It is not a
# free flag: the teacher does all training generation, so without it slime runs
# --debug-train-only and starts NO student engines at all. Turning eval on adds
# SGLang engines for the student (colocated and offloaded, woken only for
# eval), and train_async pushes the trained weights to them before each eval,
# so the numbers track the student rather than the teacher.
#
# Two things must be set explicitly or the measurement is silently wrong:
# slime defaults --eval-function-path to --rollout-function-path, which here is
# the TEACHER rollout — eval would score the 397B teacher and read flat. Naming
# slime's own sglang rollout for eval routes it through
# --custom-generate-function-path, the student's OfficeQA agent loop.
SFT_EVAL_INTERVAL=${SFT_EVAL_INTERVAL:-0}
EVAL_ARGS=()
SGLANG_ARGS=()
if [ "${SFT_EVAL_INTERVAL}" -gt 0 ]; then
   SFT_ARGS+=(
      --eval-function-path slime.rollout.sglang_rollout.generate_rollout
      --custom-generate-function-path officeqa_rl.training.rollout.generate
      --custom-eval-rollout-log-function-path officeqa_rl.training.metrics.log_eval_extra_metrics
   )
   EVAL_ARGS=(
      --eval-interval "${SFT_EVAL_INTERVAL}"
      --eval-prompt-data officeqa_pro "${REPO}/data/slime/officeqa_pro_eval.jsonl"
      --n-samples-per-eval-prompt "${SFT_EVAL_SAMPLES:-1}"
      --eval-max-response-len "${MAX_RESP_LEN}"
   )
   SGLANG_ARGS=(
      --rollout-num-gpus "${NUM_GPUS}"
      --rollout-num-gpus-per-engine 1
      --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.7}"
      --colocate
      --offload
   )
else
   SFT_ARGS+=(--debug-train-only)
fi

# train_async overlaps rollout N+1's teacher generation with training on N, but
# it refuses colocation (train_async.py:11) and a single node has no spare GPUs
# to hand the eval engines — the actor already needs all NUM_GPUS at
# TP_SIZE*CP_SIZE. So with eval on, run the synchronous train.py and let the
# engines share the actor's GPUs (asleep during training, woken for eval). The
# cost is losing the generate/train overlap, which is why eval stays opt-in.
TRAIN_SCRIPT=train_async.py
if [ "${SFT_EVAL_INTERVAL}" -gt 0 ]; then
   TRAIN_SCRIPT=train.py
fi

if [ "${MODE}" = "smoke" ]; then
   # One optimizer step over one prompt's episodes, same masking path as full.
   SFT_ARGS+=(
      --num-rollout 1
      --rollout-batch-size 1
      --n-samples-per-prompt 2
      --global-batch-size 2
      --save-debug-rollout-data "${DUMP_ROOT}/rollout_{rollout_id}.pt"
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
      --n-samples-per-prompt "${N_SAMPLES}"
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
NVLINK_COUNT=$(PATH="/usr/local/nvidia/bin:${PATH}" nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "${NVLINK_COUNT}" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
# a3-high COS nodes have NVLink but no multicast config: with NVLS on, the
# first NCCL group dies binding multicast memory ("CUDA error 2 'out of
# memory'" with >60 GB free). The /mnt/disks layout marks those nodes.
case "${REPO}" in /mnt/disks/*) HAS_NVLINK=0 ;; esac
NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-${HAS_NVLINK}}
export no_proxy="127.0.0.1,${MASTER_ADDR}"

# The FLA/Triton autotune cache; a missing/partial cache can autotune
# mid-training and hang/OOM. Prewarm with scripts/prewarm_fla_kernels.py.
TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${REPO}/results/triton_cache}
mkdir -p "${TRITON_CACHE_DIR}"
if [ ! -f "${TRITON_CACHE_DIR}/.officeqa_fla_prewarm_complete.json" ]; then
   echo "WARNING: ${TRITON_CACHE_DIR} has no completed FLA prewarm manifest." >&2
   echo "Run  python3 scripts/prewarm_fla_kernels.py  on an idle GPU first." >&2
fi

# GOOGLE_APPLICATION_CREDENTIALS is forwarded when set so the teacher
# backend's ADC lookup resolves inside the Ray workers (COS containers mount
# the host home read-only at its host path, not at /root).
EXTRA_ENV=""
if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
   EXTRA_ENV="${EXTRA_ENV},\"GOOGLE_APPLICATION_CREDENTIALS\":\"${GOOGLE_APPLICATION_CREDENTIALS}\""
fi
if [ -n "${HF_HOME_DIR:-}" ]; then
   EXTRA_ENV="${EXTRA_ENV},\"HF_HOME\":\"${HF_HOME_DIR}\""
fi
# Packed variable-length batches fragment the caching allocator badly: an OOM
# here typically reports many GB "reserved but unallocated". expandable_segments
# lets those reservations be reused across differently-shaped microbatches. It
# changes allocation only, never numerics; set to "" to opt out.
#
# It is however incompatible with in-loop eval: --offload frees the SGLang
# engines' memory through TorchMemorySaver, which refuses to initialise under
# expandable_segments ("TorchMemorySaver is disabled for the current process")
# and kills every engine at startup. Engines win, so default it on only when
# there are none.
if [ "${SFT_EVAL_INTERVAL}" -gt 0 ]; then
   PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-}
else
   PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}
fi
if [ -n "${PYTORCH_CUDA_ALLOC_CONF}" ]; then
   EXTRA_ENV="${EXTRA_ENV},\"PYTORCH_CUDA_ALLOC_CONF\":\"${PYTORCH_CUDA_ALLOC_CONF}\""
fi

RUNTIME_ENV_JSON=$(printf '%s' \
   "{\"env_vars\":{\"PYTHONPATH\":\"/root/Megatron-LM:${REPO}/src:${SLIME_DIR}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"${NCCL_NVLS_ENABLE}\",\"TRITON_CACHE_DIR\":\"${TRITON_CACHE_DIR}\",\"no_proxy\":\"${no_proxy}\",\"MASTER_ADDR\":\"${MASTER_ADDR}\"${EXTRA_ENV}}}")

# Absolute entrypoint path: the ray job driver's cwd is the cwd of the `ray
# start` daemon, not of the submit client, so a relative train_async.py breaks
# when the launcher is invoked from outside SLIME_DIR.
TRAIN_CMD=(
   python3 "${SLIME_DIR}/${TRAIN_SCRIPT}"
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${NUM_GPUS}"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${SFT_ARGS[@]}"
   "${EVAL_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${MISC_ARGS[@]}"
)

IFS=',' read -ra _teacher_urls <<< "${SFT_TEACHER_URL}"
echo "online SFT mode=${MODE} filter=${SFT_FILTER} teacher=${SFT_TEACHER_URL} (${SFT_TEACHER_PROTOCOL})"
echo "teacher pool: ${#_teacher_urls[@]} endpoint(s), ${SFT_TEACHER_CONCURRENCY} concurrent episodes each" \
     "(<= $((SFT_TEACHER_CONCURRENCY * ${#_teacher_urls[@]})) in flight)"
echo "teacher retries: ${SFT_TEACHER_ATTEMPTS} transient / ${SFT_TEACHER_QUOTA_ATTEMPTS} capacity," \
     "backoff base ${SFT_TEACHER_BACKOFF_SEC}s, timeout ${SFT_TEACHER_TIMEOUT_SEC}s"
if [ "${MODE}" = "smoke" ]; then
   echo "episodes/rollout=2 global_bs=2 temp=${TEMPERATURE}"
else
   echo "episodes/rollout=$((ROLLOUT_BS * N_SAMPLES)) global_bs=${GLOBAL_BS} temp=${TEMPERATURE}"
fi
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
   mkdir -p "${REPO}/results/debug_rollout" "${REPO}/results/debug_train"
else
   mkdir -p "${SAVE_DIR}"
fi

RAY_TMP_ARGS=()
if [ -n "${RAY_TMP_DIR:-}" ]; then
   mkdir -p "${RAY_TMP_DIR}"
   RAY_TMP_ARGS+=(--temp-dir "${RAY_TMP_DIR}")
fi

# cd before ray start so the head daemon's cwd (which the job driver
# inherits) is deterministic regardless of where the launcher was invoked.
cd "${SLIME_DIR}"
ray stop --force || true
pkill -9 sglang || true
sleep 2
ray start --head \
   --node-ip-address "${MASTER_ADDR}" \
   --num-gpus "${NUM_GPUS}" \
   --disable-usage-stats \
   --dashboard-host=0.0.0.0 \
   --dashboard-port=8265 \
   "${RAY_TMP_ARGS[@]}"

cd "${SLIME_DIR}"
ray job submit \
   --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "${TRAIN_CMD[@]}"
