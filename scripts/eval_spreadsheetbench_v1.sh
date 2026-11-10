#!/usr/bin/env bash
set -euo pipefail

# SpreadsheetBench v1 / Verified evaluation launcher.
#
# Examples:
#   SSB1_DATA_ROOT=/path/to/verified_400 scripts/eval_spreadsheetbench_v1.sh
#   SSB1_MODEL=/path/to/checkpoint scripts/eval_spreadsheetbench_v1.sh
#   SSB1_BACKEND=openai-server SSB1_MODEL=my-model scripts/eval_spreadsheetbench_v1.sh
#   SSB1_UID=SSB1__0 scripts/eval_spreadsheetbench_v1.sh
#
# Local-vLLM defaults match the standard 8-GPU evaluation recipe for models
# that fit on one GPU: pass@8 at temperature 1.0, DP=8/TP=1/PP=1, 48 global
# concurrent episodes, and prefix caching. For larger models, increase TP (or
# PP), reduce DP so DP * TP * PP equals the GPU count, and tune concurrency and
# max-num-seqs for the model's KV-cache headroom.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/eval_paths.sh"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ "${PYTHON_BIN}" == */* ]]; then
  export PATH="$(cd -- "$(dirname -- "${PYTHON_BIN}")" && pwd):${PATH}"
fi

BACKEND="${SSB1_BACKEND:-vllm}"
if [[ "${BACKEND}" == "vllm" ]]; then
  MODEL="${SSB1_MODEL:-${VLLM_MODEL:-/mnt/local-ssd/Qwen3.5-4B}}"
  DEFAULT_TEMPERATURE=1.0
  DEFAULT_SAMPLES_PER_TASK=8
  DEFAULT_CONCURRENCY=48
  DEFAULT_DATA_PARALLEL_SIZE=8
else
  MODEL="${SSB1_MODEL:?SSB1_MODEL is required for backend ${BACKEND}}"
  DEFAULT_TEMPERATURE=1.0
  DEFAULT_SAMPLES_PER_TASK=1
  DEFAULT_CONCURRENCY=4
  DEFAULT_DATA_PARALLEL_SIZE=1
fi

# A one-UID launch is normally a smoke test, not a pass@8 measurement. An
# explicit SSB1_SAMPLES_PER_TASK still takes precedence.
if [[ -n "${SSB1_UID:-}" && -z "${SSB1_SAMPLES_PER_TASK+x}" ]]; then
  DEFAULT_SAMPLES_PER_TASK=1
fi

TEMPERATURE="${SSB1_TEMPERATURE:-${DEFAULT_TEMPERATURE}}"
SAMPLES_PER_TASK="${SSB1_SAMPLES_PER_TASK:-${DEFAULT_SAMPLES_PER_TASK}}"
CONCURRENCY="${SSB1_CONCURRENCY:-${DEFAULT_CONCURRENCY}}"
DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-${DEFAULT_DATA_PARALLEL_SIZE}}"

DATA_ROOT="${SSB1_DATA_ROOT:-${ROOT_DIR}/data/spreadsheetbench_v1/verified_400}"
SANDBOX_IMAGE="${SSB1_SANDBOX_IMAGE:-spreadsheetbench-v1-sandbox}"
MODEL_TAG="$(basename "${MODEL}")"
RESULTS_DIR="${SSB1_OUTPUT_DIR:-${EVAL_RESULTS_ROOT}/ssb1-${BACKEND}-${MODEL_TAG}}"
WORKSPACE_ROOT="${SSB1_WORKSPACE_ROOT:-${RESULTS_DIR}/workspaces}"

if [[ ! -f "${DATA_ROOT}/dataset.json" ]]; then
  echo "SpreadsheetBench v1 dataset.json not found at ${DATA_ROOT}." >&2
  echo "Set SSB1_DATA_ROOT to an extracted full or Verified release." >&2
  exit 2
fi

if ! docker image inspect "${SANDBOX_IMAGE}" >/dev/null 2>&1; then
  echo "Sandbox image ${SANDBOX_IMAGE} not found; building it..."
  docker build -t "${SANDBOX_IMAGE}" "${ROOT_DIR}/docker/spreadsheetbench"
fi

ARGS=(
  "${ROOT_DIR}/scripts/run_spreadsheetbench_v1_eval.py"
  --data-root "${DATA_ROOT}"
  --generation-backend "${BACKEND}"
  --model "${MODEL}"
  --output-dir "${RESULTS_DIR}"
  --workspace-root "${WORKSPACE_ROOT}"
  --sandbox-image "${SANDBOX_IMAGE}"
  --step-limit "${SSB1_STEP_LIMIT:-50}"
  --max-tokens "${SSB1_MAX_TOKENS:-8192}"
  --command-timeout-sec "${SSB1_COMMAND_TIMEOUT_SEC:-120}"
  --max-observation-chars "${SSB1_MAX_OBSERVATION_CHARS:-12000}"
  --replay-timeout-sec "${SSB1_REPLAY_TIMEOUT_SEC:-600}"
  --recalc-timeout-sec "${SSB1_RECALC_TIMEOUT_SEC:-600}"
  --temperature "${TEMPERATURE}"
  --concurrency "${CONCURRENCY}"
  --samples-per-task "${SAMPLES_PER_TASK}"
  --num-shards "${SSB1_NUM_SHARDS:-1}"
  --shard-index "${SSB1_SHARD_INDEX:-0}"
  --failure-retries "${SSB1_FAILURE_RETRIES:-2}"
  --failure-retry-delay-sec "${SSB1_FAILURE_RETRY_DELAY_SEC:-60}"
)

if [[ -n "${SSB1_UID:-}" ]]; then
  ARGS+=(--uid "${SSB1_UID}")
fi
if [[ "${SSB1_SKIP_RECALC:-0}" == "1" ]]; then
  ARGS+=(--skip-recalc)
fi

case "${BACKEND}" in
  vllm)
    ARGS+=(
      --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE:-1}"
      --pipeline-parallel-size "${VLLM_PIPELINE_PARALLEL_SIZE:-1}"
      # The shared backend assigns episodes round-robin and pins all turns in
      # an episode to one DP rank so its prefix remains replica-local.
      --data-parallel-size "${DATA_PARALLEL_SIZE}"
      --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
      --max-num-seqs "${VLLM_MAX_NUM_SEQS:-16}"
    )
    if [[ "${VLLM_ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
      ARGS+=(--enable-prefix-caching)
    else
      ARGS+=(--no-enable-prefix-caching)
    fi
    if [[ "${SSB1_ENABLE_THINKING:-}" == "1" ]]; then
      ARGS+=(--enable-thinking)
    elif [[ "${SSB1_ENABLE_THINKING:-}" == "0" ]]; then
      ARGS+=(--no-enable-thinking)
    fi
    if [[ "${VLLM_ENABLE_EXPERT_PARALLEL:-0}" == "1" ]]; then
      ARGS+=(--enable-expert-parallel)
    fi
    if [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
      ARGS+=(--max-model-len "${VLLM_MAX_MODEL_LEN}")
    fi
    if [[ -n "${VLLM_GDN_PREFILL_BACKEND:-}" ]]; then
      ARGS+=(--gdn-prefill-backend "${VLLM_GDN_PREFILL_BACKEND}")
    fi
    if [[ -n "${VLLM_SAFETENSORS_LOAD_STRATEGY:-}" ]]; then
      ARGS+=(--safetensors-load-strategy "${VLLM_SAFETENSORS_LOAD_STRATEGY}")
    fi
    ;;
  openai-server)
    ARGS+=(--openai-base-url "${SSB1_OPENAI_BASE_URL:-http://localhost:8000/v1}")
    ;;
  vertex)
    ARGS+=(
      --vertex-project "${VERTEX_PROJECT:?VERTEX_PROJECT is required for Vertex}"
      --vertex-location "${VERTEX_LOCATION:?VERTEX_LOCATION is required for Vertex}"
      --vertex-endpoint-id "${VERTEX_ENDPOINT_ID:?VERTEX_ENDPOINT_ID is required for Vertex}"
      --vertex-protocol "${VERTEX_PROTOCOL:-sglang}"
      --vertex-tokenizer "${VERTEX_TOKENIZER:-/mnt/Qwen3.5-4B}"
      --vertex-max-retries "${VERTEX_MAX_RETRIES:-3}"
      --vertex-quota-retries "${VERTEX_QUOTA_RETRIES:-10}"
      --vertex-request-timeout-sec "${VERTEX_REQUEST_TIMEOUT_SEC:-600}"
    )
    if [[ -n "${VERTEX_URL:-}" ]]; then
      ARGS+=(--vertex-url "${VERTEX_URL}")
    fi
    if [[ -n "${VERTEX_INSTRUCTION:-}" ]]; then
      ARGS+=(--vertex-instruction "${VERTEX_INSTRUCTION}")
    fi
    ;;
esac

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${RESULTS_DIR}"
LOG_FILE="${RESULTS_DIR}/eval-$(date +%Y%m%d-%H%M%S).log"
echo "SpreadsheetBench v1 evaluation of ${MODEL} (${BACKEND}); logging to ${LOG_FILE}"
if [[ "${BACKEND}" == "vllm" ]]; then
  echo "Recipe: temperature=${TEMPERATURE}, pass@${SAMPLES_PER_TASK}, concurrency=${CONCURRENCY}, DP=${DATA_PARALLEL_SIZE}, TP=${VLLM_TENSOR_PARALLEL_SIZE:-1}, PP=${VLLM_PIPELINE_PARALLEL_SIZE:-1}, prefix-cache=${VLLM_ENABLE_PREFIX_CACHING:-1}, gdn-prefill=${VLLM_GDN_PREFILL_BACKEND:-auto}"
fi
"${PYTHON_BIN}" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
