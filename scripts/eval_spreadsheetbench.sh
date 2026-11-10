#!/usr/bin/env bash
set -euo pipefail

# SpreadsheetBench 2 evaluation launcher.
#
# Examples:
#   scripts/eval_spreadsheetbench.sh                       # vLLM, /mnt/Qwen3.5-4B
#   VLLM_MODEL=/path/to/ckpt scripts/eval_spreadsheetbench.sh
#   SSB_BACKEND=openai-server SSB_MODEL=my-model scripts/eval_spreadsheetbench.sh
#   SSB_BACKEND=gemini SSB_MODEL=gemini-3.5-flash scripts/eval_spreadsheetbench.sh
#   SSB_UID=Template__01_01 scripts/eval_spreadsheetbench.sh   # single task
#
# Traces (full message logs + command history) land in per-task JSONs under
# the results dir; graded output workbooks under <results>/outputs/.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/eval_paths.sh"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ "${PYTHON_BIN}" == */* ]]; then
  export PATH="$(cd -- "$(dirname -- "${PYTHON_BIN}")" && pwd):${PATH}"
fi

BACKEND="${SSB_BACKEND:-vllm}"
if [[ "${BACKEND}" == "vllm" ]]; then
  MODEL="${SSB_MODEL:-${VLLM_MODEL:-/mnt/Qwen3.5-4B}}"
  # Standard 8-GPU recipe for models that fit one replica per GPU. For larger
  # models, increase TP (or PP), reduce DP so DP * TP * PP equals the GPU
  # count, and tune concurrency/max-num-seqs for the available KV-cache memory.
  DEFAULT_TEMPERATURE=1.0
  DEFAULT_SAMPLES_PER_TASK=8
  DEFAULT_CONCURRENCY=48
  DEFAULT_DATA_PARALLEL_SIZE=8
else
  MODEL="${SSB_MODEL:?SSB_MODEL is required for backend ${BACKEND}}"
  DEFAULT_TEMPERATURE=1.0
  DEFAULT_SAMPLES_PER_TASK=1
  DEFAULT_CONCURRENCY=4
  DEFAULT_DATA_PARALLEL_SIZE=1
fi

# A one-task launch is normally a smoke test. An explicit pass count still wins.
if [[ -n "${SSB_UID:-}" && -z "${SSB_SAMPLES_PER_TASK+x}" ]]; then
  DEFAULT_SAMPLES_PER_TASK=1
fi

TEMPERATURE="${SSB_TEMPERATURE:-${DEFAULT_TEMPERATURE}}"
SAMPLES_PER_TASK="${SSB_SAMPLES_PER_TASK:-${DEFAULT_SAMPLES_PER_TASK}}"
CONCURRENCY="${SSB_CONCURRENCY:-${DEFAULT_CONCURRENCY}}"
DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-${DEFAULT_DATA_PARALLEL_SIZE}}"

DATA_ROOT="${SSB_DATA_ROOT:-${ROOT_DIR}/data/spreadsheetbench_v2/spreadsheetbench-v2}"
SANDBOX_IMAGE="${SSB_SANDBOX_IMAGE:-spreadsheetbench-v2-sandbox}"
MODEL_TAG="$(basename "${MODEL}")"
RESULTS_DIR="${SSB_OUTPUT_DIR:-${EVAL_RESULTS_ROOT}/ssb2-${BACKEND}-${MODEL_TAG}}"

# Build the sandbox image once; it mirrors the official benchmark image
# (python 3.11 + openpyxl/pandas + LibreOffice + view_xlsx + recalc tool).
if ! docker image inspect "${SANDBOX_IMAGE}" >/dev/null 2>&1; then
  echo "Sandbox image ${SANDBOX_IMAGE} not found; building it..."
  docker build -t "${SANDBOX_IMAGE}" "${ROOT_DIR}/docker/spreadsheetbench"
fi

ARGS=(
  "${ROOT_DIR}/scripts/run_spreadsheetbench_eval.py"
  --data-root "${DATA_ROOT}"
  --generation-backend "${BACKEND}"
  --model "${MODEL}"
  --output-dir "${RESULTS_DIR}"
  --sandbox-image "${SANDBOX_IMAGE}"
  --step-limit "${SSB_STEP_LIMIT:-50}"
  --max-tokens "${SSB_MAX_TOKENS:-8192}"
  --command-timeout-sec "${SSB_COMMAND_TIMEOUT_SEC:-120}"
  --max-observation-chars "${SSB_MAX_OBSERVATION_CHARS:-12000}"
  --temperature "${TEMPERATURE}"
  --concurrency "${CONCURRENCY}"
  --samples-per-task "${SAMPLES_PER_TASK}"
  --failure-retries "${SSB_FAILURE_RETRIES:-2}"
  --failure-retry-delay-sec "${SSB_FAILURE_RETRY_DELAY_SEC:-60}"
  --num-shards "${SSB_NUM_SHARDS:-1}"
  --shard-index "${SSB_SHARD_INDEX:-0}"
)

if [[ -n "${SSB_WORKSPACE_ROOT:-}" ]]; then
  ARGS+=(--workspace-root "${SSB_WORKSPACE_ROOT}")
fi

# Default categories are the cell-graded ones; Visualization needs the
# official Windows-only VLM evaluator.
read -r -a CATEGORIES <<< "${SSB_CATEGORIES:-Debugging Financial_Model Template}"
ARGS+=(--categories "${CATEGORIES[@]}")

if [[ -n "${SSB_UID:-}" ]]; then
  ARGS+=(--uid "${SSB_UID}")
fi

case "${BACKEND}" in
  vllm)
    ARGS+=(
      --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE:-1}"
      --pipeline-parallel-size "${VLLM_PIPELINE_PARALLEL_SIZE:-1}"
      # The shared backend pins all turns of an episode to one DP rank so its
      # growing prefix remains replica-local and cacheable.
      --data-parallel-size "${DATA_PARALLEL_SIZE}"
      --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
      --max-num-seqs "${VLLM_MAX_NUM_SEQS:-16}"
    )
    if [[ "${VLLM_ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
      ARGS+=(--enable-prefix-caching)
    else
      ARGS+=(--no-enable-prefix-caching)
    fi
    if [[ "${VLLM_ENABLE_EXPERT_PARALLEL:-0}" == "1" ]]; then
      ARGS+=(--enable-expert-parallel)
    fi
    if [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
      ARGS+=(--max-model-len "${VLLM_MAX_MODEL_LEN}")
    fi
    if [[ -n "${VLLM_SAFETENSORS_LOAD_STRATEGY:-}" ]]; then
      ARGS+=(--safetensors-load-strategy "${VLLM_SAFETENSORS_LOAD_STRATEGY}")
    fi
    if [[ -n "${VLLM_GDN_PREFILL_BACKEND:-}" ]]; then
      ARGS+=(--gdn-prefill-backend "${VLLM_GDN_PREFILL_BACKEND}")
    fi
    ;;
  openai-server)
    ARGS+=(--openai-base-url "${SSB_OPENAI_BASE_URL:-http://localhost:8000/v1}")
    ;;
  vertex)
    ARGS+=(
      --vertex-project "${VERTEX_PROJECT:?VERTEX_PROJECT is required for Vertex}"
      --vertex-location "${VERTEX_LOCATION:-us-east1}"
      --vertex-endpoint-id "${VERTEX_ENDPOINT_ID:?VERTEX_ENDPOINT_ID is required for the Vertex backend}"
      --vertex-protocol "${VERTEX_PROTOCOL:-sglang}"
      --vertex-tokenizer "${VERTEX_TOKENIZER:-/mnt/Qwen3.5-4B}"
      --vertex-max-retries "${VERTEX_MAX_RETRIES:-3}"
      --vertex-quota-retries "${VERTEX_QUOTA_RETRIES:-15}"
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
echo "SpreadsheetBench 2 evaluation of ${MODEL} (${BACKEND}); logging to ${LOG_FILE}"
"${PYTHON_BIN}" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
