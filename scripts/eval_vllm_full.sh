#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/eval_paths.sh"

# Launch defaults; every value can still be overridden from the environment.
# VLLM_MODEL accepts a HuggingFace hub id (e.g. Qwen/Qwen3.5-4B), a local HF
# checkpoint directory, or an RL checkpoint converted with
# third_party/slime/tools/convert_torch_dist_to_hf.py.
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
# Launching .venv/bin/python directly does not activate the venv, so tools
# installed next to the interpreter (ninja for vLLM's JIT kernel builds) are
# not on PATH unless we put them there.
if [[ "${PYTHON_BIN}" == */* ]]; then
  export PATH="$(cd -- "$(dirname -- "${PYTHON_BIN}")" && pwd):${PATH}"
fi
MODEL="${VLLM_MODEL:-/mnt/Qwen3.5-4B}"
# Standard 8-GPU recipe for models that fit one replica per GPU. For larger
# models, increase TP (or PP), reduce DP so DP * TP * PP equals the available
# GPU count, and tune concurrency/max-num-seqs to the model's KV-cache headroom.
DEFAULT_TEMPERATURE=1.0
DEFAULT_SAMPLES_PER_TASK=8
DEFAULT_CONCURRENCY=48
DEFAULT_DATA_PARALLEL_SIZE=8
OFFICEQA_DATASET_CSV="${OFFICEQA_DATASET_CSV:-${ROOT_DIR}/data/officeqa_pro.csv}"
OFFICEQA_CORPUS_DIR="${OFFICEQA_CORPUS_DIR:-${ROOT_DIR}/data/treasury_bulletins_parsed/transformed}"
OFFICEQA_SANDBOX_IMAGE="${OFFICEQA_SANDBOX_IMAGE:-python:3.11-slim}"
# Results are grouped per model so checkpoint sweeps do not collide or
# cross-resume: results/vllm-<model basename>/.
MODEL_TAG="$(basename "${MODEL}")"
RESULTS_DIR="${OFFICEQA_OUTPUT_DIR:-${EVAL_RESULTS_ROOT}/vllm-${MODEL_TAG}}"
WORKSPACES="${OFFICEQA_WORKSPACE_ROOT:-${RESULTS_DIR}/workspaces}"

# OFFICEQA_UID is optional: set it to evaluate one question; leave it unset to
# evaluate the whole dataset with per-UID results, resume, and a summary.

ARGS=(
  "${ROOT_DIR}/scripts/run_full_corpus_inference.py"
  --generation-backend vllm
  --model "${MODEL}"
  --dataset-csv "${OFFICEQA_DATASET_CSV}"
  --corpus-dir "${OFFICEQA_CORPUS_DIR}"
  --workspace-root "${WORKSPACES}"
  --sandbox-runtime docker
  --sandbox-image "${OFFICEQA_SANDBOX_IMAGE}"
  --step-limit "${OFFICEQA_STEP_LIMIT:-50}"
  --corpus-profile "${OFFICEQA_CORPUS_PROFILE:-v1}"
  --max-tokens "${OFFICEQA_MAX_TOKENS:-8192}"
  --command-timeout-sec "${OFFICEQA_COMMAND_TIMEOUT_SEC:-30}"
  --max-observation-chars "${OFFICEQA_MAX_OBSERVATION_CHARS:-12000}"
  --temperature "${OFFICEQA_TEMPERATURE:-${DEFAULT_TEMPERATURE}}"
  --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE:-1}"
  --pipeline-parallel-size "${VLLM_PIPELINE_PARALLEL_SIZE:-1}"
  # Episodes rotate across replicas, then remain pinned to one DP rank so the
  # growing conversation prefix stays in that replica's prefix cache.
  --data-parallel-size "${VLLM_DATA_PARALLEL_SIZE:-${DEFAULT_DATA_PARALLEL_SIZE}}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
  # This limit is per replica. The async engines batch requests routed to them.
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-16}"
)

if [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
  ARGS+=(--max-model-len "${VLLM_MAX_MODEL_LEN}")
fi

if [[ -n "${VLLM_SAFETENSORS_LOAD_STRATEGY:-}" ]]; then
  ARGS+=(--safetensors-load-strategy "${VLLM_SAFETENSORS_LOAD_STRATEGY}")
fi

if [[ -n "${VLLM_GDN_PREFILL_BACKEND:-}" ]]; then
  ARGS+=(--gdn-prefill-backend "${VLLM_GDN_PREFILL_BACKEND}")
fi

if [[ "${VLLM_ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
  ARGS+=(--enable-prefix-caching)
elif [[ "${VLLM_ENABLE_PREFIX_CACHING:-}" == "0" ]]; then
  ARGS+=(--no-enable-prefix-caching)
fi

if [[ "${OFFICEQA_ENABLE_THINKING:-}" == "1" ]]; then
  ARGS+=(--enable-thinking)
elif [[ "${OFFICEQA_ENABLE_THINKING:-}" == "0" ]]; then
  ARGS+=(--no-enable-thinking)
fi

if [[ "${VLLM_ENABLE_EXPERT_PARALLEL:-0}" == "1" ]]; then
  ARGS+=(--enable-expert-parallel)
fi

if [[ "${OFFICEQA_ENABLE_VISUAL_PDF_TOOL:-0}" == "1" ]]; then
  : "${OFFICEQA_PDF_DIR:?Set OFFICEQA_PDF_DIR when enabling the visual PDF tool}"
  ARGS+=(
    --enable-visual-pdf-tool
    --pdf-dir "${OFFICEQA_PDF_DIR}"
    --pdf-render-dpi "${OFFICEQA_PDF_RENDER_DPI:-144}"
  )
  if [[ -n "${OFFICEQA_PAGE_TEXT_DIR:-}" ]]; then
    ARGS+=(--page-text-dir "${OFFICEQA_PAGE_TEXT_DIR}")
  fi
fi

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${OFFICEQA_UID:-}" ]]; then
  RUN_ID="${OFFICEQA_RUN_ID:-vllm-${MODEL_TAG}-${OFFICEQA_UID}-$(date +%Y%m%d-%H%M%S)}"
  OUTPUT="${OFFICEQA_OUTPUT:-${RESULTS_DIR}/${OFFICEQA_UID}.json}"
  ARGS+=(--uid "${OFFICEQA_UID}" --run-id "${RUN_ID}" --output "${OUTPUT}")
  exec "${PYTHON_BIN}" "${ARGS[@]}"
fi

ARGS+=(
  --output-dir "${RESULTS_DIR}"
  --concurrency "${OFFICEQA_CONCURRENCY:-${DEFAULT_CONCURRENCY}}"
  --failure-retries "${OFFICEQA_FAILURE_RETRIES:-2}"
  --failure-retry-delay-sec "${OFFICEQA_FAILURE_RETRY_DELAY_SEC:-60}"
  # >1 runs pass@k: independent attempts per question; requires
  # OFFICEQA_TEMPERATURE > 0 so attempts differ.
  --samples-per-task "${OFFICEQA_SAMPLES_PER_TASK:-${DEFAULT_SAMPLES_PER_TASK}}"
)
mkdir -p "${RESULTS_DIR}"
LOG_FILE="${RESULTS_DIR}/eval-$(date +%Y%m%d-%H%M%S).log"
echo "Full-dataset evaluation of ${MODEL}; logging to ${LOG_FILE}"
"${PYTHON_BIN}" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
