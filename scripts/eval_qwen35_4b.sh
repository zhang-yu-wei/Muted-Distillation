#!/usr/bin/env bash
set -euo pipefail

# Required inputs for a selected-UID evaluation smoke test.
: "${OFFICEQA_DATASET_CSV:?Set OFFICEQA_DATASET_CSV to officeqa_pro.csv or officeqa_full.csv}"
: "${OFFICEQA_UID:?Set OFFICEQA_UID to the single evaluation UID}"
: "${OFFICEQA_CORPUS_DIR:?Set OFFICEQA_CORPUS_DIR to the transformed parsed corpus}"
: "${OFFICEQA_SANDBOX_IMAGE:?Set OFFICEQA_SANDBOX_IMAGE to the OfficeQA sandbox image}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/eval_paths.sh"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
# Launching .venv/bin/python directly does not activate the venv, so tools
# installed next to the interpreter (ninja for vLLM's JIT kernel builds) are
# not on PATH unless we put them there.
if [[ "${PYTHON_BIN}" == */* ]]; then
  export PATH="$(cd -- "$(dirname -- "${PYTHON_BIN}")" && pwd):${PATH}"
fi
MODEL="${QWEN35_4B_MODEL:-/mnt/Qwen3.5-4B}"
RUN_ID="${OFFICEQA_RUN_ID:-qwen35-4b-${OFFICEQA_UID}-$(date +%Y%m%d-%H%M%S)}"
OUTPUT="${OFFICEQA_OUTPUT:-${EVAL_RESULTS_ROOT}/qwen35-4b/${OFFICEQA_UID}.json}"
WORKSPACES="${OFFICEQA_WORKSPACE_ROOT:-${EVAL_RESULTS_ROOT}/qwen35-4b/workspaces}"

ARGS=(
  "${ROOT_DIR}/scripts/run_full_corpus_inference.py"
  --generation-backend vllm
  --model "${MODEL}"
  --dataset-csv "${OFFICEQA_DATASET_CSV}"
  --uid "${OFFICEQA_UID}"
  --corpus-dir "${OFFICEQA_CORPUS_DIR}"
  --workspace-root "${WORKSPACES}"
  --sandbox-runtime docker
  --sandbox-image "${OFFICEQA_SANDBOX_IMAGE}"
  --run-id "${RUN_ID}"
  --output "${OUTPUT}"
  --step-limit "${OFFICEQA_STEP_LIMIT:-50}"
  --max-tokens "${OFFICEQA_MAX_TOKENS:-8192}"
  --command-timeout-sec "${OFFICEQA_COMMAND_TIMEOUT_SEC:-30}"
  --max-observation-chars "${OFFICEQA_MAX_OBSERVATION_CHARS:-12000}"
  --temperature "${OFFICEQA_TEMPERATURE:-1.0}"
  --tensor-parallel-size "${QWEN35_TENSOR_PARALLEL_SIZE:-1}"
  --pipeline-parallel-size "${QWEN35_PIPELINE_PARALLEL_SIZE:-1}"
  --gpu-memory-utilization "${QWEN35_GPU_MEMORY_UTILIZATION:-0.85}"
  --max-num-seqs "${QWEN35_MAX_NUM_SEQS:-1}"
)

if [[ -n "${QWEN35_MAX_MODEL_LEN:-}" ]]; then
  ARGS+=(--max-model-len "${QWEN35_MAX_MODEL_LEN}")
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
exec "${PYTHON_BIN}" "${ARGS[@]}"
