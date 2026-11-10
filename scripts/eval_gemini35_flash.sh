#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/eval_paths.sh"

# Launch defaults; every value can still be overridden from the environment.
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
OFFICEQA_DATASET_CSV="${OFFICEQA_DATASET_CSV:-${ROOT_DIR}/data/officeqa_pro.csv}"
OFFICEQA_CORPUS_DIR="${OFFICEQA_CORPUS_DIR:-${ROOT_DIR}/data/treasury_bulletins_parsed/transformed}"
OFFICEQA_SANDBOX_IMAGE="${OFFICEQA_SANDBOX_IMAGE:-python:3.11-slim}"
MODEL="${GEMINI_MODEL:-gemini-3.5-flash}"
RESULTS_DIR="${OFFICEQA_OUTPUT_DIR:-${EVAL_RESULTS_ROOT}/gemini35-flash-keypool}"
WORKSPACES="${OFFICEQA_WORKSPACE_ROOT:-${RESULTS_DIR}/workspaces}"

# Provide one of, in order of precedence: GEMINI_API_KEYS_FILE (path to a
# text file with one key per line), GEMINI_API_KEYS (comma-separated pool),
# or a single GOOGLE_API_KEY. A pool is rotated round-robin across projects
# to spread tier-1 rate limits.
if [[ -n "${GEMINI_API_KEYS_FILE:-}" && ! -r "${GEMINI_API_KEYS_FILE}" ]]; then
  echo "GEMINI_API_KEYS_FILE is set but not readable: ${GEMINI_API_KEYS_FILE}" >&2
  exit 1
fi
if [[ -z "${GEMINI_API_KEYS_FILE:-}" && -z "${GEMINI_API_KEYS:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "Set GEMINI_API_KEYS_FILE, GEMINI_API_KEYS, or GOOGLE_API_KEY for Gemini evaluation" >&2
  exit 1
fi
# OFFICEQA_UID is optional: set it to evaluate one question; leave it unset to
# evaluate the whole dataset with per-UID results, resume, and a summary.

ARGS=(
  "${ROOT_DIR}/scripts/run_full_corpus_inference.py"
  --generation-backend gemini
  --model "${MODEL}"
  --dataset-csv "${OFFICEQA_DATASET_CSV}"
  --corpus-dir "${OFFICEQA_CORPUS_DIR}"
  --workspace-root "${WORKSPACES}"
  --sandbox-runtime docker
  --sandbox-image "${OFFICEQA_SANDBOX_IMAGE}"
  --step-limit "${OFFICEQA_STEP_LIMIT:-50}"
  --max-tokens "${OFFICEQA_MAX_TOKENS:-8192}"
  --command-timeout-sec "${OFFICEQA_COMMAND_TIMEOUT_SEC:-30}"
  --max-observation-chars "${OFFICEQA_MAX_OBSERVATION_CHARS:-12000}"
  --temperature "${OFFICEQA_TEMPERATURE:-1.0}"
  --gemini-api-version "${GEMINI_API_VERSION:-v1beta}"
  --gemini-max-retries "${GEMINI_MAX_RETRIES:-3}"
  --gemini-quota-retries "${GEMINI_QUOTA_RETRIES:-10}"
  --gemini-thinking-budget "${GEMINI_THINKING_BUDGET:-0}"
)

if [[ "${GEMINI_INCLUDE_THOUGHTS:-false}" == "true" ]]; then
  ARGS+=(--gemini-include-thoughts)
fi

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${OFFICEQA_UID:-}" ]]; then
  RUN_ID="${OFFICEQA_RUN_ID:-gemini35-flash-${OFFICEQA_UID}-$(date +%Y%m%d-%H%M%S)}"
  OUTPUT="${OFFICEQA_OUTPUT:-${RESULTS_DIR}/${OFFICEQA_UID}.json}"
  ARGS+=(--uid "${OFFICEQA_UID}" --run-id "${RUN_ID}" --output "${OUTPUT}")
  exec "${PYTHON_BIN}" "${ARGS[@]}"
fi

ARGS+=(
  --output-dir "${RESULTS_DIR}"
  --concurrency "${OFFICEQA_CONCURRENCY:-8}"
  --failure-retries "${OFFICEQA_FAILURE_RETRIES:-2}"
  --failure-retry-delay-sec "${OFFICEQA_FAILURE_RETRY_DELAY_SEC:-60}"
  # >1 runs pass@k: independent attempts per question; requires
  # OFFICEQA_TEMPERATURE > 0 so attempts differ.
  --samples-per-task "${OFFICEQA_SAMPLES_PER_TASK:-1}"
)
mkdir -p "${RESULTS_DIR}"
LOG_FILE="${RESULTS_DIR}/eval-$(date +%Y%m%d-%H%M%S).log"
echo "Full-dataset evaluation; logging to ${LOG_FILE}"
"${PYTHON_BIN}" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
