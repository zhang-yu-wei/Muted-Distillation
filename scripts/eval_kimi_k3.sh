#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/eval_paths.sh"

# Launch defaults; every value can still be overridden from the environment.
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
OFFICEQA_DATASET_CSV="${OFFICEQA_DATASET_CSV:-${ROOT_DIR}/data/officeqa_pro.csv}"
OFFICEQA_CORPUS_DIR="${OFFICEQA_CORPUS_DIR:-${ROOT_DIR}/data/treasury_bulletins_parsed/transformed}"
OFFICEQA_SANDBOX_IMAGE="${OFFICEQA_SANDBOX_IMAGE:-python:3.11-slim}"
MODEL="${KIMI_MODEL:-kimi-k3}"
RESULTS_DIR="${OFFICEQA_OUTPUT_DIR:-${EVAL_RESULTS_ROOT}/kimi-k3}"
WORKSPACES="${OFFICEQA_WORKSPACE_ROOT:-${RESULTS_DIR}/workspaces}"

# The Moonshot API authenticates with a single key; there is no key pool.
if [[ -z "${MOONSHOT_API_KEY:-}" ]]; then
  echo "Set MOONSHOT_API_KEY for Kimi evaluation" >&2
  exit 1
fi
# OFFICEQA_UID is optional: set it to evaluate one question; leave it unset to
# evaluate the whole dataset with per-UID results, resume, and a summary.

ARGS=(
  "${ROOT_DIR}/scripts/run_full_corpus_inference.py"
  --generation-backend kimi
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
  --kimi-base-url "${KIMI_BASE_URL:-https://api.moonshot.ai/v1}"
  --kimi-max-retries "${KIMI_MAX_RETRIES:-3}"
  --kimi-quota-retries "${KIMI_QUOTA_RETRIES:-10}"
  --kimi-reasoning-effort "${KIMI_REASONING_EFFORT:-max}"
)

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${OFFICEQA_UID:-}" ]]; then
  RUN_ID="${OFFICEQA_RUN_ID:-kimi-k3-${OFFICEQA_UID}-$(date +%Y%m%d-%H%M%S)}"
  OUTPUT="${OFFICEQA_OUTPUT:-${RESULTS_DIR}/${OFFICEQA_UID}.json}"
  ARGS+=(--uid "${OFFICEQA_UID}" --run-id "${RUN_ID}" --output "${OUTPUT}")
  exec "${PYTHON_BIN}" "${ARGS[@]}"
fi

ARGS+=(
  --output-dir "${RESULTS_DIR}"
  --concurrency "${OFFICEQA_CONCURRENCY:-4}"
  --failure-retries "${OFFICEQA_FAILURE_RETRIES:-2}"
  --failure-retry-delay-sec "${OFFICEQA_FAILURE_RETRY_DELAY_SEC:-60}"
)
mkdir -p "${RESULTS_DIR}"
LOG_FILE="${RESULTS_DIR}/eval-$(date +%Y%m%d-%H%M%S).log"
echo "Full-dataset evaluation; logging to ${LOG_FILE}"
"${PYTHON_BIN}" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
