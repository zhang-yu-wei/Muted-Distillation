#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Step 5 of the SFT data pipeline: generate teacher agent traces over the
# synthetic question set with a self-hosted OpenAI-compatible server (the
# GLM-5.2 vLLM cluster). Same runner and sandbox path as the eval launchers;
# only the dataset, output layout, and defaults differ.
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
OFFICEQA_DATASET_CSV="${OFFICEQA_DATASET_CSV:-${ROOT_DIR}/data/synthetic/officeqa_synth_author_v1.csv}"
OFFICEQA_CORPUS_DIR="${OFFICEQA_CORPUS_DIR:-${ROOT_DIR}/data/treasury_bulletins_parsed/transformed}"
OFFICEQA_SANDBOX_IMAGE="${OFFICEQA_SANDBOX_IMAGE:-python:3.11-slim}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"

# OPENAI_MODEL is optional: an unset model is discovered from the server's
# /models listing (single-model vLLM servers expose exactly one id).
if [[ -z "${OPENAI_MODEL:-}" ]]; then
  if ! OPENAI_MODEL="$(curl -fsS -m 10 "${OPENAI_BASE_URL}/models" \
      | "${PYTHON_BIN}" -c 'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])')"; then
    echo "Could not discover a model from ${OPENAI_BASE_URL}/models;" \
      "check the server or set OPENAI_MODEL explicitly" >&2
    exit 1
  fi
  echo "Discovered served model: ${OPENAI_MODEL}"
fi

# Trace directories carry both the dataset and teacher tags so a new synthetic
# CSV version or a different teacher never cross-resumes an older run.
MODEL_TAG="$(basename "${OPENAI_MODEL}")"
DATASET_TAG="$(basename "${OFFICEQA_DATASET_CSV}" .csv)"
RESULTS_DIR="${OFFICEQA_OUTPUT_DIR:-${ROOT_DIR}/results/sft-traces-${DATASET_TAG}-${MODEL_TAG}}"
WORKSPACES="${OFFICEQA_WORKSPACE_ROOT:-${RESULTS_DIR}/workspaces}"

# OFFICEQA_UID is optional: set it to trace one question; leave it unset to
# trace the whole synthetic set with per-UID results, resume, and a summary.

ARGS=(
  "${ROOT_DIR}/scripts/run_full_corpus_inference.py"
  --generation-backend openai-server
  --model "${OPENAI_MODEL}"
  --dataset-csv "${OFFICEQA_DATASET_CSV}"
  --corpus-dir "${OFFICEQA_CORPUS_DIR}"
  --workspace-root "${WORKSPACES}"
  --sandbox-runtime docker
  --sandbox-image "${OFFICEQA_SANDBOX_IMAGE}"
  --step-limit "${OFFICEQA_STEP_LIMIT:-50}"
  --max-tokens "${OFFICEQA_MAX_TOKENS:-8192}"
  --command-timeout-sec "${OFFICEQA_COMMAND_TIMEOUT_SEC:-30}"
  --max-observation-chars "${OFFICEQA_MAX_OBSERVATION_CHARS:-12000}"
  --temperature "${OFFICEQA_TEMPERATURE:-0.0}"
  --openai-base-url "${OPENAI_BASE_URL}"
  --openai-max-retries "${OPENAI_MAX_RETRIES:-3}"
  --openai-quota-retries "${OPENAI_QUOTA_RETRIES:-10}"
  --openai-request-timeout-sec "${OPENAI_REQUEST_TIMEOUT_SEC:-600}"
)

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${OFFICEQA_UID:-}" ]]; then
  RUN_ID="${OFFICEQA_RUN_ID:-sft-${MODEL_TAG}-${OFFICEQA_UID}-$(date +%Y%m%d-%H%M%S)}"
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
LOG_FILE="${RESULTS_DIR}/traces-$(date +%Y%m%d-%H%M%S).log"
echo "SFT trace generation: ${OFFICEQA_DATASET_CSV} with ${OPENAI_MODEL} at ${OPENAI_BASE_URL}; logging to ${LOG_FILE}"
"${PYTHON_BIN}" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
