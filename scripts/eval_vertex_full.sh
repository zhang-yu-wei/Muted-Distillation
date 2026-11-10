#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/eval_paths.sh"

# Full-dataset OfficeQA evaluation against an SGLang or vLLM server behind a
# Vertex AI endpoint. Same
# runner and Docker sandbox path as the other eval_* launchers; every value
# can be overridden from the environment.
#
# Requirements on this machine:
#   * ADC with aiplatform access: `gcloud auth application-default login`
#     (the GCE metadata service account has read-only scopes and cannot call
#     rawPredict).
#   * The repo venv with `google-auth` and `transformers` importable
#     (`uv sync --all-extras`).
#
# CAUTION: the default endpoint doubles as the OPD teacher during training.
# Evaluation traffic (full generation, not just scoring) competes with
# teacher scoring and the deployment has a history of crash-restarts under
# load — prefer running evals while training is down, and keep
# OFFICEQA_CONCURRENCY modest. The generous VERTEX_QUOTA_RETRIES default
# rides out the endpoint's ~5-20 min restart windows.
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
OFFICEQA_DATASET_CSV="${OFFICEQA_DATASET_CSV:-${ROOT_DIR}/data/officeqa_pro.csv}"
OFFICEQA_CORPUS_DIR="${OFFICEQA_CORPUS_DIR:-${ROOT_DIR}/data/treasury_bulletins_parsed/transformed}"
OFFICEQA_SANDBOX_IMAGE="${OFFICEQA_SANDBOX_IMAGE:-python:3.11-slim}"

VERTEX_PROJECT="${VERTEX_PROJECT:-}"
VERTEX_LOCATION="${VERTEX_LOCATION:-us-east1}"
VERTEX_ENDPOINT_ID="${VERTEX_ENDPOINT_ID:-}"
VERTEX_URL="${VERTEX_URL:-}"                       # full URL override (also accepts http://host:port/generate)
VERTEX_PROTOCOL="${VERTEX_PROTOCOL:-sglang}"
VERTEX_TOKENIZER="${VERTEX_TOKENIZER:-/mnt/Qwen3.5-4B}"
# Names the results directory: results/vertex-<tag>/. Set it to the served
# model's name so different deployments on the same endpoint do not
# cross-resume each other's partial results.
VERTEX_MODEL_NAME="${VERTEX_MODEL_NAME:-endpoint-${VERTEX_ENDPOINT_ID}}"

# Optional reasoning-effort instruction, prepended to every prompt as its own
# leading system block — the same context layout the OPD teacher sees during
# training scoring, so this measures the pure teacher under an instruction.
#   OFFICEQA_INSTRUCTION=low    built-in low-resource text (matches OPD_INSTRUCTION=low)
#   OFFICEQA_INSTRUCTION=high   built-in high-resource text (matches OPD_INSTRUCTION_HIGH=high)
#   OFFICEQA_INSTRUCTION="..."  custom text
# Empty (default) leaves the eval exactly as before. The default results dir
# gets an -inst-* suffix so instruction runs never cross-resume with the
# plain eval of the same deployment.
OFFICEQA_INSTRUCTION="${OFFICEQA_INSTRUCTION:-}"
INSTRUCTION_TAG=""
if [[ "${OFFICEQA_INSTRUCTION}" == "low" ]]; then
  OFFICEQA_INSTRUCTION="Reasoning Effort: Minimum viable, shortcuts strongly encouraged. Be maximally efficient in this session: keep any thinking brief, run as few commands and tool calls as possible, skip redundant reasoning or verifications, and produce the final answer with the fewest output tokens that still solve the task."
  INSTRUCTION_TAG="-inst-low"
elif [[ "${OFFICEQA_INSTRUCTION}" == "high" ]]; then
  OFFICEQA_INSTRUCTION="Reasoning Effort: Absolute maximum with no shortcuts permitted. You MUST be very thorough in this session: think through every step in detail, run as many commands and tool calls as needed, double-check intermediate results, and only give the final answer after exhaustive verification."
  INSTRUCTION_TAG="-inst-high"
elif [[ -n "${OFFICEQA_INSTRUCTION}" ]]; then
  INSTRUCTION_TAG="-inst-$(printf %s "${OFFICEQA_INSTRUCTION}" | md5sum | cut -c1-8)"
fi

RESULTS_DIR="${OFFICEQA_OUTPUT_DIR:-${EVAL_RESULTS_ROOT}/vertex-${VERTEX_MODEL_NAME}${INSTRUCTION_TAG}}"
WORKSPACES="${OFFICEQA_WORKSPACE_ROOT:-${RESULTS_DIR}/workspaces}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Preflight: one 1-token generation before the sweep. Catches missing ADC,
# wrong endpoint id, and a down model server here rather than 133 times in
# the run log.
if [[ "${VERTEX_PREFLIGHT:-1}" == "1" ]]; then
  VERTEX_URL="${VERTEX_URL}" VERTEX_PROJECT="${VERTEX_PROJECT}" \
  VERTEX_LOCATION="${VERTEX_LOCATION}" VERTEX_ENDPOINT_ID="${VERTEX_ENDPOINT_ID}" \
  VERTEX_PROTOCOL="${VERTEX_PROTOCOL}" VERTEX_TOKENIZER="${VERTEX_TOKENIZER}" \
  "${PYTHON_BIN}" - <<'EOF_PREFLIGHT'
import os, sys
from officeqa_rl.backends.vertex import VertexSGLangGenerationBackend, rawpredict_url

url = os.environ["VERTEX_URL"] or rawpredict_url(
    os.environ["VERTEX_PROJECT"], os.environ["VERTEX_LOCATION"], os.environ["VERTEX_ENDPOINT_ID"]
)
backend = VertexSGLangGenerationBackend(
    "preflight", base_url=url, tokenizer_path=os.environ["VERTEX_TOKENIZER"],
    protocol=os.environ["VERTEX_PROTOCOL"],
    max_tokens=1, max_retries=1, quota_max_retries=1, request_timeout_sec=60.0,
)
print(f"Vertex preflight -> {url}", flush=True)
try:
    backend._generate_sync([{"role": "user", "content": "ping"}], {"temperature": 0.0})
except Exception as error:
    print(f"Vertex preflight FAILED: {type(error).__name__}: {error}", file=sys.stderr)
    print("Check `gcloud auth application-default login`, the endpoint id,", file=sys.stderr)
    print("and that the model server is up. VERTEX_PREFLIGHT=0 skips this.", file=sys.stderr)
    sys.exit(1)
print("Vertex preflight OK", flush=True)
EOF_PREFLIGHT
fi

ARGS=(
  "${ROOT_DIR}/scripts/run_full_corpus_inference.py"
  --generation-backend vertex
  --corpus-profile "${OFFICEQA_CORPUS_PROFILE:-v1}"
  --model "${VERTEX_MODEL_NAME}"
  --vertex-project "${VERTEX_PROJECT}"
  --vertex-location "${VERTEX_LOCATION}"
  --vertex-endpoint-id "${VERTEX_ENDPOINT_ID}"
  --vertex-protocol "${VERTEX_PROTOCOL}"
  --vertex-tokenizer "${VERTEX_TOKENIZER}"
  --vertex-max-retries "${VERTEX_MAX_RETRIES:-3}"
  --vertex-quota-retries "${VERTEX_QUOTA_RETRIES:-15}"
  --vertex-request-timeout-sec "${VERTEX_REQUEST_TIMEOUT_SEC:-600}"
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
)
if [[ -n "${VERTEX_URL}" ]]; then
  ARGS+=(--vertex-url "${VERTEX_URL}")
fi
if [[ -n "${OFFICEQA_INSTRUCTION}" ]]; then
  ARGS+=(--vertex-instruction "${OFFICEQA_INSTRUCTION}")
  echo "Instruction block active (${INSTRUCTION_TAG#-}): ${OFFICEQA_INSTRUCTION:0:80}..."
fi

# OFFICEQA_UID is optional: set it to evaluate one question; leave it unset
# to evaluate the whole dataset with per-UID results, resume, and a summary.
if [[ -n "${OFFICEQA_UID:-}" ]]; then
  RUN_ID="${OFFICEQA_RUN_ID:-vertex-${VERTEX_MODEL_NAME}-${OFFICEQA_UID}-$(date +%Y%m%d-%H%M%S)}"
  OUTPUT="${OFFICEQA_OUTPUT:-${RESULTS_DIR}/${OFFICEQA_UID}.json}"
  ARGS+=(--uid "${OFFICEQA_UID}" --run-id "${RUN_ID}" --output "${OUTPUT}")
  exec "${PYTHON_BIN}" "${ARGS[@]}"
fi

ARGS+=(
  --output-dir "${RESULTS_DIR}"
  --concurrency "${OFFICEQA_CONCURRENCY:-4}"
  --failure-retries "${OFFICEQA_FAILURE_RETRIES:-2}"
  --failure-retry-delay-sec "${OFFICEQA_FAILURE_RETRY_DELAY_SEC:-60}"
  # >1 runs pass@k: independent attempts per question; requires
  # OFFICEQA_TEMPERATURE > 0 so attempts differ.
  --samples-per-task "${OFFICEQA_SAMPLES_PER_TASK:-1}"
  # Split one dataset across several endpoints: each shard runs its own
  # process against its own endpoint and writes its own results dir.
  --num-shards "${OFFICEQA_NUM_SHARDS:-1}"
  --shard-index "${OFFICEQA_SHARD_INDEX:-0}"
)
mkdir -p "${RESULTS_DIR}"
LOG_FILE="${RESULTS_DIR}/eval-$(date +%Y%m%d-%H%M%S).log"
echo "Full-dataset evaluation of ${VERTEX_MODEL_NAME} via Vertex endpoint ${VERTEX_ENDPOINT_ID} (${VERTEX_LOCATION}); logging to ${LOG_FILE}"
"${PYTHON_BIN}" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
