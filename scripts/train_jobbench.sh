#!/bin/bash
# JobBench agent RL launcher — single node.
#
# Sibling of scripts/train_ssb1.sh with the JobBench rollout hook
# (officeqa_rl.training.jobbench_rollout.generate) instead of the
# SpreadsheetBench one; the OfficeQA and SpreadsheetBench scripts and modules
# are untouched. The agent is the vendored mini-swe-agent DefaultAgent
# (third_party/mini-swe-agent), not the OfficeQA agent loop.
#
# Training rows are the 63 easy-split tasks (every referenced file sits inside
# the task folder, so an episode is solvable without network access); periodic
# eval runs the 65-task main split, which is what the public JobBench
# leaderboard reports. The reward is an LLM judge over each task's weighted
# rubric — one call per rubric item, ~9 per episode — normalized to [0, 1],
# the same number the official judge produces.
#
# Be careful before trusting a curve: an
# LLM-judge reward is neither stationary nor hard to game, and a deliverable
# that asserts the right sentences without doing the work can score.
#
# Prerequisites (single node):
#   * data/slime/jobbench_train.jsonl + jobbench_eval.jsonl — build with
#       python3 scripts/build_jobbench_dataset.py
#   * the jobbench-sandbox image on the HOST Docker daemon — build with
#       docker build -t jobbench-sandbox docker/jobbench
#   * a judge API key exported as $JUDGE_API_KEY (see JUDGE_* below)
#   * everything train_officeqa_opd.sh needs (model checkpoints, FLA prewarm).
#
# Modes:
#   bash scripts/train_jobbench.sh rollout-debug   # 1 rollout incl. judging, no training
#   bash scripts/train_jobbench.sh train-smoke     # exactly 1 GRPO iteration
#   bash scripts/train_jobbench.sh full            # full run with periodic main-split eval
#   bash scripts/train_jobbench.sh full-async      # fully-async rollout, disjoint GPUs
#
# The train_officeqa.sh hyperparameter knobs apply with the same defaults (LR,
# ROLLOUT_BS, N_SAMPLES, GLOBAL_BS, NUM_EPOCH, MAX_CTX_LEN, MAX_RESP_LEN,
# ADV_ESTIMATOR, TEMPERATURE, KL_LOSS_COEF, EPS_CLIP, EPS_CLIP_HIGH,
# ENTROPY_COEF, DYNAMIC_FILTER, OVER_SAMPLING_BS, TP_SIZE, CP_SIZE), plus:
#   AGENT_STEP_LIMIT (60)   per-episode step cap -> officeqa_jobbench_step_limit
#                           (the offline eval default; empty keeps the yaml value)
#   EVAL_INTERVAL    (5)    rollouts between main-split evals. Each eval runs
#                           65 episodes and ~570 judge calls; raise this if
#                           eval wall-clock or judge spend dominates.
#   EVAL_SAMPLES     (1)    --n-samples-per-eval-prompt
#   JUDGE_API_KEY_ENV (JUDGE_API_KEY)  env var holding the judge credential
#   JUDGE_MODEL       (yaml value)     judge model override
#   JUDGE_API_BASE    (yaml value)     judge endpoint override
#
# With only 63 training tasks, ROLLOUT_BS above ~32 repeats prompts within an
# epoch. Prefer raising N_SAMPLES (more attempts per task, which GRPO wants
# anyway) over raising ROLLOUT_BS.
#
# RL+OPD is OPTIONAL here (default plain GRPO). USE_OPD=1 enables it with the
# same teacher knobs as train_officeqa_opd.sh (OPD_TEACHER_URL or
# OPD_PROJECT/OPD_LOCATION/OPD_ENDPOINT_ID, OPD_KL_COEF, OPD_CONCURRENCY,
# OPD_TIMEOUT_SEC, OPD_ATTEMPTS, OPD_BACKOFF_SEC, OPD_PREFLIGHT,
# OPD_TEACHER_FAILURE_LOG). Only per-sample scoring is wired here; the
# per-turn hill-climb engine is not.

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

# --------------------------------------------------------- JobBench data
# JOBBENCH_DATA_PREFIX selects which built dataset to use. The default pair is
# easy-train / main-eval; build a swapped pair with
#   python3 scripts/build_jobbench_dataset.py --no-download \
#     --train-split main --eval-split easy --output-prefix jobbench_mtrain
# and launch with JOBBENCH_DATA_PREFIX=jobbench_mtrain.
JOBBENCH_DATA_PREFIX=${JOBBENCH_DATA_PREFIX:-jobbench}
TRAIN_DATA="${REPO}/data/slime/${JOBBENCH_DATA_PREFIX}_train.jsonl"
EVAL_DATA="${REPO}/data/slime/${JOBBENCH_DATA_PREFIX}_eval.jsonl"
# Which split trains and which evaluates is a property of the BUILT DATA, not
# of this launcher, so read it back from the manifest the builder wrote. Every
# default below keys off it; guessing from the prefix name would let the two
# disagree silently.
JOBBENCH_MANIFEST="${REPO}/data/slime/${JOBBENCH_DATA_PREFIX}_manifest.json"
if [ -f "${JOBBENCH_MANIFEST}" ]; then
   TRAIN_SPLIT_NAME=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['policy']['train_split'])" "${JOBBENCH_MANIFEST}" 2>/dev/null || echo "")
   EVAL_SPLIT_NAME=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['policy']['eval_split'])" "${JOBBENCH_MANIFEST}" 2>/dev/null || echo "")
fi
TRAIN_SPLIT_NAME=${TRAIN_SPLIT_NAME:-easy}
EVAL_SPLIT_NAME=${EVAL_SPLIT_NAME:-main}

# Label for the eval set in W&B / slime metrics; names the split actually used.
EVAL_SET_LABEL=${EVAL_SET_LABEL:-jobbench_${EVAL_SPLIT_NAME}}

# omit (the official runner's behaviour, leaderboard-comparable) | stage.
#
# 57 of the 65 MAIN tasks ship files_required_to_search/: documents the
# instructions reference but that a network-less sandbox cannot fetch. `easy`
# has none, so this only ever matters for whichever side holds main.
#   * main EVALUATES -> omit, so the number stays comparable with published
#     JobBench results.
#   * main TRAINS    -> stage, or those rubric items are unreachable BY
#     CONSTRUCTION and the policy is optimised against reward it cannot earn.
#     That failure is invisible: episodes run, the judge scores them, and the
#     ceiling is simply lower than it looks.
# The default therefore follows the manifest rather than a fixed value.
if [ "${TRAIN_SPLIT_NAME}" = "main" ]; then
   SEARCH_FILE_POLICY=${SEARCH_FILE_POLICY:-stage}
else
   SEARCH_FILE_POLICY=${SEARCH_FILE_POLICY:-omit}
fi
if [ "${TRAIN_SPLIT_NAME}" = "main" ] && [ "${SEARCH_FILE_POLICY}" = "omit" ]; then
   if [ "${JOBBENCH_ALLOW_UNREACHABLE_REWARD:-0}" != "1" ]; then
      echo "REFUSING TO LAUNCH: training on the main split with" >&2
      echo "SEARCH_FILE_POLICY=omit. 57/65 main tasks reference documents that" >&2
      echo "are not staged into the sandbox and cannot be fetched (no network)," >&2
      echo "so their rubric items are unreachable and training optimises against" >&2
      echo "a reward it cannot earn. Use SEARCH_FILE_POLICY=stage, or set" >&2
      echo "JOBBENCH_ALLOW_UNREACHABLE_REWARD=1 if that is deliberate." >&2
      exit 1
   fi
   echo "WARNING: training on main with unreachable search-file rubric items" >&2
   echo "(JOBBENCH_ALLOW_UNREACHABLE_REWARD=1)." >&2
fi
for path in "${TRAIN_DATA}" "${EVAL_DATA}"; do
   if [ ! -f "${path}" ]; then
      echo "JobBench slime dataset missing: ${path}" >&2
      echo "Build it first:  python3 scripts/build_jobbench_dataset.py" >&2
      exit 1
   fi
done
# Every row's metadata.data_root must exist here (and, under DooD, at the same
# path on the Docker host).
python3 - "${TRAIN_DATA}" "${EVAL_DATA}" <<'EOF_DATACHECK'
import json, pathlib, sys
for jsonl in sys.argv[1:]:
    roots = {json.loads(line)["metadata"]["data_root"] for line in open(jsonl)}
    for root in sorted(roots):
        if not (pathlib.Path(root) / "dataset.json").is_file():
            sys.exit(f"{jsonl}: metadata.data_root {root} has no dataset.json; "
                     "rerun scripts/build_jobbench_dataset.py on this machine")
print("JobBench data roots OK")
EOF_DATACHECK

# The episode container runs this image on the host daemon.
# Probe the CLI first so a broken DooD setup fails with its own message
# instead of masquerading as a missing image.
if ! docker version >/dev/null 2>&1; then
   echo "The docker CLI does not work in this environment:" >&2
   docker version 2>&1 | head -3 >&2 || true
   exit 1
fi
JOBBENCH_SANDBOX_IMAGE=${JOBBENCH_SANDBOX_IMAGE:-jobbench-sandbox}
if ! docker image inspect "${JOBBENCH_SANDBOX_IMAGE}" >/dev/null 2>&1; then
   echo "Sandbox image not found on the host Docker daemon: ${JOBBENCH_SANDBOX_IMAGE}" >&2
   echo "Build it first:  docker build -t ${JOBBENCH_SANDBOX_IMAGE} ${REPO}/docker/jobbench" >&2
   exit 1
fi

# ----------------------------------------------------------------- judge
# The reward is an LLM judge over each task's rubric, called from the rollout
# worker. Without a key every episode would abort with scoring_failure, so
# fail here instead of after the first rollout.
JUDGE_API_KEY_ENV=${JUDGE_API_KEY_ENV:-JUDGE_API_KEY}
JUDGE_MODEL=${JUDGE_MODEL:-}
JUDGE_API_BASE=${JUDGE_API_BASE:-}
if [ -z "${!JUDGE_API_KEY_ENV:-}" ]; then
   echo "No judge API key in \$${JUDGE_API_KEY_ENV}." >&2
   echo "JobBench rewards come from an LLM judge; export the key before launching:" >&2
   echo "  export ${JUDGE_API_KEY_ENV}=<key>            # judge endpoint credential" >&2
   echo "  export JUDGE_MODEL=grok-4.3                  # optional override" >&2
   echo "  export JUDGE_API_BASE=https://api.x.ai/v1    # optional override" >&2
   exit 1
fi

# ------------------------------------------------------------- OPD (optional)
USE_OPD=${USE_OPD:-0}
OPD_TYPE_DEF=sglang;     OPD_TYPE=${OPD_TYPE:-${OPD_TYPE_DEF}}
OPD_KL_COEF_DEF=1.0;     OPD_KL_COEF=${OPD_KL_COEF:-${OPD_KL_COEF_DEF}}
OPD_PROJECT=${OPD_PROJECT:-}
OPD_LOCATION=${OPD_LOCATION:-}
OPD_ENDPOINT_ID=${OPD_ENDPOINT_ID:-}
OPD_TEACHER_URL=${OPD_TEACHER_URL:-}
OPD_CONCURRENCY_DEF=16;  OPD_CONCURRENCY=${OPD_CONCURRENCY:-${OPD_CONCURRENCY_DEF}}
OPD_TIMEOUT_SEC_DEF=180; OPD_TIMEOUT_SEC=${OPD_TIMEOUT_SEC:-${OPD_TIMEOUT_SEC_DEF}}
OPD_ATTEMPTS_DEF=5;      OPD_ATTEMPTS=${OPD_ATTEMPTS:-${OPD_ATTEMPTS_DEF}}
OPD_BACKOFF_SEC_DEF=1.0; OPD_BACKOFF_SEC=${OPD_BACKOFF_SEC:-${OPD_BACKOFF_SEC_DEF}}
OPD_PREFLIGHT=${OPD_PREFLIGHT:-1}

if [ "${USE_OPD}" = "1" ]; then
   if [ "${OPD_TYPE}" != "sglang" ]; then
      echo "OPD_TYPE=${OPD_TYPE} is not supported: the JobBench hook only attaches" >&2
      echo "teacher log-probs for opd_type == 'sglang' (per-sample scoring)." >&2
      exit 1
   fi
   if [ "${OPD_SCORE_MODE:-per_sample}" != "per_sample" ]; then
      echo "OPD_SCORE_MODE=${OPD_SCORE_MODE} is not supported: the JobBench hook" >&2
      echo "wires only per-sample teacher scoring. The per-turn hill-climb engine" >&2
      echo "(officeqa_rl.training.opd_rollout) is not connected to this hook." >&2
      exit 1
   fi
   if [ -z "${OPD_TEACHER_URL}" ]; then
      if [ -z "${OPD_PROJECT}" ] || [ -z "${OPD_LOCATION}" ] || [ -z "${OPD_ENDPOINT_ID}" ]; then
         echo "USE_OPD=1 but the teacher endpoint is not configured. Set either:" >&2
         echo "  OPD_TEACHER_URL=<full url>   (also accepts a local http SGLang /generate)" >&2
         echo "or all three of OPD_PROJECT / OPD_LOCATION / OPD_ENDPOINT_ID." >&2
         exit 1
      fi
      case "${OPD_LOCATION}" in
      *-*-*)
         echo "OPD_LOCATION=${OPD_LOCATION} looks like a zone, not a Vertex location." >&2
         echo "Use a region (us-central1) or 'global'." >&2
         exit 1
         ;;
      esac
   fi
fi

# Key hyperparameters — same defaults as the OfficeQA launchers so runs stay
# comparable across benchmarks.
LR_DEF=1e-6;             LR=${LR:-${LR_DEF}}
ROLLOUT_BS_DEF=8;        ROLLOUT_BS=${ROLLOUT_BS:-${ROLLOUT_BS_DEF}}
N_SAMPLES_DEF=8;         N_SAMPLES=${N_SAMPLES:-${N_SAMPLES_DEF}}
GLOBAL_BS_DEF=$((ROLLOUT_BS * N_SAMPLES)); GLOBAL_BS=${GLOBAL_BS:-${GLOBAL_BS_DEF}}
NUM_EPOCH_DEF=10;        NUM_EPOCH=${NUM_EPOCH:-${NUM_EPOCH_DEF}}
MAX_CTX_LEN_DEF=65536;   MAX_CTX_LEN=${MAX_CTX_LEN:-${MAX_CTX_LEN_DEF}}
MAX_RESP_LEN_DEF=8192;   MAX_RESP_LEN=${MAX_RESP_LEN:-${MAX_RESP_LEN_DEF}}
# Per-episode agent step cap; explicitly empty (AGENT_STEP_LIMIT=) = yaml value.
AGENT_STEP_LIMIT_DEF=60; AGENT_STEP_LIMIT=${AGENT_STEP_LIMIT-${AGENT_STEP_LIMIT_DEF}}
EVAL_INTERVAL_DEF=5;     EVAL_INTERVAL=${EVAL_INTERVAL:-${EVAL_INTERVAL_DEF}}
EVAL_SAMPLES_DEF=1;      EVAL_SAMPLES=${EVAL_SAMPLES:-${EVAL_SAMPLES_DEF}}

# Loss shape / exploration knobs.
ADV_ESTIMATOR_DEF=grpo;  ADV_ESTIMATOR=${ADV_ESTIMATOR:-${ADV_ESTIMATOR_DEF}}  # grpo | gspo | cispo | ...
TEMPERATURE_DEF=1.0;     TEMPERATURE=${TEMPERATURE:-${TEMPERATURE_DEF}}
KL_LOSS_COEF_DEF=0.00;   KL_LOSS_COEF=${KL_LOSS_COEF:-${KL_LOSS_COEF_DEF}}     # nonzero enables --use-kl-loss
EPS_CLIP_DEF=0.2;        EPS_CLIP=${EPS_CLIP:-${EPS_CLIP_DEF}}
EPS_CLIP_HIGH=${EPS_CLIP_HIGH:-}           # e.g. 0.28 for DAPO clip-higher; empty = symmetric
ENTROPY_COEF_DEF=0.00;   ENTROPY_COEF=${ENTROPY_COEF:-${ENTROPY_COEF_DEF}}

# DAPO-style dynamic sampling (full mode only; the fully-async rollout worker
# ignores the filter). The execution-level filter also drops the JobBench tasks a
# group scores identically on (e.g. all-zero), which is where soft restriction
# helps: partial case successes create within-group contrast hard restriction
# would hide.
DYNAMIC_FILTER_DEF=0;    DYNAMIC_FILTER=${DYNAMIC_FILTER:-${DYNAMIC_FILTER_DEF}}
OVER_SAMPLING_BS=${OVER_SAMPLING_BS:-}

# Training parallelism (see train_officeqa.sh for why CP=2 at ctx 64k).
if [ "${MODE}" = "rollout-debug" ]; then
   TP_SIZE=${TP_SIZE:-1}
   CP_SIZE=${CP_SIZE:-1}
else
   TP_SIZE=${TP_SIZE:-2}
   CP_SIZE=${CP_SIZE:-2}
fi

TRAIN_MEM_MARGIN_BYTES=${TRAIN_MEM_MARGIN_BYTES:-1073741824}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-4096}
PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-garbage_collection_threshold:0.7}

nonzero() { [ "$(awk "BEGIN{print ($1 != 0) ? 1 : 0}")" = "1" ]; }

# _add <tag> <value> [<default>]: append -<tag><value> to EXP_NAME unless the
# value is empty or equals its default.
_add() {
   local tag=$1 val=$2 def=${3:-}
   if [ -n "${val}" ] && [ "${val}" != "${def}" ]; then
      EXP_NAME+="-${tag}${val}"
   fi
}

# Experiment name: the "jobbench" prefix keeps these checkpoints and W&B groups
# disjoint from every OfficeQA run.
if [ -z "${EXP_NAME:-}" ]; then
   EXP_NAME="qwen3.5-4B-jobbench-${ADV_ESTIMATOR}-${MODE}"
   if [ "${JOBBENCH_DATA_PREFIX}" != "jobbench" ]; then
      EXP_NAME+="-${JOBBENCH_DATA_PREFIX#jobbench_}"
   fi
   if [ "${SEARCH_FILE_POLICY}" != "omit" ]; then
      EXP_NAME+="-sf${SEARCH_FILE_POLICY}"
   fi
   if [ "${USE_OPD}" = "1" ]; then
      EXP_NAME+="-opdkl${OPD_KL_COEF}"
   fi
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

# Derived rollout config: the officeqa_jobbench_* keys have no CLI equivalent, so
# the hook reads them from the YAML slime merges onto args. Strip the keys
# this script owns from the base file so the values below win, and key the
# filename on EXP_NAME so two concurrent runs never share one derived config.
CUSTOM_CONFIG="${REPO}/configs/training/jobbench_rollout.yaml"
DERIVED_CONFIG="${REPO}/results/jobbench_rollout_${EXP_NAME}.yaml"
if [ ! -f "${CUSTOM_CONFIG}" ]; then
   echo "Base rollout config not found: ${CUSTOM_CONFIG}" >&2
   echo "REPO=${REPO} — is the repo mounted at that path inside the container?" >&2
   exit 1
fi
OPD_TEACHER_FAILURE_LOG=${OPD_TEACHER_FAILURE_LOG-${REPO}/results/opd_teacher_failures/${EXP_NAME}.jsonl}
# Must exist on the Docker HOST too (DooD mounts it); derive from REPO so
# non-canonical repo locations (COS nodes) work without editing the YAML.
WORKSPACE_ROOT=${WORKSPACE_ROOT:-${REPO}/results/rollout_workspaces}
mkdir -p "${REPO}/results"
{
   grep -v -E '^(officeqa_workspace_root|officeqa_jobbench_step_limit|officeqa_jobbench_sandbox_image|officeqa_jobbench_search_file_policy|officeqa_jobbench_judge_(model|api_base|api_key_env)|officeqa_opd_[a-z_]+):' "${CUSTOM_CONFIG}"
   echo "officeqa_jobbench_search_file_policy: ${SEARCH_FILE_POLICY}"
   echo "officeqa_workspace_root: ${WORKSPACE_ROOT}"
   if [ -n "${AGENT_STEP_LIMIT}" ]; then
      echo "officeqa_jobbench_step_limit: ${AGENT_STEP_LIMIT}"
   fi
   echo "officeqa_jobbench_sandbox_image: ${JOBBENCH_SANDBOX_IMAGE}"
   echo "officeqa_jobbench_judge_api_key_env: ${JUDGE_API_KEY_ENV}"
   if [ -n "${JUDGE_MODEL}" ]; then
      echo "officeqa_jobbench_judge_model: \"${JUDGE_MODEL}\""
   fi
   if [ -n "${JUDGE_API_BASE}" ]; then
      echo "officeqa_jobbench_judge_api_base: \"${JUDGE_API_BASE}\""
   fi
   if [ "${USE_OPD}" = "1" ]; then
      echo "# --- OPD teacher (written by scripts/train_jobbench.sh) ---"
      if [ -n "${OPD_TEACHER_URL}" ]; then
         echo "officeqa_opd_teacher_url: \"${OPD_TEACHER_URL}\""
      else
         echo "officeqa_opd_project: \"${OPD_PROJECT}\""
         echo "officeqa_opd_location: \"${OPD_LOCATION}\""
         echo "officeqa_opd_endpoint_id: \"${OPD_ENDPOINT_ID}\""
      fi
      echo "officeqa_opd_concurrency: ${OPD_CONCURRENCY}"
      echo "officeqa_opd_timeout_sec: ${OPD_TIMEOUT_SEC}"
      echo "officeqa_opd_attempts: ${OPD_ATTEMPTS}"
      echo "officeqa_opd_backoff_sec: ${OPD_BACKOFF_SEC}"
      if [ -n "${OPD_TEACHER_FAILURE_LOG}" ]; then
         echo "officeqa_opd_failure_log: \"${OPD_TEACHER_FAILURE_LOG}\""
      fi
   fi
} > "${DERIVED_CONFIG}"
CUSTOM_CONFIG="${DERIVED_CONFIG}"

# OPD preflight: one tiny scoring call before Ray, GPUs, or checkpoint
# conversion (same rationale and failure modes as train_officeqa_opd.sh).
if [ "${USE_OPD}" = "1" ] && [ "${OPD_PREFLIGHT}" = "1" ]; then
   PYTHONPATH="${REPO}/src:${SLIME_DIR}" OPD_CONFIG="${CUSTOM_CONFIG}" python3 - <<'EOF_PREFLIGHT'
import asyncio, os, sys, types, yaml
from officeqa_rl.training.vertex_teacher_client import get_teacher_client

args = types.SimpleNamespace(**(yaml.safe_load(open(os.environ["OPD_CONFIG"])) or {}))
client = get_teacher_client(args)
print(f"OPD preflight -> {client.teacher_url}", flush=True)

async def main():
    try:
        log_probs = await client.score_tokens([1, 2, 3, 4])
    finally:
        await client.close()
    assert len(log_probs) == 3, log_probs
    print(f"OPD preflight OK: scored {len(log_probs)} tokens", flush=True)

try:
    asyncio.run(main())
except Exception as error:
    print(f"OPD preflight FAILED: {type(error).__name__}: {error}", file=sys.stderr)
    print("Check the endpoint id/location, `gcloud auth application-default login`,", file=sys.stderr)
    print("and that the endpoint's predict route is SGLang /generate.", file=sys.stderr)
    print("Set OPD_PREFLIGHT=0 to launch anyway.", file=sys.stderr)
    sys.exit(1)
EOF_PREFLIGHT
fi

# rerun hygiene (single node). A crashed run orphans episode sandboxes and,
# before the named-replay fix, replay containers; sweep both so leaked memory
# from a previous run cannot bleed into this one.
ray stop --force || true
pkill -9 sglang || true
docker ps -q --filter name=officeqa-uid | xargs -r docker rm -f >/dev/null 2>&1 || true
docker ps -q --filter name=jobbench- | xargs -r docker rm -f >/dev/null 2>&1 || true
docker ps -q --filter "ancestor=${JOBBENCH_SANDBOX_IMAGE}" | xargs -r docker rm -f >/dev/null 2>&1 || true
sleep 2

source "${SLIME_DIR}/scripts/models/qwen3.5-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL}"
)

ROLLOUT_ARGS=(
   --prompt-data "${TRAIN_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --custom-generate-function-path officeqa_rl.training.jobbench_rollout.generate
   --custom-config-path "${CUSTOM_CONFIG}"
   --custom-rollout-log-function-path officeqa_rl.training.jobbench_metrics.log_rollout_extra_metrics
   --custom-eval-rollout-log-function-path officeqa_rl.training.jobbench_metrics.log_eval_extra_metrics
   --rollout-max-context-len "${MAX_CTX_LEN}"
   --rollout-max-response-len "${MAX_RESP_LEN}"
   --rollout-temperature "${TEMPERATURE}"
)

OPD_ARGS=()
if [ "${USE_OPD}" = "1" ]; then
   # OPD flags stay on the CLI: the YAML merge happens after slime's OPD
   # validation, so setting these in the yaml would bypass the checks.
   OPD_ARGS+=(
      --use-opd
      --opd-type "${OPD_TYPE}"
      --opd-kl-coef "${OPD_KL_COEF}"
   )
fi

EVAL_ARGS=()

if [ "${MODE}" != "rollout-debug" ]; then
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
   # One prompt, one sample, rollout only — exercises the full episode AND the
   # replay/recalc grading path end to end. Inspect with
   # scripts/inspect_rollout_dump.py (metadata.jobbench_metrics holds the scores).
   ROLLOUT_ARGS+=(
      --num-rollout 1
      --rollout-batch-size 1
      --n-samples-per-prompt 1
      --global-batch-size 1
      --debug-rollout-only
      --save-debug-rollout-data "${DUMP_ROOT}/jobbench_rollout_{rollout_id}.pt"
   )
   ;;
train-smoke)
   # Exactly one GRPO iteration; >= 2 samples per prompt so the group has
   # within-group reward contrast. No checkpoint saving.
   ROLLOUT_ARGS+=(
      --num-rollout 1
      --rollout-batch-size 1
      --n-samples-per-prompt 2
      --global-batch-size 2
      --save-debug-rollout-data "${DUMP_ROOT}/jobbench_rollout_{rollout_id}.pt"
      --save-debug-train-data "${REPO}/results/debug_train/jobbench_train_{rollout_id}_{rank}.pt"
   )
   ;;
full|full-async)
   ROLLOUT_ARGS+=(
      --num-epoch "${NUM_EPOCH}"
      --rollout-batch-size "${ROLLOUT_BS}"
      --n-samples-per-prompt "${N_SAMPLES}"
      --global-batch-size "${GLOBAL_BS}"
      --rollout-shuffle
   )
   SAVE_INTERVAL=${SAVE_INTERVAL:-10}
   CKPT_ARGS+=(
      --save "${SAVE_DIR}/"
      --load "${SAVE_DIR}/"
      --save-interval "${SAVE_INTERVAL}"
   )

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
      # Fully-async rollout: no periodic eval (unsupported there); use the
      # offline scripts/run_jobbench_eval.py on saved checkpoints.
      ROLLOUT_ARGS+=(
         --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
      )
   else
      # Held-out periodic eval on the 65-task main split, the same set and
      # protocol scripts/run_jobbench_eval.py reports. Each eval episode also
      # costs ~9 judge calls, so EVAL_INTERVAL trades wall-clock and judge
      # spend against curve resolution. Eval rollouts skip OPD teacher scoring
      # in the hook.
      EVAL_ARGS+=(
         --eval-interval "${EVAL_INTERVAL}"
         --eval-prompt-data "${EVAL_SET_LABEL}" "${EVAL_DATA}"
         --n-samples-per-eval-prompt "${EVAL_SAMPLES}"
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

OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD:-1}
if [ "${OPTIMIZER_CPU_OFFLOAD}" = "1" ]; then
   OPTIMIZER_ARGS+=(
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
      --use-precision-aware-optimizer
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
   TRAIN_FLAGS=()
else
   TRAIN_FLAGS=(--colocate)
   if [ "${MODE}" != "rollout-debug" ]; then
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
    "TRITON_CACHE_DIR": "${TRITON_CACHE_DIR}",
    "GOOGLE_APPLICATION_CREDENTIALS": "${GOOGLE_APPLICATION_CREDENTIALS:-}",
    "GOOGLE_CLOUD_PROJECT": "${OPD_PROJECT}"
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
   "${OPD_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
