#!/bin/bash
# OfficeQA agent RL + on-policy-distillation (OPD) launcher — single node.
#
# Port of scripts/train_officeqa.sh with slime's OPD path enabled. The teacher
# is NOT a second local server: it is a Vertex AI self-deployed SGLang
# container reached through :rawPredict by
# officeqa_rl.training.vertex_teacher_client, which attaches
# sample.teacher_log_probs during rollout. slime's train side then tilts the
# advantages by the reverse KL:
#
#   advantages[i] -= opd_kl_coef * (student_log_probs[i] - teacher_log_probs[i])
#
# (third_party/slime/slime/backends/megatron_utils/loss.py:620-658, called from
# compute_advantages_and_returns at loss.py:769-775). OPD is orthogonal to the
# advantage estimator — it applies on top of grpo/gspo/cispo/... alike.
#
# Modes (same as train_officeqa.sh):
#   bash scripts/train_officeqa_opd.sh rollout-debug   # 1 rollout, no training (also smoke-tests the teacher)
#   bash scripts/train_officeqa_opd.sh train-smoke     # exactly 1 OPD+GRPO iteration
#   bash scripts/train_officeqa_opd.sh full            # OPD run with periodic eval
#   bash scripts/train_officeqa_opd.sh full-async      # fully-async rollout, disjoint GPUs
#
# All train_officeqa.sh environment knobs still apply (LR, ROLLOUT_BS,
# N_SAMPLES, GLOBAL_BS, NUM_EPOCH, MAX_CTX_LEN, MAX_RESP_LEN, ADV_ESTIMATOR,
# TEMPERATURE, KL_LOSS_COEF, EPS_CLIP, EPS_CLIP_HIGH, ENTROPY_COEF,
# DYNAMIC_FILTER, OVER_SAMPLING_BS, AGENT_STEP_LIMIT, TP_SIZE, CP_SIZE, ...),
# with the same defaults so OPD runs stay comparable to the GRPO baseline.
#
# OPD knobs (defaults in parentheses):
#   OPD_TYPE        (sglang)  slime --opd-type; only "sglang" works here, see below
#   OPD_KL_COEF     (1.0)     slime --opd-kl-coef, the reverse-KL weight
#   OPD_PROJECT     (unset)   teacher endpoint GCP project
#   OPD_LOCATION    (unset)   teacher endpoint LOCATION (a Vertex region such as
#                             us-central1, or "global" — NOT a zone like us-central1-a)
#   OPD_ENDPOINT_ID (unset)   teacher endpoint numeric id
#   OPD_TEACHER_URL (unset)   full URL override; wins over the trio above and lets
#                             you point at a plain local SGLang /generate (no auth
#                             is attempted for non-https URLs)
#   OPD_BOS_TOKEN_ID (unset)  force this token before any teacher instruction and
#                             student context (Gemma uses 2); unset preserves the
#                             historical request exactly
#   OPD_CONCURRENCY (16)      teacher-client semaphore, deliberately far below
#                             --sglang-server-concurrency so Vertex backpressure
#                             cannot starve local policy generation
#   OPD_TIMEOUT_SEC (180)     hard per-request asyncio.wait_for timeout
#   OPD_ATTEMPTS    (5)       retry budget per request (retries the HTTP call,
#                             never the trajectory)
#   OPD_BACKOFF_SEC (1.0)     truncated-exponential backoff base, jittered
#   OPD_PREFLIGHT   (1)       score a 4-token dummy against the teacher before
#                             starting Ray; 0 to skip
#   OPD_TEACHER_FAILURE_LOG   (results/opd_teacher_failures/<EXP_NAME>.jsonl)
#                             a terminal teacher-scoring failure stops the run
#                             (deliberately fatal); its full reason is appended
#                             here as a JSON line first. Empty disables.
#
# Why the officeqa_opd_* settings go through the YAML and the rest through the
# CLI: slime merges --custom-config-path over args in slime_validate_args
# (third_party/slime/slime/utils/arguments.py:1958-1964) — every top-level YAML
# key becomes an attribute on args, and the YAML WINS over the CLI. The
# officeqa_opd_* keys are not argparse flags at all, so the YAML is their only
# route. But that merge runs AFTER the OPD validation block (arguments.py:
# 1728-1758), so use_opd/opd_type must be passed as real CLI flags or a typo
# would silently skip validation. This script therefore writes a derived YAML
# (as train_officeqa.sh already does for officeqa_step_limit) and passes
# --use-opd/--opd-type/--opd-kl-coef on the command line.
#
# Note: OfficeQA does NOT need --rm-url / --custom-rm-path /
# --custom-reward-post-process-path, which upstream slime's OPD example
# requires (third_party/slime/examples/on_policy_distillation/run-qwen3-8B-opd.sh).
# Those wire up slime's own teacher round-trip; the OfficeQA hook scores rewards
# inline and attaches teacher log-probs itself (src/officeqa_rl/training/rollout.py:344-353).
#
# Cost note: --use-opd disables the log-prob reuse fast path
# (third_party/slime/slime/backends/megatron_utils/actor.py:447), so expect an
# extra forward pass per training step relative to the GRPO baseline.

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

# ---------------------------------------------------------------- OPD config
OPD_TYPE_DEF=sglang;     OPD_TYPE=${OPD_TYPE:-${OPD_TYPE_DEF}}
OPD_KL_COEF_DEF=1.0;     OPD_KL_COEF=${OPD_KL_COEF:-${OPD_KL_COEF_DEF}}
OPD_PROJECT=${OPD_PROJECT:-}
OPD_LOCATION=${OPD_LOCATION:-}
OPD_ENDPOINT_ID=${OPD_ENDPOINT_ID:-}
OPD_TEACHER_URL=${OPD_TEACHER_URL:-}
OPD_BOS_TOKEN_ID=${OPD_BOS_TOKEN_ID:-}
case "${OPD_BOS_TOKEN_ID}" in
   "") ;;
   *[!0-9]*) echo "OPD_BOS_TOKEN_ID must be a non-negative integer." >&2; exit 1 ;;
esac
# Both OPD_ENDPOINT_ID and OPD_TEACHER_URL accept a COMMA-SEPARATED list of
# endpoints serving the same model: scoring round-robins across them with
# per-episode affinity (an episode always scores against one endpoint, so its
# prefix cache keeps hitting). OPD_CONCURRENCY applies per endpoint.
# Short teacher mnemonic for the auto experiment name (e.g. q36-35ba3b);
# empty keeps the historical names. The exact endpoint always lives in the
# derived yaml (officeqa_opd_endpoint_id / _teacher_url) and thus in W&B
# config — this tag only makes runs distinguishable at a glance.
OPD_TEACHER_TAG=${OPD_TEACHER_TAG:-}
OPD_CONCURRENCY_DEF=16;  OPD_CONCURRENCY=${OPD_CONCURRENCY:-${OPD_CONCURRENCY_DEF}}
OPD_TIMEOUT_SEC_DEF=180; OPD_TIMEOUT_SEC=${OPD_TIMEOUT_SEC:-${OPD_TIMEOUT_SEC_DEF}}
OPD_ATTEMPTS_DEF=5;      OPD_ATTEMPTS=${OPD_ATTEMPTS:-${OPD_ATTEMPTS_DEF}}
OPD_BACKOFF_SEC_DEF=1.0; OPD_BACKOFF_SEC=${OPD_BACKOFF_SEC:-${OPD_BACKOFF_SEC_DEF}}
OPD_PREFLIGHT=${OPD_PREFLIGHT:-1}
# Teacher scoring mode:
#   per_turn   (default) score each agent turn while the episode runs
#              (officeqa_rl.training.opd_rollout.generate): the teacher's
#              radix cache prefills each context region once per episode and
#              the post-rollout scoring drain disappears.
#   per_sample score each finished sample's full sequence after the episode
#              (officeqa_rl.training.rollout.generate) - the original path,
#              kept as the fallback for A/B comparison.
# OPD_MASK_TO_THINK=1 is a complete think-only launch on its own: it defaults
# OPD_SCORE_MODE=per_turn (hard-required for this arm) and PURE_OPD=1 (every
# distill arm so far runs pure). Explicit values still win — PURE_OPD=0 keeps
# an RL+think-distill mix possible. Added after 2026-08-22, when a
# "-thinkdist"-named run trained as plain pure OPD because the knob was
# dropped from a multi-variable command.
if [ "${OPD_MASK_TO_THINK:-0}" = "1" ]; then
   OPD_SCORE_MODE=${OPD_SCORE_MODE:-per_turn}
   PURE_OPD=${PURE_OPD:-1}
fi

OPD_SCORE_MODE_DEF=per_turn; OPD_SCORE_MODE=${OPD_SCORE_MODE:-${OPD_SCORE_MODE_DEF}}
case "${OPD_SCORE_MODE}" in
per_turn)   OPD_GENERATE_PATH=officeqa_rl.training.opd_rollout.generate ;;
per_sample) OPD_GENERATE_PATH=officeqa_rl.training.rollout.generate ;;
*) echo "OPD_SCORE_MODE=${OPD_SCORE_MODE} must be per_turn or per_sample" >&2; exit 1 ;;
esac
# OPD_ACTION_ONLY=1 = action-only distillation (AOD): each fence-bearing
# turn becomes its own single-turn training sample whose sequence replaces
# the student's <think> block with the canonical empty one; the teacher
# scores that SAME reasoning-free sequence and the loss keeps only the
# fenced-command span. Both policies condition identically, so the student
# is never trained to act against reasoning it just produced. Turns with
# no/malformed fences contribute nothing. Per-turn scoring only.
OPD_ACTION_ONLY_DEF=0; OPD_ACTION_ONLY=${OPD_ACTION_ONLY:-${OPD_ACTION_ONLY_DEF}}
if [ "${OPD_ACTION_ONLY}" = "1" ] && [ "${OPD_SCORE_MODE}" != "per_turn" ]; then
   echo "OPD_ACTION_ONLY=1 requires OPD_SCORE_MODE=per_turn" >&2; exit 1
fi
# OPD_MASK_THINK=1 keeps the reasoning in the scored sequence — the teacher
# annotates the whole turn as usual — and only narrows the loss mask to the
# fenced command. The alternative to OPD_ACTION_ONLY, which instead removes
# the reasoning from what both policies condition on. Per-turn scoring only.
OPD_MASK_THINK_DEF=0; OPD_MASK_THINK=${OPD_MASK_THINK:-${OPD_MASK_THINK_DEF}}
if [ "${OPD_MASK_THINK}" = "1" ] && [ "${OPD_SCORE_MODE}" != "per_turn" ]; then
   echo "OPD_MASK_THINK=1 requires OPD_SCORE_MODE=per_turn" >&2; exit 1
fi
if [ "${OPD_MASK_THINK}" = "1" ] && [ "${OPD_ACTION_ONLY}" = "1" ]; then
   echo "OPD_MASK_THINK=1 and OPD_ACTION_ONLY=1 are alternatives, not a pair:" >&2
   echo "  ACTION_ONLY removes the turn's reasoning from the scored sequence;" >&2
   echo "  MASK_THINK keeps it and only masks its loss." >&2
   exit 1
fi

# OPD_TEACHER_NO_THINK=1 = the on-policy AOD variant, named "-aod2op" so it
# reads as a variant of the aod2 arm rather than a separate method.
#
# Where the three modes put the turn's reasoning:
#
#                        teacher reads it?   trained sequence has it?   loss on
#   (none)                     yes                    yes               whole turn
#   OPD_ACTION_ONLY            no                     no                command
#   OPD_MASK_THINK             yes                    yes               command
#   OPD_TEACHER_NO_THINK       no                     YES               command
#
# It sends the same reasoning-free request ACTION_ONLY does — the current
# turn's <think> block replaced by the canonical empty one — but trains the
# sequence the student actually sampled, reasoning ahead of the action. The
# fenced-command tokens are identical either way, so the teacher's scores
# transfer onto them positionally.
#
# The pairing is mismatched on purpose: the KL compares the student's action
# given its reasoning against the teacher's action given none. ACTION_ONLY
# avoids that by ablating both sides, and pays by training a sequence the
# student never sampled; this keeps the student on-policy and accepts the
# mismatch instead. Which trade wins is the experiment. Per-turn scoring only.
OPD_TEACHER_NO_THINK_DEF=0
OPD_TEACHER_NO_THINK=${OPD_TEACHER_NO_THINK:-${OPD_TEACHER_NO_THINK_DEF}}
if [ "${OPD_TEACHER_NO_THINK}" = "1" ] && [ "${OPD_SCORE_MODE}" != "per_turn" ]; then
   echo "OPD_TEACHER_NO_THINK=1 requires OPD_SCORE_MODE=per_turn" >&2; exit 1
fi
if [ "${OPD_TEACHER_NO_THINK}" = "1" ] && \
   { [ "${OPD_ACTION_ONLY}" = "1" ] || [ "${OPD_MASK_THINK}" = "1" ]; }; then
   echo "OPD_TEACHER_NO_THINK, OPD_ACTION_ONLY and OPD_MASK_THINK are three" >&2
   echo "  alternatives, not a combination — pick one." >&2
   exit 1
fi

# OPD_MASK_TO_THINK=1 distills the THINK region: the teacher scores the full
# real rollout (as pure OPD / mask-think do) and the loss mask narrows to the
# think content through </think> — the mirror image of mask-think, training
# the deliberation and leaving actions untrained. OPD_THINK_SPAN picks what
# carries loss: "all" = that think region; "first"/"last" with
# OPD_THINK_SPAN_TOKENS=t are POSITIONAL over the whole turn response — its
# first/last t tokens, no think parsing and no turn dropped. The template
# opens <think> in the prompt, so "first" lands on the opening deliberation
# (where teacher entropy concentrates) while "last" lands on whatever ends
# the turn, normally the fenced command rather than reasoning.
# Appends "-thinkdist[-first<t>|-last<t>]" to the name.
OPD_MASK_TO_THINK_DEF=0
OPD_MASK_TO_THINK=${OPD_MASK_TO_THINK:-${OPD_MASK_TO_THINK_DEF}}
OPD_THINK_SPAN=${OPD_THINK_SPAN:-all}
OPD_THINK_SPAN_TOKENS=${OPD_THINK_SPAN_TOKENS:-0}
if [ "${OPD_MASK_TO_THINK}" = "1" ] && [ "${OPD_SCORE_MODE}" != "per_turn" ]; then
   echo "OPD_MASK_TO_THINK=1 requires OPD_SCORE_MODE=per_turn" >&2; exit 1
fi
if [ "${OPD_MASK_TO_THINK}" = "1" ] && \
   { [ "${OPD_ACTION_ONLY}" = "1" ] || [ "${OPD_MASK_THINK}" = "1" ] || \
     [ "${OPD_TEACHER_NO_THINK}" = "1" ]; }; then
   echo "OPD_MASK_TO_THINK, OPD_ACTION_ONLY, OPD_MASK_THINK and" >&2
   echo "  OPD_TEACHER_NO_THINK are alternatives, not a combination." >&2
   exit 1
fi
case "${OPD_THINK_SPAN}" in
all|first|last) ;;
*) echo "OPD_THINK_SPAN=${OPD_THINK_SPAN} must be all, first or last" >&2; exit 1 ;;
esac
if [ "${OPD_MASK_TO_THINK}" = "1" ] && [ "${OPD_THINK_SPAN}" != "all" ] && \
   [ "${OPD_THINK_SPAN_TOKENS}" -le 0 ]; then
   echo "OPD_THINK_SPAN=${OPD_THINK_SPAN} needs OPD_THINK_SPAN_TOKENS > 0" >&2
   exit 1
fi

# OPD_TAIL_LOSS=1 widens the trained span of OPD_ACTION_ONLY or
# OPD_TEACHER_NO_THINK from the fenced command to the WHOLE visible region —
# every token after </think>, prose included — and keeps fence-less turns
# instead of dropping them. Offline pass@8 evals of aod2 iter-139 showed the
# command-only mask leaks: visible prose carries no loss, and in failing
# episodes at temp 0.7 the deliberation migrates there (3.4k prose chars/turn
# vs pure OPD's 2.2k, commands identical), erasing the median token advantage
# in the mean (44.0k vs 42.9k despite medians 15.2k vs 19.2k). Tail loss
# trains prose toward the teacher's terse style. Appends "-tail" to the name.
OPD_TAIL_LOSS_DEF=0; OPD_TAIL_LOSS=${OPD_TAIL_LOSS:-${OPD_TAIL_LOSS_DEF}}
# OPD_TAIL_THINK_TOKENS=N (default 0 = off) extends the tail loss under
# OPD_TEACHER_NO_THINK into the last N tokens of the turn's thought (never past
# its start, never an empty thought). The teacher request keeps the student's
# sequence minus the head of the thought: <think> + "[... earlier reasoning
# omitted ...]" + thought ending + </think> + visible tail, so the judge
# deliberates on the truncated thought as its own and scores the tail after
# it; the close marker carries no loss. Cannot combine with OPD_TEACHER_THINK. Built for gemma, whose tail arm trains a bare
# command in 83-97% of turns (no prose between thought and fence); the plan
# for the command sits in the thought's last sentences (last two sentences
# median 35-39 tokens, so N=64 covers them in >90% of thoughts). Appends
# "-tailthink<N>" to the name.
OPD_TAIL_THINK_TOKENS_DEF=0; OPD_TAIL_THINK_TOKENS=${OPD_TAIL_THINK_TOKENS:-${OPD_TAIL_THINK_TOKENS_DEF}}
if [ "${OPD_TAIL_THINK_TOKENS}" != "0" ]; then
   case "${OPD_TAIL_THINK_TOKENS}" in
      ''|*[!0-9]*) echo "OPD_TAIL_THINK_TOKENS must be a non-negative integer (got '${OPD_TAIL_THINK_TOKENS}')" >&2; exit 1;;
   esac
   if [ "${OPD_TAIL_LOSS}" != "1" ] || [ "${OPD_TEACHER_NO_THINK}" != "1" ]; then
      echo "OPD_TAIL_THINK_TOKENS=${OPD_TAIL_THINK_TOKENS} extends the tail loss on the on-policy base:" >&2
      echo "  set OPD_TAIL_LOSS=1 and OPD_TEACHER_NO_THINK=1 alongside it." >&2
      exit 1
   fi
   if [ "${OPD_TEACHER_THINK:-0}" = "1" ]; then
      echo "OPD_TAIL_THINK_TOKENS and OPD_TEACHER_THINK both want the teacher's think block — pick one." >&2
      exit 1
   fi
fi
if [ "${OPD_TAIL_LOSS}" = "1" ] && \
   [ "${OPD_ACTION_ONLY}" != "1" ] && [ "${OPD_TEACHER_NO_THINK}" != "1" ]; then
   echo "OPD_TAIL_LOSS=1 modifies a command-only mode:" >&2
   echo "  set OPD_ACTION_ONLY=1 or OPD_TEACHER_NO_THINK=1 alongside it." >&2
   exit 1
fi

# OPD_PROSE_AS_THINK=1 is the opposite treatment of the visible prose: it is
# reclassified as REASONING and ablated together with the think block, so the
# teacher request (and, under ACTION_ONLY, the trained sample) becomes
# context + <think>\n\n</think>\n\n + command — the teacher conditions on no
# deliberation at all, closing the leak where prose-borne reasoning reached a
# nominally reasoning-blind teacher. Loss stays on the fenced command; turn
# eligibility matches plain AOD. Mutually exclusive with OPD_TAIL_LOSS, which
# trains the prose instead of removing it. Appends "-noprose" to the name.
OPD_PROSE_AS_THINK_DEF=0
OPD_PROSE_AS_THINK=${OPD_PROSE_AS_THINK:-${OPD_PROSE_AS_THINK_DEF}}
if [ "${OPD_PROSE_AS_THINK}" = "1" ] && \
   [ "${OPD_ACTION_ONLY}" != "1" ] && [ "${OPD_TEACHER_NO_THINK}" != "1" ]; then
   echo "OPD_PROSE_AS_THINK=1 modifies a command-only mode:" >&2
   echo "  set OPD_ACTION_ONLY=1 or OPD_TEACHER_NO_THINK=1 alongside it." >&2
   exit 1
fi
if [ "${OPD_PROSE_AS_THINK}" = "1" ] && [ "${OPD_TAIL_LOSS}" = "1" ]; then
   echo "OPD_PROSE_AS_THINK trains nothing new and ablates the prose;" >&2
   echo "  OPD_TAIL_LOSS trains it. Opposite treatments — pick one." >&2
   exit 1
fi

# OPD_TEACHER_THINK=1 lets the judge deliberate: before scoring each turn the
# teacher GENERATES its own think block from the same context (greedy, capped
# at OPD_TEACHER_THINK_MAX_TOKENS), and the scored request becomes
# context + <think>teacher's deliberation</think> + student's visible tail.
# Scores still transfer positionally onto the command tokens of the student's
# REAL rollout, so the student stays fully on-policy — this modifies the
# on-policy base only (requires OPD_TEACHER_NO_THINK=1) and does not compose
# with the tail/prose modifiers. The extra teacher decode is the dominant
# cost; the cap bounds it. Appends "-tthink" ("-aod2op-tthink" overall) to
# the name ("-tt" alone belongs to the train-turns curriculum).
OPD_TEACHER_THINK_DEF=0
OPD_TEACHER_THINK=${OPD_TEACHER_THINK:-${OPD_TEACHER_THINK_DEF}}
OPD_TEACHER_THINK_MAX_TOKENS=${OPD_TEACHER_THINK_MAX_TOKENS:-512}
OPD_TEACHER_THINK_TEMP=${OPD_TEACHER_THINK_TEMP:-0.0}
if [ "${OPD_TEACHER_THINK}" = "1" ] && [ "${OPD_TEACHER_NO_THINK}" != "1" ]; then
   echo "OPD_TEACHER_THINK=1 modifies the on-policy base:" >&2
   echo "  set OPD_TEACHER_NO_THINK=1 alongside it." >&2
   exit 1
fi
if [ "${OPD_TEACHER_THINK}" = "1" ] && \
   { [ "${OPD_TAIL_LOSS}" = "1" ] || [ "${OPD_PROSE_AS_THINK}" = "1" ]; }; then
   echo "OPD_TEACHER_THINK composes only with the plain command-only request;" >&2
   echo "  drop OPD_TAIL_LOSS / OPD_PROSE_AS_THINK." >&2
   exit 1
fi

# COPD low-resource conditioning (contrastive on-policy distillation,
# the single-context variant): prepend a reasoning-effort instruction
# to the TEACHER's scoring context only — the student never sees it — so the
# reverse-KL advantage tilt prefers tokens compatible with efficient,
# low-tool-call behaviour. One teacher pass per score, the existing tilt
# machinery, no loss changes; the constant prefix stays radix-cache friendly.
#   OPD_INSTRUCTION=low    use the built-in low-resource text below
#   OPD_INSTRUCTION="..."  custom single-line text, no double quotes (it is
#                          written into a quoted YAML value); wrapped as a
#                          Qwen system block unless it carries <|im_start|>
#   empty (default)        plain OPD — teacher sees exactly the student context
OPD_INSTRUCTION_LOW="Reasoning Effort: Minimum viable, shortcuts strongly encouraged. Be maximally efficient in this session: keep any thinking brief, run as few commands and tool calls as possible, skip redundant reasoning or verifications, and produce the final answer with the fewest output tokens that still solve the task."
OPD_INSTRUCTION=${OPD_INSTRUCTION:-}
if [ "${OPD_INSTRUCTION}" = "low" ]; then
   OPD_INSTRUCTION="${OPD_INSTRUCTION_LOW}"
fi
case "${OPD_INSTRUCTION}" in *'"'*)
   echo "OPD_INSTRUCTION must not contain double quotes (it is written into a quoted YAML value)." >&2
   exit 1 ;;
esac

# Full COPD (paper §2.2-2.3, paired scoring): ALSO set OPD_INSTRUCTION_HIGH
# and every scoring window is evaluated TWICE (light and heavy contexts,
# concurrent passes) — the client returns the clipped per-token contrast
# clip(l_LT - l_HT, ±OPD_CONTRAST_CLIP) and the loss adds
# opd_kl_coef * contrast directly to the advantages (no student term; the
# GRPO reward advantage stays underneath). This DOUBLES teacher load per
# rollout. "high" selects the built-in text; light may stay empty for a
# neutral-vs-heavy contrast, but the paper pairs light with heavy.
# rollout/opd_reverse_kl logs the mean clipped contrast on these runs.
OPD_INSTRUCTION_HIGH_DEF="Reasoning Effort: Absolute maximum with no shortcuts permitted. You MUST be very thorough in this session: think through every step in detail, run as many commands and tool calls as needed, double-check intermediate results, and only give the final answer after exhaustive verification."
OPD_INSTRUCTION_HIGH=${OPD_INSTRUCTION_HIGH:-}
if [ "${OPD_INSTRUCTION_HIGH}" = "high" ]; then
   OPD_INSTRUCTION_HIGH="${OPD_INSTRUCTION_HIGH_DEF}"
fi
case "${OPD_INSTRUCTION_HIGH}" in *'"'*)
   echo "OPD_INSTRUCTION_HIGH must not contain double quotes (it is written into a quoted YAML value)." >&2
   exit 1 ;;
esac
OPD_CONTRAST_CLIP=${OPD_CONTRAST_CLIP:-10.0}

# PURE_OPD=1 removes the reward-derived advantage in the loss (slime's
# apply_opd_kl_to_advantages zeroes the base advantage before adding the OPD
# term), so the distillation tilt is the entire gradient at ANY N_SAMPLES.
# Rewards are still computed and logged (rollout/reward_mean tracks train
# accuracy); they just stop contributing to the update. Composes with
# OPD_INSTRUCTION / OPD_INSTRUCTION_HIGH (pure COPD = contrast-only gradient).
PURE_OPD_DEF=0; PURE_OPD=${PURE_OPD:-${PURE_OPD_DEF}}
if [ "${PURE_OPD}" = "1" ] && [ "${DYNAMIC_FILTER:-0}" = "1" ]; then
   echo "PURE_OPD=1 with DYNAMIC_FILTER=1 makes no sense: the filter selects prompts by" >&2
   echo "within-group reward variance, but pure OPD takes no gradient from rewards." >&2
   exit 1
fi

# OPD_TRAIN_TURNS=N — horizon curriculum: train on the first N turns of the
# real episode (commands execute, so later turns see genuine observations).
# The cap applies to the agent on TRAINING rollouts only; officeqa_step_limit
# is untouched, so evaluation still runs the full task and every rung's eval
# stays comparable. N=1 trains on the opening move alone.
OPD_TRAIN_TURNS_DEF=0; OPD_TRAIN_TURNS=${OPD_TRAIN_TURNS:-${OPD_TRAIN_TURNS_DEF}}
if [ "${OPD_TRAIN_TURNS}" != "0" ]; then
   case "${OPD_TRAIN_TURNS}" in ''|*[!0-9]*)
      echo "OPD_TRAIN_TURNS=${OPD_TRAIN_TURNS} must be a positive integer" >&2; exit 1 ;;
   esac
   if [ "${PURE_OPD}" != "1" ]; then
      echo "OPD_TRAIN_TURNS requires PURE_OPD=1: a capped episode rarely submits," >&2
      echo "so rewards are ~all zero and the reward-derived advantage has no signal." >&2
      exit 1
   fi
fi

# Only opd_type=sglang reaches the OfficeQA teacher client: the rollout hook
# gates on it explicitly (src/officeqa_rl/training/rollout.py:229), and slime
# rejects --opd-teacher-load in sglang mode while requiring it in megatron mode
# (arguments.py:1730-1758). A megatron teacher would need a second local
# checkpoint and a different launcher, so refuse it here rather than fail deep
# inside placement-group setup.
if [ "${OPD_TYPE}" != "sglang" ]; then
   echo "OPD_TYPE=${OPD_TYPE} is not supported by this script." >&2
   echo "officeqa_rl.training.rollout only attaches teacher log-probs when" >&2
   echo "opd_type == 'sglang' (src/officeqa_rl/training/rollout.py:229)." >&2
   echo "For a Megatron teacher use --opd-teacher-load with a local checkpoint" >&2
   echo "and slime's own OPD example instead." >&2
   exit 1
fi

# Resolve the teacher endpoint. Either a full URL, or the project/location/
# endpoint-id trio that vertex_teacher_client.rawpredict_url() assembles into
#   https://{location-}aiplatform.googleapis.com/v1/projects/{p}/locations/{l}/endpoints/{id}:rawPredict
# Fail here rather than at the first rollout, 20 minutes into a run.
if [ -z "${OPD_TEACHER_URL}" ]; then
   if [ -z "${OPD_PROJECT}" ] || [ -z "${OPD_LOCATION}" ] || [ -z "${OPD_ENDPOINT_ID}" ]; then
      echo "OPD teacher endpoint is not configured. Set either:" >&2
      echo "  OPD_TEACHER_URL=<full url>                  (also accepts a local http SGLang /generate)" >&2
      echo "or all three of:" >&2
      echo "  OPD_PROJECT=<gcp project>" >&2
      echo "  OPD_LOCATION=<vertex region, e.g. us-central1, or global>" >&2
      echo "  OPD_ENDPOINT_ID=<numeric endpoint id>" >&2
      exit 1
   fi
   case "${OPD_LOCATION}" in
   *-*-*)
      # us-central1-a is a zone; Vertex endpoints live in a region/location.
      echo "OPD_LOCATION=${OPD_LOCATION} looks like a zone, not a Vertex location." >&2
      echo "Use a region (us-central1) or 'global'." >&2
      exit 1
      ;;
   esac
fi

# Key hyperparameters (full modes; the smoke modes hard-code theirs). The
# *_DEF values are the baseline; the experiment name only records knobs that
# differ from them.
LR_DEF=1e-6;             LR=${LR:-${LR_DEF}}
ROLLOUT_BS_DEF=8;        ROLLOUT_BS=${ROLLOUT_BS:-${ROLLOUT_BS_DEF}}
N_SAMPLES_DEF=8;         N_SAMPLES=${N_SAMPLES:-${N_SAMPLES_DEF}}
GLOBAL_BS_DEF=$((ROLLOUT_BS * N_SAMPLES)); GLOBAL_BS=${GLOBAL_BS:-${GLOBAL_BS_DEF}}
NUM_EPOCH_DEF=10;        NUM_EPOCH=${NUM_EPOCH:-${NUM_EPOCH_DEF}}
MAX_CTX_LEN_DEF=65536;   MAX_CTX_LEN=${MAX_CTX_LEN:-${MAX_CTX_LEN_DEF}}
MAX_RESP_LEN_DEF=8192;   MAX_RESP_LEN=${MAX_RESP_LEN:-${MAX_RESP_LEN_DEF}}
# Per-episode agent step cap; explicitly empty (AGENT_STEP_LIMIT=) = yaml value.
AGENT_STEP_LIMIT_DEF=50; AGENT_STEP_LIMIT=${AGENT_STEP_LIMIT-${AGENT_STEP_LIMIT_DEF}}

# Loss shape / exploration knobs (all modes).
ADV_ESTIMATOR_DEF=grpo;  ADV_ESTIMATOR=${ADV_ESTIMATOR:-${ADV_ESTIMATOR_DEF}}  # grpo | gspo | cispo | ...
TEMPERATURE_DEF=1.0;     TEMPERATURE=${TEMPERATURE:-${TEMPERATURE_DEF}}
KL_LOSS_COEF_DEF=0.00;   KL_LOSS_COEF=${KL_LOSS_COEF:-${KL_LOSS_COEF_DEF}}     # nonzero enables --use-kl-loss
EPS_CLIP_DEF=0.2;        EPS_CLIP=${EPS_CLIP:-${EPS_CLIP_DEF}}
EPS_CLIP_HIGH=${EPS_CLIP_HIGH:-}           # e.g. 0.28 for DAPO clip-higher; empty = symmetric
ENTROPY_COEF_DEF=0.00;   ENTROPY_COEF=${ENTROPY_COEF:-${ENTROPY_COEF_DEF}}

# DAPO-style dynamic sampling (full mode only; the fully-async rollout worker
# ignores the filter).
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

# Experiment name. The "opd" prefix keeps these runs' checkpoints and W&B
# groups disjoint from the GRPO baseline's, and opdkl is always recorded (it is
# the headline knob of this script) rather than only when it deviates.
if [ -z "${EXP_NAME:-}" ]; then
   EXP_NAME="qwen3.5-4B-opd-${ADV_ESTIMATOR}-${MODE}-opdkl${OPD_KL_COEF}"
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
      _add sm   "${OPD_SCORE_MODE}"   "${OPD_SCORE_MODE_DEF}"
      _add tch  "${OPD_TEACHER_TAG}"
      if [ "${OPD_ACTION_ONLY}" = "1" ]; then
         EXP_NAME+="-aod"
      fi
      if [ "${OPD_MASK_THINK}" = "1" ]; then
         EXP_NAME+="-maskthink"
      fi
      if [ "${OPD_MASK_TO_THINK}" = "1" ]; then
         EXP_NAME+="-thinkdist"
         if [ "${OPD_THINK_SPAN}" != "all" ]; then
            EXP_NAME+="-${OPD_THINK_SPAN}${OPD_THINK_SPAN_TOKENS}"
         fi
      fi
      if [ "${OPD_TEACHER_NO_THINK}" = "1" ]; then
         EXP_NAME+="-aod2op"
      fi
      if [ "${OPD_TAIL_LOSS}" = "1" ]; then
         EXP_NAME+="-tail"
      fi
      if [ "${OPD_TAIL_THINK_TOKENS}" != "0" ]; then
         EXP_NAME+="-tailthink${OPD_TAIL_THINK_TOKENS}"
      fi
      if [ "${OPD_PROSE_AS_THINK}" = "1" ]; then
         EXP_NAME+="-noprose"
      fi
      if [ "${OPD_TEACHER_THINK}" = "1" ]; then
         EXP_NAME+="-tthink"
      fi
      if [ "${OPD_TRAIN_TURNS}" != "0" ]; then
         EXP_NAME+="-tt${OPD_TRAIN_TURNS}"
      fi
      # Not via _add: the instruction is free text, so the name records only
      # that low-resource conditioning was on, keeping checkpoints and W&B
      # groups disjoint from plain-OPD runs.
      if [ -n "${OPD_INSTRUCTION_HIGH}" ]; then
         EXP_NAME+="-copd"
      elif [ -n "${OPD_INSTRUCTION}" ]; then
         EXP_NAME+="-lowres"
      fi
      if [ "${PURE_OPD}" = "1" ]; then
         EXP_NAME+="-pure"
      fi
      ;;
   esac
fi

# A "-thinkdist" name over knobs that do not implement it reproduces the
# 2026-08-22 incident: the run trains as pure OPD while wearing the thinkdist
# name, and nothing downstream notices. Auto-derived names cannot disagree by
# construction; explicit EXP_NAMEs are checked here.
case "-${EXP_NAME}-" in
*-thinkdist-*)
   if [ "${OPD_MASK_TO_THINK:-0}" != "1" ]; then
      echo "EXP_NAME contains -thinkdist but OPD_MASK_TO_THINK=1 is not set;" >&2
      echo "  this would train plain OPD under a thinkdist name. Pass" >&2
      echo "  OPD_MASK_TO_THINK=1 (implies OPD_SCORE_MODE=per_turn, PURE_OPD=1)" >&2
      echo "  or drop -thinkdist from EXP_NAME." >&2
      exit 1
   fi ;;
esac

# The rollout hook reads its settings from the custom config yaml, and the
# officeqa_opd_* keys have no CLI equivalent, so this script ALWAYS writes a
# derived config (train_officeqa.sh only does so for officeqa_step_limit).
# Strip any officeqa_step_limit / officeqa_opd_* already in the base file so
# the values below win, and key the filename on EXP_NAME so two concurrent runs
# never share one derived config.
CUSTOM_CONFIG="${REPO}/configs/training/officeqa_rollout.yaml"
DERIVED_CONFIG="${REPO}/results/officeqa_rollout_opd_${EXP_NAME}.yaml"
if [ ! -f "${CUSTOM_CONFIG}" ]; then
   echo "Base rollout config not found: ${CUSTOM_CONFIG}" >&2
   echo "REPO=${REPO} — is the repo mounted at that path inside the container?" >&2
   exit 1
fi
# A terminal teacher-scoring failure (after all retries) is FATAL: its full
# reason is appended here as a JSON line, then the run stops. Empty disables
# the file.
OPD_TEACHER_FAILURE_LOG=${OPD_TEACHER_FAILURE_LOG-${REPO}/results/opd_teacher_failures/${EXP_NAME}.jsonl}
mkdir -p "${REPO}/results"
{
   grep -v -E '^(officeqa_step_limit|officeqa_opd_[a-z_]+):' "${CUSTOM_CONFIG}"
   if [ -n "${AGENT_STEP_LIMIT}" ]; then
      echo "officeqa_step_limit: ${AGENT_STEP_LIMIT}"
   fi
   echo "# --- OPD teacher (written by scripts/train_officeqa_opd.sh) ---"
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
   if [ -n "${OPD_BOS_TOKEN_ID}" ]; then
      echo "officeqa_opd_bos_token_id: ${OPD_BOS_TOKEN_ID}"
   fi
   if [ "${OPD_ACTION_ONLY}" = "1" ]; then
      echo "officeqa_opd_action_only: true"
   fi
   if [ "${OPD_MASK_THINK}" = "1" ]; then
      echo "officeqa_opd_mask_think: true"
   fi
   if [ "${OPD_MASK_TO_THINK}" = "1" ]; then
      echo "officeqa_opd_mask_to_think: true"
      echo "officeqa_opd_think_span: \"${OPD_THINK_SPAN}\""
      echo "officeqa_opd_think_span_tokens: ${OPD_THINK_SPAN_TOKENS}"
   fi
   if [ "${OPD_TEACHER_NO_THINK}" = "1" ]; then
      echo "officeqa_opd_teacher_no_think: true"
   fi
   if [ "${OPD_TAIL_LOSS}" = "1" ]; then
      echo "officeqa_opd_tail_loss: true"
   fi
   if [ "${OPD_TAIL_THINK_TOKENS}" != "0" ]; then
      echo "officeqa_opd_tail_think_tokens: ${OPD_TAIL_THINK_TOKENS}"
   fi
   if [ "${OPD_PROSE_AS_THINK}" = "1" ]; then
      echo "officeqa_opd_prose_as_think: true"
   fi
   if [ "${OPD_TEACHER_THINK}" = "1" ]; then
      echo "officeqa_opd_teacher_think: true"
      echo "officeqa_opd_teacher_think_max_tokens: ${OPD_TEACHER_THINK_MAX_TOKENS}"
      echo "officeqa_opd_teacher_think_temperature: ${OPD_TEACHER_THINK_TEMP}"
   fi
   if [ "${OPD_TRAIN_TURNS}" != "0" ]; then
      echo "officeqa_opd_train_turns: ${OPD_TRAIN_TURNS}"
   fi
   if [ -n "${OPD_INSTRUCTION}" ]; then
      echo "officeqa_opd_instruction: \"${OPD_INSTRUCTION}\""
   fi
   if [ -n "${OPD_INSTRUCTION_HIGH}" ]; then
      echo "officeqa_opd_instruction_high: \"${OPD_INSTRUCTION_HIGH}\""
      echo "officeqa_opd_contrast: true"
      echo "officeqa_opd_contrast_clip: ${OPD_CONTRAST_CLIP}"
   fi
   if [ "${PURE_OPD}" = "1" ]; then
      echo "officeqa_opd_pure: true"
   fi
   if [ -n "${OPD_TEACHER_FAILURE_LOG}" ]; then
      echo "officeqa_opd_failure_log: \"${OPD_TEACHER_FAILURE_LOG}\""
   fi
} > "${DERIVED_CONFIG}"
CUSTOM_CONFIG="${DERIVED_CONFIG}"

# Preflight: one tiny scoring call before Ray, GPUs, or checkpoint conversion.
# Catches a bad endpoint id, a missing ADC credential, a wrong predict route,
# or an endpoint whose container was not started with return_logprob support —
# all of which otherwise surface as ABORTED samples after the first rollout.
if [ "${OPD_PREFLIGHT}" = "1" ]; then
   PYTHONPATH="${REPO}/src:${SLIME_DIR}" OPD_CONFIG="${CUSTOM_CONFIG}" \
   OPD_MODEL="${MODEL}" python3 - <<'EOF_PREFLIGHT'
import asyncio, os, sys, types, yaml
from officeqa_rl.training.vertex_teacher_client import get_teacher_client

args = types.SimpleNamespace(**(yaml.safe_load(open(os.environ["OPD_CONFIG"])) or {}))
# The rollout workers get hf_checkpoint from slime's CLI args, not the yaml;
# the instruction tokenizer needs it here too, so a bad tokenizer path or
# instruction fails at launch rather than at the first rollout.
args.hf_checkpoint = os.environ["OPD_MODEL"]
client = get_teacher_client(args)
print(f"OPD preflight -> {client.teacher_url}", flush=True)
if client.contrast_enabled:
    print(
        f"COPD contrast ACTIVE: light {client.instruction_prefix_len} + heavy "
        f"{client.heavy_prefix_len} prefix tokens — every score is TWO teacher passes",
        flush=True,
    )
elif client.instruction_prefix_len:
    print(
        f"OPD low-resource instruction ACTIVE: {client.instruction_prefix_len} "
        "prefix tokens prepended to every teacher scoring call",
        flush=True,
    )

async def main():
    try:
        # Any valid ids work; we only check the response shape and auth path.
        # With a round-robin pool every endpoint is probed, so a dead pool
        # member fails here rather than on the episodes it would be assigned.
        for index, url in enumerate(client.endpoint_urls):
            log_probs = await client.score_tokens([1, 2, 3, 4], endpoint_index=index)
            assert len(log_probs) == 3, log_probs
            print(f"OPD preflight OK [{index + 1}/{len(client.endpoint_urls)}]: scored {len(log_probs)} tokens", flush=True)
    finally:
        await client.close()

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

# rerun hygiene (single node)
ray stop --force || true
pkill -9 sglang || true
sleep 2

source "${SLIME_DIR}/scripts/models/qwen3.5-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL}"
)

ROLLOUT_ARGS=(
   --prompt-data "${REPO}/data/slime/officeqa_full_minus_pro_train.jsonl"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --custom-generate-function-path "${OPD_GENERATE_PATH}"
   --custom-config-path "${CUSTOM_CONFIG}"
   --custom-rollout-log-function-path officeqa_rl.training.metrics.log_rollout_extra_metrics
   --custom-eval-rollout-log-function-path officeqa_rl.training.metrics.log_eval_extra_metrics
   --rollout-max-context-len "${MAX_CTX_LEN}"
   --rollout-max-response-len "${MAX_RESP_LEN}"
   --rollout-temperature "${TEMPERATURE}"
)

# OPD flags stay on the CLI: the YAML merge happens after slime's OPD
# validation, so setting these in the yaml would bypass the checks.
# --opd-teacher-load is deliberately absent — slime rejects it for
# opd_type=sglang (arguments.py:1730-1758).
OPD_ARGS=(
   --use-opd
   --opd-type "${OPD_TYPE}"
   --opd-kl-coef "${OPD_KL_COEF}"
)

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
   # One prompt, one sample, rollout only. Teacher scoring still runs (the hook
   # skips it only for eval rollouts), so this doubles as the cheapest
   # end-to-end check that teacher_log_probs come back correctly aligned.
   # Inspect with scripts/inspect_rollout_dump.py.
   ROLLOUT_ARGS+=(
      --num-rollout 1
      --rollout-batch-size 1
      --n-samples-per-prompt 1
      --global-batch-size 1
      --debug-rollout-only
      --save-debug-rollout-data "${DUMP_ROOT}/opd_rollout_{rollout_id}.pt"
   )
   ;;
train-smoke)
   # Exactly one OPD+GRPO iteration; >= 2 samples per prompt so the group has
   # within-group reward contrast. No checkpoint saving.
   ROLLOUT_ARGS+=(
      --num-rollout 1
      --rollout-batch-size 1
      --n-samples-per-prompt 2
      --global-batch-size 2
      --save-debug-rollout-data "${DUMP_ROOT}/opd_rollout_{rollout_id}.pt"
      --save-debug-train-data "${REPO}/results/debug_train/opd_train_{rollout_id}_{rank}.pt"
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
      ROLLOUT_ARGS+=(
         --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
      )
   else
      EVAL_ARGS+=(
         --eval-interval 5
         --eval-prompt-data officeqa_pro "${REPO}/data/slime/officeqa_pro_eval.jsonl"
         --n-samples-per-eval-prompt 1
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
# The reference-policy KL (--use-kl-loss) is independent of the OPD reverse KL:
# OPD tilts advantages toward the teacher, this pulls the policy toward
# --ref-load. Leave KL_LOSS_COEF at 0.00 unless you want both.
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

# GOOGLE_APPLICATION_CREDENTIALS is forwarded when set so the teacher client's
# ADC lookup resolves inside the Ray workers too; with an attached service
# account or `gcloud auth application-default login` state on /root it is
# unset and google.auth.default() finds the credential on its own.
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
