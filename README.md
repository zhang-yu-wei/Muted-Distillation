# Muted Distillation

On-policy distillation for tool-using agents, where the teacher scores the
student's actions without seeing the student's reasoning.

The goal is to transfer a teacher's capabilities to a small agent while
letting the student develop its own, more economical reasoning. In standard
on-policy distillation the teacher scores every token the student samples,
reasoning included, so the student is pulled towards the teacher's long
deliberation. Muted distillation removes the student's reasoning from what the
teacher sees and trains only the visible part of each turn.

The code is built on [slime](https://github.com/THUDM/slime) and covers three
benchmarks: OfficeQA (question answering over a document corpus),
SpreadsheetBench and JobBench. In each, the model works in turns of reasoning
plus one shell command inside a sandbox and is scored on its final submission.

## Method

For every agent turn, the student's exact sampled tokens are sent to a served
teacher, which returns its log-probability of each token without generating
anything. The per-token reverse KL between student and teacher is then
subtracted from the advantage.

What the teacher sees, and which tokens carry the loss, is configurable:

| Setting | Teacher sees | Loss on |
|---|---|---|
| Standard distillation | The full turn | Every token |
| **Muted distillation** (`OPD_TEACHER_NO_THINK=1 OPD_TAIL_LOSS=1`) | The turn with the reasoning replaced by an empty think block | The visible part of the turn; the student stays on-policy |
| Action-only (`OPD_ACTION_ONLY=1`) | Reasoning removed | The command, on a reasoning-free sequence for student and teacher alike |
| Masked-think (`OPD_MASK_THINK=1`) | The full turn | The command only |

`PURE_OPD=1` drops the task-reward advantage so that distillation is the whole
gradient. GRPO, Dr. GRPO and teacher-trace SFT (offline and online) are
included as baselines.

## Where to look

| File | What it does |
|---|---|
| `src/officeqa_rl/training/opd_rollout.py` | Rollout hook. `generate` runs one episode and scores each turn with the teacher while the episode is still running; `_action_only_plan` decides what the teacher sees. |
| `src/officeqa_rl/training/vertex_teacher_client.py` | Async scoring client for SGLang and vLLM teachers, with retries and per-endpoint concurrency. |
| `third_party/slime/slime/backends/megatron_utils/loss.py` | `apply_opd_kl_to_advantages` turns teacher log-probs into the advantage term. |
| `src/officeqa_rl/training/slime_sglang_backend.py` | Exact-token generation: sampled token ids and log-probs are the training source of truth. |
| `src/officeqa_rl/agent.py`, `full_corpus_env.py` | Agent loop and Docker sandbox, shared by training and evaluation. |
| `src/officeqa_rl/training/teacher_sft_rollout.py` | Online SFT, with a frozen teacher running episodes inside the training loop. |

## Repository layout

```text
src/officeqa_rl/     agent, environments, scoring, inference backends, training hooks
scripts/             evaluation runners, training launchers, dataset builders, inspection tools
configs/training/    rollout settings per benchmark
docker/              sandbox and training images
tests/               unit and contract tests
bugs/                write-ups of framework bugs found along the way
third_party/         vendored slime, mini-swe-agent, SpreadsheetBench-2
```

## Usage

```bash
uv sync --extra dev
uv run pytest
```

Training runs inside the slime container (`docker/`) with the host Docker
socket mounted:

```bash
# GRPO baseline
bash scripts/train_officeqa.sh full

# Muted distillation
OPD_TEACHER_URL=http://host:30000/generate \
PURE_OPD=1 OPD_TEACHER_NO_THINK=1 OPD_TAIL_LOSS=1 \
  bash scripts/train_officeqa_opd.sh full
```

Evaluation uses the same agent and sandbox as training:

```bash
VLLM_MODEL=Qwen/Qwen3.5-4B bash scripts/eval_vllm_full.sh
```

`train_ssb1*.sh` and `train_jobbench*.sh` are the equivalent launchers for the
other two benchmarks. Every launcher documents its environment variables in
its header. Benchmark data is not included; the `scripts/build_*_dataset.py`
scripts convert the source datasets into the format the hooks read.

## Changes to slime

slime is vendored as plain files, with the upstream commit recorded in
`third_party/slime/UPSTREAM.md`. The changes on top of it:

- distillation options in the loss: pure mode, reverse-KL clipping, a
  no-action penalty and a contrastive variant;
- a Dr. GRPO advantage estimator;
- log-probs computed from bf16 logits chunk by chunk, to fit long contexts;
- a fix for variable fan-out rollouts (see `bugs/`);
- Gemma-4 model support.

## Third-party code

`third_party/slime` (THUDM/slime, modified as above),
`third_party/mini-swe-agent` (unmodified) and
`third_party/SpreadsheetBench-2` keep their own licenses.
