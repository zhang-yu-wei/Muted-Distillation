# SpreadsheetBench 2 Evaluation — Reproduction Guide

Step-by-step setup and launch instructions for running the SpreadsheetBench 2
evaluation ([site](https://spreadsheetbench.github.io/),
[paper](https://arxiv.org/abs/2606.29955)) with this repo's agent harness on a
fresh machine. The harness reuses the OfficeQA agent loop (one fenced bash
command per turn, Docker sandbox, full trace saving, resume, pass@k) and
grades with the benchmark's official pipeline: LibreOffice recalculation of
cached formula values, then cell-level comparison against the golden workbook
using the unmodified official `evaluation.py`.

## 1. Prerequisites

- Linux host with **Docker** (the current user must be able to run `docker
  run`; the sandbox and the grading recalculation both use containers).
- **uv** and **git**.
- For the `vllm` backend: an NVIDIA GPU (a 4B model fits on one 80 GB card at
  `VLLM_GPU_MEMORY_UTILIZATION=0.5`) plus system FFmpeg
  (`sudo apt-get install -y ffmpeg`, needed by torchcodec).
- For the `gemini` / `kimi` / `openai-server` backends: no GPU, only API
  access or a served endpoint.

```bash
git clone <this repo> officeqa-rl && cd officeqa-rl
uv sync --all-extras     # --all-extras pulls the vLLM inference extra;
                         # plain `uv sync` is enough for API-only backends
```

## 2. Download the benchmark data (~300 MB)

The dataset is on HuggingFace (MIT license) and is gitignored, so every
machine downloads it once:

```bash
.venv/bin/python - <<'EOF'
import zipfile
from huggingface_hub import snapshot_download
root = snapshot_download(repo_id="KAKA22/SpreadsheetBench-v2", repo_type="dataset",
                         local_dir="data/spreadsheetbench_v2")
zipfile.ZipFile("data/spreadsheetbench_v2/spreadsheetbench-v2.zip").extractall(
    "data/spreadsheetbench_v2")
print("extracted to data/spreadsheetbench_v2/spreadsheetbench-v2")
EOF
```

Expected layout (321 tasks): `data/spreadsheetbench_v2/spreadsheetbench-v2/`
containing `Debugging/` (100), `Financial_Model/` (100), `Template/` (97),
`Visualization/` (24), each with a `dataset.json` and a `spreadsheet/` tree
holding input and golden `.xlsx` files. Quick validation:

```bash
.venv/bin/python - <<'EOF'
import json, pathlib
root = pathlib.Path("data/spreadsheetbench_v2/spreadsheetbench-v2")
for cat in ("Debugging", "Financial_Model", "Template", "Visualization"):
    rows = json.loads((root / cat / "dataset.json").read_text())
    missing = [r["id"] for r in rows
               if not (root / cat / r["spreadsheet_path"]).is_file()
               or not (root / cat / r["golden_response_path"]).is_file()]
    print(f"{cat}: {len(rows)} tasks, missing files: {len(missing)}")
EOF
```

## 3. Official evaluation code (vendored, nothing to do)

Grading imports the benchmark's comparison functions directly from the
official repo, which is vendored as plain files at
`third_party/SpreadsheetBench-2/` (from
<https://github.com/RUCKBReasoning/SpreadsheetBench-2>, kept unmodified), so
a clone of this repo already has it. Only if that folder is missing:

```bash
git clone --depth 1 https://github.com/RUCKBReasoning/SpreadsheetBench-2 \
    third_party/SpreadsheetBench-2
```

## 4. Build the sandbox Docker image

Mirrors the official benchmark image: python 3.11 with
openpyxl/pandas/numpy/xlsxwriter/matplotlib, LibreOffice (agents use it to
recalculate and self-verify; grading uses it to refresh cached values), and
the official `view_xlsx` inspection tool. The build context is committed
under `docker/spreadsheetbench/`:

```bash
docker build -t spreadsheetbench-v2-sandbox docker/spreadsheetbench
```

The launcher auto-builds this image when it is missing, so this step is
optional but makes the first run predictable. The build takes a few minutes
(LibreOffice apt install).

## 5. Smoke test: one task

Always run a single task first on a new machine. This exercises the full
path: sandbox boot, agent loop, output workbook collection, LibreOffice
recalculation, official comparison, trace writing.

```bash
# Local checkpoint through vLLM (any HF hub id or local HF directory):
SSB_UID=Template__01_01 VLLM_MODEL=/path/to/model bash scripts/eval_spreadsheetbench.sh

# Or, without a GPU, through the Gemini API:
SSB_UID=Template__01_01 SSB_BACKEND=gemini SSB_MODEL=gemini-3.5-flash \
  GEMINI_API_KEYS_FILE=~/gemini_keys.txt bash scripts/eval_spreadsheetbench.sh
```

Success looks like a per-task JSON under the results dir whose `metrics`
block has a nonzero `regression_accuracy` (regression cells ≈ preserved
input) and an `outputs/<uid>_output.xlsx` workbook next to it. A `status` of
`submitted` with `score` 0.0 is a normal model result, not a harness failure
— strict accuracy is hard (best frontier model: ~35%).

## 6. Full evaluation

```bash
# Open-weights checkpoint through vLLM (results in results/ssb2-vllm-<tag>/):
VLLM_MODEL=/path/to/ckpt bash scripts/eval_spreadsheetbench.sh

# Gemini anchor:
SSB_BACKEND=gemini SSB_MODEL=gemini-3.5-flash GEMINI_API_KEYS_FILE=... \
  bash scripts/eval_spreadsheetbench.sh

# Model served elsewhere (vllm serve / SGLang, OpenAI-compatible):
SSB_BACKEND=openai-server SSB_MODEL=<served id> \
  SSB_OPENAI_BASE_URL=http://<host>:8000/v1 bash scripts/eval_spreadsheetbench.sh

# pass@k (8 attempts per task, like the OfficeQA pro evals):
SSB_SAMPLES_PER_TASK=8 SSB_TEMPERATURE=1.0 VLLM_MODEL=/path/to/ckpt \
  bash scripts/eval_spreadsheetbench.sh
```

Runs are resumable: attempts that already have a non-empty result JSON are
skipped, so rerunning the same command continues an interrupted eval.
Crashed attempts are retried (`SSB_FAILURE_RETRIES`, default 2).

RL checkpoints must first be converted to HF format with
`third_party/slime/tools/convert_torch_dist_to_hf.py` (see the main
`README.md`, "Open-weights models through vLLM").

### Knobs (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `SSB_BACKEND` | `vllm` | `vllm`, `gemini`, `kimi`, `openai-server` |
| `SSB_MODEL` / `VLLM_MODEL` | `/mnt/Qwen3.5-4B` | model id/path (`SSB_MODEL` wins) |
| `SSB_CATEGORIES` | `Debugging Financial_Model Template` | space-separated category list |
| `SSB_UID` | unset | run one task, e.g. `Template__01_01` |
| `SSB_SAMPLES_PER_TASK` | 8 local vLLM; 1 API/single-UID | attempts per task for pass@k (needs `SSB_TEMPERATURE>0`) |
| `SSB_TEMPERATURE` | 1.0 | sampling temperature; matches training rollouts |
| `SSB_CONCURRENCY` | 48 local vLLM; 4 API | concurrent tasks (each boots one sandbox container) |
| `SSB_STEP_LIMIT` | 50 | agent turns per attempt (official cap is also 50) |
| `SSB_COMMAND_TIMEOUT_SEC` | 120 | per-command sandbox timeout (workbook edits are slow) |
| `SSB_MAX_TOKENS` | 8192 | generation cap per turn |
| `SSB_SANDBOX_IMAGE` | `spreadsheetbench-v2-sandbox` | sandbox + recalculation image |
| `SSB_OUTPUT_DIR` | `results/ssb2-<backend>-<model tag>` | results directory |
| `SSB_DATA_ROOT` | `data/spreadsheetbench_v2/spreadsheetbench-v2` | dataset root |
| `VLLM_DATA_PARALLEL_SIZE`, `VLLM_TENSOR_PARALLEL_SIZE`, `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_MAX_NUM_SEQS`, `VLLM_MAX_MODEL_LEN` | 8 / 1 / 0.85 / 16 / unset | local-vLLM backend knobs |

The local defaults target 4B-scale models that fit one replica per GPU. For a
larger model, increase TP (or PP), reduce DP so `DP × TP × PP` matches the GPU
count, and tune concurrency and `VLLM_MAX_NUM_SEQS` for KV-cache headroom.

Gemini/Kimi credentials follow the OfficeQA conventions:
`GEMINI_API_KEYS_FILE` (one key per line) or `GEMINI_API_KEYS`, and
`MOONSHOT_API_KEY` for Kimi.

## 7. Reading the results

`results/ssb2-<backend>-<tag>/` contains:

- `<uid>.json` (or `<uid>.s<i>.json` for pass@k) — one per attempt, same
  shape as OfficeQA results: full `messages` trace, per-command
  `metadata.environment.command_history`, `status`, `steps`, plus:
  - `score` — the official strict accuracy (1.0 only if **every** graded cell
    matches: all modification cells right and ≥99.8% of regression cells
    preserved). pass@k therefore reads "task fully correct".
  - `metrics.modification_accuracy` — fraction of the cells the task is
    supposed to change that are correct. This is the dense signal; expect it
    to move long before strict accuracy does for small models.
  - `metrics.regression_accuracy` — fraction of untouched-region cells
    preserved (low values mean the agent clobbered existing content).
  - `metrics.cell_errors` — up to 50 messages naming the exact mismatched
    cells (`answer=` golden vs `output=` produced), for failure analysis.
- `outputs/<uid>_output.xlsx` — the graded workbook of every attempt
  (already recalculated; open it to inspect what the agent actually built).
- `summary.json` — aggregate strict accuracy and modification accuracy,
  per-category breakdown, status counts, `pass_at_k` when sampling.
- `workspaces/` — the per-attempt sandbox workspaces; safe to delete after a
  run.

Reference points on strict accuracy (official leaderboard, 321 tasks): best
frontier model ≈ 34.9% overall, debugging ≈ 12%.

## 8. Notes and caveats

- **Visualization category** (24 tasks) is excluded by default: the official
  grading renders charts through the Excel/WPS COM interface (Windows-only)
  and judges them with a GLM-4.6V checklist. Adding `Visualization` to
  `SSB_CATEGORIES` runs the agent and saves the workbooks but scores them 0
  locally; grade them separately with
  `third_party/SpreadsheetBench-2/evaluation/run_visual_vlm_checklist_eval.py`
  on a Windows box.
- **The LibreOffice recalculation step is not optional.** Workbooks written
  by openpyxl carry formulas without cached values; skipping recalculation
  (`--skip-recalc`) makes every formula cell read as empty and scores near 0.
  A verified-good workbook scored 0.03 before recalculation and 1.00 after.
- **Context overflow with verbose models.** The eval path has no per-turn
  context budgeting (same as the OfficeQA evals). Long-winded models — the
  base Qwen3.5-4B in particular — can generate ~5k+ tokens per turn here
  (thinking plus full openpyxl scripts), overflowing a 262k context before
  the 50-step limit; the attempt then fails with a "maximum context length"
  error (recorded in `summary.json` under `failure_counts.context_length`,
  and not retried at temperature 0, where the overflow is deterministic).
  For such models bound the budget, e.g. `SSB_MAX_TOKENS=4096
  SSB_STEP_LIMIT=30` (worst case ≈ 210k tokens). RL/SFT checkpoints and API
  models run far terser turns and are fine at the defaults.
- The sandbox runs with `--network none` and a read-only root filesystem
  (writable `/workspace` and `/tmp`), matching the official no-network
  environment. Golden workbooks are never mounted into the sandbox.
- The agent submits with a bare `submit` (no answer string); the environment
  refuses it until `/workspace/<id>_output.xlsx` exists. Attempts that hit
  the step limit are still graded on whatever output file they saved.
- Disk: each attempt keeps its workspace (input copy) and graded output
  workbook; a full 297-task run is a few hundred MB. GPU eval throughput is
  bound by episode length (up to 50 turns × 8k tokens), not grading;
  LibreOffice recalculation adds ~5–15 s per attempt.
