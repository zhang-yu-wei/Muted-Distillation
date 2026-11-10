"""Generate fresh on-policy OfficeQA rollouts for the per-turn KL analysis.

The OPD training launchers never saved ``--save-debug-rollout-data`` dumps, so
there are no recorded trajectories for those students. This script regenerates
them online, outside slime: it serves any HF student checkpoint through a
plain SGLang server, runs the standard OfficeQA agent loop against the full
corpus environment (same prompts, sandbox, step limit, and sampling settings
as training), and records every turn's EXACT sampled token ids and per-token
log-probs via :class:`officeqa_rl.training.slime_sglang_backend.SlimeSGLangBackend`
— the same token-exact backend the training rollout hook uses.

Output is a ``torch.save({"samples": [...]})`` file in the same layout as a
slime debug-rollout dump, readable by ``scripts/analyze_opd_turn_kl.py``
unchanged: one sample per agent turn (tokens = prompt_ids + output_ids,
loss_mask all ones over the response, ``rollout_log_probs`` from sampling),
with all of an episode's turns sharing one ``metadata.run_id`` in turn order,
which is exactly how the analysis reconstructs episodes.

Usage (inside the officeqa-train container on a GPU node):

    # 1. serve the student (any HF checkpoint) — e.g. in a tmux window
    python3 -m sglang.launch_server \\
      --model-path /mnt/disks/local-ssd/Qwen3.5-4B \\
      --host 127.0.0.1 --port 30000 --dp-size 8 --mem-fraction-static 0.8

    # 2. generate (workspace/corpus paths must be host-visible under DooD)
    cd /mnt/disks/local-ssd/officeqa-rl
    PYTHONPATH=src python3 scripts/generate_online_rollouts.py \\
      --sglang-url http://127.0.0.1:30000 \\
      --hf-checkpoint /mnt/disks/local-ssd/Qwen3.5-4B \\
      --prompt-data data/slime/officeqa_full_minus_pro_train.jsonl \\
      --num-tasks 32 --samples-per-task 4 \\
      --out results/online_rollouts/qwen35_base.pt

    # 3. analyze (see analyze_opd_turn_kl.py header)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from pathlib import Path

from officeqa_rl.agent import OfficeQAAgent
from officeqa_rl.dataset import task_from_slime_fields
from officeqa_rl.full_corpus_env import FullCorpusEnvironment
from officeqa_rl.scoring import DEFAULT_TOLERANCE, score_answer
from officeqa_rl.training.slime_sglang_backend import SlimeSGLangBackend
from officeqa_rl.types import AgentConfig, EnvironmentConfig

logger = logging.getLogger("generate_online_rollouts")

# Agent exit statuses that still yield analyzable turns, mirroring
# officeqa_rl.training.rollout._STATUS_BY_EXIT ("error: ..." statuses mean the
# episode died for infrastructure reasons and is skipped).
_STATUS_BY_EXIT = {"submitted": "completed", "limits_exceeded": "truncated"}


async def run_episode(
    row: dict,
    sample_index: int,
    args: argparse.Namespace,
    tokenizer,
    environment_config: EnvironmentConfig,
) -> list[dict]:
    task = task_from_slime_fields(row["prompt"], row.get("label"), row.get("metadata"))
    if task.answer is None:
        logger.warning("task %s has no label; skipped", task.uid)
        return []
    metadata = row.get("metadata") or {}
    tolerance = float(metadata.get("officeqa_reward_tolerance", DEFAULT_TOLERANCE))
    run_id = f"uid{task.uid}-s{sample_index}-{uuid.uuid4().hex[:8]}"

    backend = SlimeSGLangBackend(
        base_url=args.sglang_url,
        tokenizer=tokenizer,
        max_context_len=args.max_context_len,
    )
    agent = OfficeQAAgent(
        backend,
        AgentConfig(step_limit=args.step_limit),
        sampling_params={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_response_len,
        },
    )
    try:
        async with FullCorpusEnvironment(environment_config, task, run_id=run_id) as env:
            result = await agent.run(env)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        logger.warning("episode %s failed: %s: %s", run_id, type(error).__name__, error)
        return []

    status = _STATUS_BY_EXIT.get(result.status)
    if status is None:
        logger.warning("episode %s ended with %s; skipped", run_id, result.status)
        return []
    reward = (
        score_answer(
            ground_truth=task.answer,
            predicted=result.submission,
            tolerance=tolerance,
        )
        if result.status == "submitted"
        else 0.0
    )

    turns = [st.turn for st in backend.session_turns if st.turn.output_ids]
    samples = []
    for turn_number, turn in enumerate(turns, start=1):
        output_ids = list(turn.output_ids)
        samples.append(
            {
                "tokens": list(turn.prompt_ids) + output_ids,
                "response_length": len(output_ids),
                "loss_mask": [1] * len(output_ids),
                "rollout_log_probs": list(turn.output_log_probs or []),
                "reward": reward,
                "status": status,
                "metadata": {
                    "run_id": run_id,
                    "uid": task.uid,
                    "sample_index": sample_index,
                    "exit_status": result.status,
                    "steps": result.steps,
                    "submission": result.submission,
                    "finish_reason": turn.finish_reason,
                    "turn": turn_number,
                    "n_turns": len(turns),
                },
            }
        )
    logger.info(
        "episode %s: %s reward=%.2f steps=%d turns=%d",
        run_id, result.status, reward, result.steps, len(samples),
    )
    return samples


async def main_async(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoTokenizer

    rows = []
    with open(args.prompt_data, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if args.num_tasks:
        rows = rows[: args.num_tasks]
    logger.info(
        "%d tasks x %d samples = %d episodes",
        len(rows), args.samples_per_task, len(rows) * args.samples_per_task,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    repo = Path(__file__).resolve().parent.parent
    environment_config = EnvironmentConfig(
        corpus_dir=Path(args.corpus_dir or repo / "data/treasury_bulletins_parsed/transformed"),
        pdf_dir=None,
        workspace_root=Path(args.workspace_root or repo / "results/rollout_workspaces"),
        sandbox_image=args.sandbox_image,
        command_timeout_sec=args.command_timeout_sec,
        max_observation_chars=args.max_observation_chars,
        sandbox_runtime="docker",
        max_concurrent_sandbox_boots=args.max_concurrent_sandbox_boots,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(args.concurrency)
    collected: list[dict] = []
    finished = 0
    total = len(rows) * args.samples_per_task
    flush_lock = asyncio.Lock()

    async def one(row: dict, sample_index: int) -> None:
        nonlocal finished
        async with gate:
            samples = await run_episode(
                row, sample_index, args, tokenizer, environment_config
            )
        async with flush_lock:
            collected.extend(samples)
            finished += 1
            # Periodic flush so a crash late in a long run loses little.
            if finished % args.flush_every == 0 or finished == total:
                torch.save({"samples": list(collected)}, out_path)
                logger.info(
                    "progress %d/%d episodes, %d turn-samples -> %s",
                    finished, total, len(collected), out_path,
                )

    await asyncio.gather(
        *(one(row, k) for row in rows for k in range(args.samples_per_task))
    )
    torch.save({"samples": collected}, out_path)
    episodes = {s["metadata"]["run_id"] for s in collected}
    rewards = [s["reward"] for s in collected]
    print(
        f"wrote {len(collected)} turn-samples from {len(episodes)} episodes "
        f"to {out_path}"
        + (f" (mean reward {sum(rewards) / len(rewards):.3f})" if rewards else "")
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--sglang-url", required=True,
                        help="base URL of the SGLang server hosting the student")
    parser.add_argument("--hf-checkpoint", required=True,
                        help="tokenizer path (must match the served model)")
    parser.add_argument("--prompt-data", required=True,
                        help="slime prompt jsonl (prompt/label/metadata rows)")
    parser.add_argument("--out", required=True, help="output .pt dump path")
    parser.add_argument("--num-tasks", type=int, default=0,
                        help="use only the first N tasks (0 = all)")
    parser.add_argument("--samples-per-task", type=int, default=1,
                        help="episodes per task (temperature sampling fan-out)")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="concurrent episodes")
    parser.add_argument("--flush-every", type=int, default=8,
                        help="save partial output every N finished episodes")
    # Sampling/agent settings; defaults mirror the training launchers.
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-context-len", type=int, default=65536)
    parser.add_argument("--max-response-len", type=int, default=8192)
    parser.add_argument("--step-limit", type=int, default=50)
    # Environment settings; defaults mirror configs/training/officeqa_rollout.yaml
    # with repo-relative paths (host-visible under DooD on COS nodes).
    parser.add_argument("--corpus-dir", default=None)
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--sandbox-image", default="python:3.11-slim")
    parser.add_argument("--command-timeout-sec", type=int, default=30)
    parser.add_argument("--max-observation-chars", type=int, default=12_000)
    parser.add_argument("--max-concurrent-sandbox-boots", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
