"""Generate one online teacher-SFT rollout batch with no Ray/Megatron stack.

Drives ``officeqa_rl.training.teacher_sft_rollout.generate_rollout`` — the
``--rollout-function-path`` used by ``scripts/train_officeqa_sft_online.sh`` —
against a live teacher endpoint, using a tiny in-memory data source over the
real slime JSONL. The same bridge role ``scripts/smoke_rollout.py`` plays for
the RL hook: verify teacher generation, episode filtering, and per-turn
loss-mask construction end-to-end before committing GPUs to training.

    PYTHONPATH=/mnt/officeqa-rl/src:/mnt/officeqa-rl/third_party/slime \
    python3 scripts/smoke_teacher_sft_rollout.py \
      --teacher-url http://localhost:30000/generate \
      --hf-checkpoint /mnt/Qwen3.5-4B \
      --prompt-data data/slime/officeqa_full_minus_pro_train.jsonl \
      --corpus-dir data/treasury_bulletins_parsed/transformed \
      --workspace-root results/rollout_workspaces \
      --output results/smoke_teacher_sft.json

Prints a per-episode checklist and writes the full sample dump for review.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from slime.utils.types import Sample

from officeqa_rl.training import teacher_sft_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--prompt-data", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1,
                        help="Prompts per rollout call (rollout_batch_size)")
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=1,
                        help="Episodes per prompt (n_samples_per_prompt)")
    parser.add_argument("--teacher-url", required=True,
                        help="Comma-separated SGLang/vLLM URLs (http://... or Vertex :rawPredict)")
    parser.add_argument("--teacher-protocol", choices=["sglang", "vllm"], default="sglang")
    parser.add_argument("--teacher-model", default="teacher")
    parser.add_argument("--filter", choices=["correct", "all"], default="all",
                        help="'all' inspects whatever the teacher produced; 'correct' exercises refill")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="in-flight episodes PER teacher endpoint")
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--loss-mask-type", default="qwen3_5",
                        choices=["qwen3_5", "gemma4", "gemma4_think"],
                        help="slime --loss-mask-type; gemma4* also switches the "
                             "reasoning rendering to gemma's reasoning_content field")
    parser.add_argument("--enable-thinking", choices=["0", "1"], default=None,
                        help="chat_template_kwargs enable_thinking for the TEACHER "
                             "prompt (unset = template default; gemma-4 needs 1 to think)")
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--sandbox-image", default="python:3.11-slim")
    parser.add_argument("--sandbox-runtime", choices=["docker", "local"], default="docker")
    parser.add_argument("--step-limit", type=int, default=50)
    parser.add_argument("--max-context-len", type=int, default=65_536)
    parser.add_argument("--max-response-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--request-timeout-sec", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=Path("results/smoke_teacher_sft.json"))
    return parser.parse_args()


class _ListDataSource:
    """Minimal stand-in for slime's data source: cycles the given rows and
    stamps unique ``index`` per attempt, like ``RolloutDataSourceWithBuffer``."""

    def __init__(self, rows: list[dict], n_samples_per_prompt: int) -> None:
        self._rows = rows
        self._n = n_samples_per_prompt
        self._cursor = 0
        self._index = 0

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        groups = []
        for _ in range(num_samples):
            row = self._rows[self._cursor % len(self._rows)]
            self._cursor += 1
            base = Sample(
                prompt=row["prompt"],
                label=row.get("label"),
                metadata=row.get("metadata") or {},
            )
            group = []
            for _ in range(self._n):
                attempt = copy.deepcopy(base)
                attempt.group_index = self._cursor - 1
                attempt.index = self._index
                self._index += 1
                group.append(attempt)
            groups.append(group)
        return groups


def rollout_args(cli: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        hf_checkpoint=cli.hf_checkpoint,
        loss_mask_type=cli.loss_mask_type,
        rollout_batch_size=cli.rows,
        n_samples_per_prompt=cli.attempts,
        rollout_temperature=cli.temperature,
        rollout_top_p=cli.top_p,
        rollout_max_context_len=cli.max_context_len,
        rollout_max_response_len=cli.max_response_len,
        officeqa_corpus_dir=str(cli.corpus_dir),
        officeqa_workspace_root=str(cli.workspace_root),
        officeqa_sandbox_image=cli.sandbox_image,
        officeqa_sandbox_runtime=cli.sandbox_runtime,
        officeqa_step_limit=cli.step_limit,
        officeqa_sft_teacher_url=cli.teacher_url,
        officeqa_sft_teacher_protocol=cli.teacher_protocol,
        officeqa_sft_teacher_model=cli.teacher_model,
        officeqa_sft_teacher_timeout_sec=cli.request_timeout_sec,
        officeqa_sft_concurrency=cli.concurrency,
        officeqa_sft_filter=cli.filter,
        officeqa_sft_enable_thinking=(
            None if cli.enable_thinking is None else cli.enable_thinking == "1"
        ),
    )


def inspect_episode(samples: list[Sample], tokenizer) -> tuple[dict, bool]:
    first = samples[0]
    metadata = first.metadata or {}
    print(
        f"episode rollout_id={first.rollout_id} uid={metadata.get('uid')} "
        f"exit={metadata.get('exit_status')} steps={metadata.get('steps')} "
        f"reward={first.reward} turns={len(samples)}"
    )
    checks_ok = True
    dumps = []
    for sample in samples:
        response_length = int(sample.response_length or 0)
        checks = {
            "rollout_id_shared": sample.rollout_id == first.rollout_id,
            "loss_mask_matches_response": len(sample.loss_mask or []) == response_length,
            "tokens_cover_response": len(sample.tokens) >= response_length,
            "has_trained_tokens": sum(sample.loss_mask or []) > 0,
            "status_is_enum": isinstance(sample.status, Sample.Status),
        }
        checks_ok = checks_ok and all(checks.values())
        trained = [
            token
            for token, mask in zip(sample.tokens[-response_length:], sample.loss_mask or [])
            if mask
        ]
        print(
            f"  turn {sample.metadata.get('assistant_turn')}: tokens={len(sample.tokens)} "
            f"trained={len(trained)} truncated={sample.metadata.get('truncated')} "
            f"checks={'ok' if all(checks.values()) else 'FAIL'}"
        )
        dumps.append(
            {
                "metadata": sample.metadata,
                "reward": sample.reward,
                "status": str(sample.status),
                "checks": checks,
                "tokens": sample.tokens,
                "loss_mask": sample.loss_mask,
                "trained_text": tokenizer.decode(trained, skip_special_tokens=False),
                "response_text": sample.response,
            }
        )
    return {"rollout_id": first.rollout_id, "turns": dumps}, checks_ok


def main() -> int:
    cli = parse_args()
    with cli.prompt_data.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    rows = rows[cli.row_offset : cli.row_offset + cli.rows] or rows[: cli.rows]

    args = rollout_args(cli)
    data_source = _ListDataSource(rows, cli.attempts)
    output = teacher_sft_rollout.generate_rollout(args, rollout_id=0, data_source=data_source)

    print(f"metrics: {output.metrics}")
    tokenizer = teacher_sft_rollout._get_state(args).tokenizer
    dumps = []
    all_ok = bool(output.samples)
    for episode in output.samples:
        dump, ok = inspect_episode(episode, tokenizer)
        dumps.append(dump)
        all_ok = all_ok and ok

    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(
        json.dumps({"metrics": output.metrics, "episodes": dumps}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {cli.output}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
