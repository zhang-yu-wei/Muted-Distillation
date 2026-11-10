"""Generate one real OfficeQA trajectory through the Task 9 rollout hook.

This is the Task 9 -> Task 10 bridge: it drives
``officeqa_rl.training.rollout.generate`` against a standalone SGLang server
serving the actor checkpoint, with no Ray/Megatron/slime-training stack. Run it
inside an environment that has the slime dependencies (the ``slimerl/slime``
container) with the repository visible, for example:

    PYTHONPATH=/mnt/officeqa-rl/src:/mnt/officeqa-rl/third_party/slime \
    python3 scripts/smoke_rollout.py \
      --router-ip 127.0.0.1 --router-port 30000 \
      --hf-checkpoint /mnt/Qwen3.5-4B \
      --prompt-data data/slime/officeqa_full_minus_pro_train.jsonl \
      --corpus-dir data/treasury_bulletins_parsed/transformed \
      --workspace-root results/rollout_workspaces \
      --output results/smoke_rollout.json

It prints a human-readable trajectory inspection (the Task 10 checklist) and
writes the full sample dump (tokens, loss mask, log probabilities, decoded
segments, metadata) to ``--output`` for manual review.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from slime.utils.types import Sample

from officeqa_rl.training import rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--prompt-data", type=Path, required=True,
                        help="Slime JSONL produced by build_slime_dataset.py")
    parser.add_argument("--uid", default=None,
                        help="Pick the row with this metadata uid (default: --row-index)")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--router-ip", default="127.0.0.1")
    parser.add_argument("--router-port", type=int, default=30000)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--sandbox-image", default="python:3.11-slim")
    parser.add_argument("--sandbox-runtime", choices=["docker", "local"], default="docker")
    parser.add_argument("--step-limit", type=int, default=120)
    parser.add_argument("--command-timeout-sec", type=int, default=30)
    parser.add_argument("--max-observation-chars", type=int, default=12_000)
    parser.add_argument("--max-context-len", type=int, default=32_768)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("results/smoke_rollout.json"))
    return parser.parse_args()


def load_row(prompt_data: Path, uid: str | None, row_index: int) -> tuple[int, dict]:
    with prompt_data.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    if uid is not None:
        matches = [
            (index, row) for index, row in enumerate(rows)
            if (row.get("metadata") or {}).get("uid") == uid
        ]
        if not matches:
            raise SystemExit(f"uid {uid!r} not found in {prompt_data}")
        return matches[0]
    return row_index, rows[row_index]


def rollout_args(cli: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        hf_checkpoint=cli.hf_checkpoint,
        sglang_router_ip=cli.router_ip,
        sglang_router_port=cli.router_port,
        rollout_max_context_len=cli.max_context_len,
        eval_max_context_len=cli.max_context_len,
        officeqa_corpus_dir=str(cli.corpus_dir),
        officeqa_workspace_root=str(cli.workspace_root),
        officeqa_sandbox_image=cli.sandbox_image,
        officeqa_sandbox_runtime=cli.sandbox_runtime,
        officeqa_step_limit=cli.step_limit,
        officeqa_command_timeout_sec=cli.command_timeout_sec,
        officeqa_max_observation_chars=cli.max_observation_chars,
    )


def segment_spans(loss_mask: list[int]) -> list[tuple[int, int, int]]:
    """Split the response region into (mask_value, start, end) runs."""
    spans = []
    start = 0
    for position in range(1, len(loss_mask) + 1):
        if position == len(loss_mask) or loss_mask[position] != loss_mask[start]:
            spans.append((loss_mask[start], start, position))
            start = position
    return spans


def inspect_sample(sample: Sample, tokenizer, label: str) -> dict:
    """Print the Task 10 checklist for one sample and return its JSON dump."""
    response_length = int(sample.response_length or 0)
    prompt_length = len(sample.tokens) - response_length
    prompt_text = tokenizer.decode(sample.tokens[:prompt_length], skip_special_tokens=False)

    checks = {
        "loss_mask_matches_response": len(sample.loss_mask or []) == response_length,
        "log_probs_match_response": len(sample.rollout_log_probs or []) == response_length,
        "tokens_cover_response": len(sample.tokens) >= response_length,
        "reward_set": sample.reward is not None,
        "status_is_enum": isinstance(sample.status, Sample.Status),
        "has_trained_tokens": sum(sample.loss_mask or []) > 0,
        "label_not_in_prompt": label not in prompt_text,
    }

    print(f"  status={sample.status} reward={sample.reward} rollout_id={sample.rollout_id}")
    print(f"  tokens={len(sample.tokens)} prompt={prompt_length} "
          f"response={response_length} trained={sum(sample.loss_mask or [])}")
    for key, value in sample.metadata.items():
        if key in {"uid", "run_id", "submission", "exit_status", "steps",
                   "truncated", "abort_reason"}:
            print(f"  metadata.{key} = {value!r}")
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")

    response_tokens = sample.tokens[prompt_length:]
    segments = []
    for mask_value, start, end in segment_spans(list(sample.loss_mask or [])):
        text = tokenizer.decode(response_tokens[start:end], skip_special_tokens=False)
        segments.append({"loss_mask": mask_value, "tokens": end - start, "text": text})

    return {
        "status": str(sample.status),
        "reward": sample.reward,
        "rollout_id": sample.rollout_id,
        "metadata": sample.metadata,
        "checks": checks,
        "prompt_text": prompt_text,
        "response_text": sample.response,
        "segments": segments,
        "tokens": sample.tokens,
        "loss_mask": sample.loss_mask,
        "rollout_log_probs": sample.rollout_log_probs,
    }


def main() -> int:
    cli = parse_args()
    row_index, row = load_row(cli.prompt_data, cli.uid, cli.row_index)
    metadata = row.get("metadata") or {}
    print(f"row {row_index}: uid={metadata.get('uid')} label={row.get('label')!r}")

    base_sample = Sample(
        index=row_index,
        group_index=0,
        rollout_id=row_index,
        prompt=row["prompt"],
        label=row["label"],
        metadata=metadata,
    )
    sampling_params = {
        "temperature": cli.temperature,
        "top_p": cli.top_p,
        "max_new_tokens": cli.max_new_tokens,
    }

    args = rollout_args(cli)
    result = asyncio.run(rollout.generate(args, base_sample, sampling_params))
    samples = result if isinstance(result, list) else [result]
    tokenizer = rollout._get_tokenizer(args)

    dumps = []
    all_ok = True
    for position, sample in enumerate(samples):
        print(f"sample {position + 1}/{len(samples)}:")
        if sample.status == Sample.Status.ABORTED:
            print(f"  ABORTED: {sample.metadata.get('abort_reason')}")
            dumps.append({"status": str(sample.status), "metadata": sample.metadata})
            all_ok = False
            continue
        dump = inspect_sample(sample, tokenizer, label=str(row["label"]))
        all_ok = all_ok and all(dump["checks"].values())
        dumps.append(dump)

    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(dumps, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {cli.output}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
