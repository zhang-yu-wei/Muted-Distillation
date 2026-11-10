"""Inspect a slime --save-debug-rollout-data dump against the Task 9 contract.

Usage (needs torch + the actor tokenizer, so run inside the slime container):

    PYTHONPATH=/mnt/officeqa-rl/src:/mnt/officeqa-rl/third_party/slime \
    python3 scripts/inspect_rollout_dump.py \
      results/debug_rollout/rollout_0.pt --hf-checkpoint /mnt/Qwen3.5-4B

Prints per-sample invariant checks and (with --show N) the decoded prompt and
trained response of sample N for manual review.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch


def flatten(node: Any) -> list[dict]:
    if isinstance(node, dict):
        return [node]
    if isinstance(node, (list, tuple)):
        return [sample for child in node for sample in flatten(child)]
    raise TypeError(f"unexpected dump node type: {type(node).__name__}")


def status_name(status: Any) -> str:
    """'Sample.Status.TRUNCATED' / 'truncated' / enum -> 'truncated'."""
    return str(status).split(".")[-1].lower()


def truncation_cause(
    status: str,
    metadata: dict,
    total_tokens: int,
    response_length: int,
    *,
    max_ctx_len: int | None,
    max_resp_len: int | None,
) -> str:
    """Classify why a sample is truncated, mirroring the marking logic in
    officeqa_rl.training.rollout.generate:

    - metadata["truncated"] means this sample's own last turn ended with
      finish_reason == "length" (a token limit);
    - otherwise a truncated status was inherited from the episode exiting
      with limits_exceeded (the agent step limit).

    With --max-ctx-len / --max-resp-len, length truncations are further split
    into context-exhausted vs per-turn response cap.
    """
    if status != "truncated":
        return status
    if not metadata.get("truncated"):
        if metadata.get("exit_status") == "limits_exceeded":
            return "step-limit"
        return "truncated-unknown"
    if max_ctx_len and total_tokens >= max_ctx_len:
        return "context-exhausted"
    if max_resp_len and response_length >= max_resp_len:
        return "per-turn-cap"
    return "length"


def print_truncation_report(episodes: dict[str, dict], cause_counts: dict[str, int]) -> None:
    total = sum(cause_counts.values())
    print(f"\n== truncation breakdown ({total} samples) ==")
    for cause, count in sorted(cause_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cause:20s} {count:5d}  ({count / total:.1%})")

    by_exit: dict[str, list[dict]] = {}
    for episode in episodes.values():
        by_exit.setdefault(str(episode["exit_status"]), []).append(episode)
    print(f"== episodes ({len(episodes)}) ==")
    for exit_status, group in sorted(by_exit.items(), key=lambda kv: -len(kv[1])):
        steps = sorted(e["steps"] for e in group if e["steps"] is not None)
        steps_note = (
            f" steps min/med/max={steps[0]}/{steps[len(steps) // 2]}/{steps[-1]}"
            if steps else ""
        )
        n_samples = sum(e["samples"] for e in group)
        length_turns = sum(e["length_truncated_turns"] for e in group)
        print(
            f"  {exit_status:20s} {len(group):4d} episodes -> {n_samples} samples,"
            f" {length_turns} length-truncated turns,{steps_note}"
        )
    print(
        "note: slime's truncated_ratio is sample-weighted; one unfinished episode"
        " contributes every one of its per-turn samples."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("dump", help="rollout_<id>.pt file")
    parser.add_argument("--hf-checkpoint", default=None,
                        help="decode tokens with this tokenizer")
    parser.add_argument("--show", type=int, default=None,
                        help="print the decoded transcript of sample N (0-based)")
    parser.add_argument("--max-ctx-len", type=int, default=None,
                        help="rollout/eval max context len; lets length-truncated "
                             "samples be split into context-exhausted vs per-turn-cap")
    parser.add_argument("--max-resp-len", type=int, default=None,
                        help="per-turn max response len (see --max-ctx-len)")
    args = parser.parse_args()

    data = torch.load(args.dump, weights_only=False)
    samples = flatten(data["samples"])
    print(f"{len(samples)} sample(s) in {args.dump}")

    tokenizer = None
    if args.hf_checkpoint:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

    all_ok = True
    episodes: dict[str, dict] = {}
    cause_counts: dict[str, int] = {}
    for index, sample in enumerate(samples):
        tokens = sample.get("tokens") or []
        loss_mask = sample.get("loss_mask") or []
        log_probs = sample.get("rollout_log_probs") or []
        response_length = int(sample.get("response_length") or 0)
        metadata = sample.get("metadata") or {}
        status = sample.get("status")
        cause = truncation_cause(
            status_name(status), metadata, len(tokens), response_length,
            max_ctx_len=args.max_ctx_len, max_resp_len=args.max_resp_len,
        )
        cause_counts[cause] = cause_counts.get(cause, 0) + 1
        episode = episodes.setdefault(metadata.get("run_id") or f"sample{index}", {
            "exit_status": metadata.get("exit_status"),
            "steps": metadata.get("steps"),
            "samples": 0,
            "length_truncated_turns": 0,
        })
        episode["samples"] += 1
        episode["length_truncated_turns"] += bool(metadata.get("truncated"))

        checks = {
            "loss_mask_matches_response": len(loss_mask) == response_length,
            "log_probs_match_response": len(log_probs) == response_length,
            "tokens_cover_response": len(tokens) >= response_length,
            "reward_set": sample.get("reward") is not None,
            "has_trained_tokens": sum(loss_mask) > 0,
            "same_rollout_id_recorded": sample.get("rollout_id") is not None,
        }
        if tokenizer is not None and response_length:
            prompt_text = tokenizer.decode(tokens[: len(tokens) - response_length])
            label = str(sample.get("label") or "")
            checks["label_not_in_prompt"] = not label or label not in prompt_text

        ok = all(checks.values())
        all_ok = all_ok and ok
        print(
            f"sample {index}: status={status} reward={sample.get('reward')} "
            f"rollout_id={sample.get('rollout_id')} tokens={len(tokens)} "
            f"response={response_length} trained={sum(loss_mask)} "
            f"uid={metadata.get('uid')} submission={metadata.get('submission')!r} "
            f"steps={metadata.get('steps')} "
            f"exit={metadata.get('exit_status')} cause={cause} "
            f"abort={metadata.get('abort_reason')!r}"
        )
        for name, value in checks.items():
            if not value:
                print(f"  [FAIL] {name}")

    print_truncation_report(episodes, cause_counts)

    if args.show is not None:
        if tokenizer is None:
            raise SystemExit("--show requires --hf-checkpoint")
        sample = samples[args.show]
        tokens = sample["tokens"]
        response_length = int(sample.get("response_length") or 0)
        print("\n===== PROMPT (loss-free) =====")
        print(tokenizer.decode(tokens[: len(tokens) - response_length]))
        print("===== TRAINED RESPONSE =====")
        print(tokenizer.decode(tokens[len(tokens) - response_length :]))

    print("all checks passed" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
