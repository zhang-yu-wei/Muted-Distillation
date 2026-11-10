"""Terminal viewer for slime rollout dumps (.pt): read what the model did.

Groups per-turn training samples back into episodes. Default output is a
one-line-per-episode summary (reward, exit status, steps, submission vs
label); `-e N` prints episode N's fullest conversation turn by turn.

Run inside the training container (needs torch + the actor tokenizer):

    # episode summary table
    sudo docker exec -w /mnt/officeqa-rl officeqa-train \
      python3 scripts/view_rollout.py results/debug_rollout/<EXP>/rollout_5.pt

    # full transcript of episode 3 (pipe to less -R to page with colors)
    ... python3 scripts/view_rollout.py <dump> -e 3 | less -R

Dumps come from DUMP_ROLLOUTS=1 (train: rollout_<id>.pt, eval:
rollout_eval_<id>.pt).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

TURN_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>|$)", re.DOTALL)


def flatten(node):
    if isinstance(node, dict):
        return [node]
    return [s for child in node for s in flatten(child)]


def group_episodes(samples):
    episodes: dict[str, list[dict]] = {}
    for sample in samples:
        metadata = sample.get("metadata") or {}
        episodes.setdefault(metadata.get("run_id") or f"sample-{id(sample)}", []).append(sample)
    return list(episodes.values())


def episode_summary(eps_samples):
    fullest = max(eps_samples, key=lambda s: len(s.get("tokens") or []))
    metadata = fullest.get("metadata") or {}
    return {
        "fullest": fullest,
        "uid": metadata.get("uid"),
        "reward": max(float(s.get("reward") or 0.0) for s in eps_samples),
        "exit": metadata.get("exit_status"),
        "steps": metadata.get("steps"),
        "submission": metadata.get("submission"),
        "label": fullest.get("label"),
        "n_samples": len(eps_samples),
        "tokens": len(fullest.get("tokens") or []),
    }


class Palette:
    def __init__(self, enabled: bool):
        pick = (lambda code: f"\033[{code}m") if enabled else (lambda code: "")
        self.reset, self.bold, self.dim = pick("0"), pick("1"), pick("2")
        self.green, self.red, self.cyan, self.yellow = (
            pick("32"), pick("31"), pick("36"), pick("33"),
        )

    def role(self, name: str) -> str:
        return {"assistant": self.green, "user": self.cyan, "system": self.yellow}.get(
            name, ""
        )


def print_table(episodes, palette):
    print(f"{'#':>3} {'uid':10} {'reward':>7} {'exit':16} {'steps':>5} "
          f"{'toks':>6} {'submission -> label'}")
    for index, eps_samples in enumerate(episodes):
        s = episode_summary(eps_samples)
        color = palette.green if s["reward"] > 0 else palette.red
        sub = str(s["submission"])[:40]
        label = str(s["label"])[:30]
        print(f"{index:>3} {str(s['uid']):10} {color}{s['reward']:>7g}{palette.reset} "
              f"{str(s['exit']):16} {str(s['steps']):>5} {s['tokens']:>6} "
              f"{sub!r} -> {label!r}")


def print_transcript(eps_samples, tokenizer, palette, max_turn_chars, only_role=None):
    s = episode_summary(eps_samples)
    color = palette.green if s["reward"] > 0 else palette.red
    print(f"{palette.bold}uid={s['uid']}  reward={color}{s['reward']:g}{palette.reset}"
          f"{palette.bold}  exit={s['exit']}  steps={s['steps']}  "
          f"tokens={s['tokens']}{palette.reset}")
    print(f"submitted: {s['submission']!r}   label: {s['label']!r}\n")
    text = tokenizer.decode(s["fullest"]["tokens"], skip_special_tokens=False)
    matches = list(TURN_RE.finditer(text))
    if not matches:
        print(text)
        return
    for turn_index, match in enumerate(matches):
        role, content = match.group(1), match.group(2).strip()
        if only_role and role != only_role:
            continue
        if max_turn_chars and len(content) > max_turn_chars:
            half = max_turn_chars // 2
            trimmed = len(content) - max_turn_chars
            content = (f"{content[:half]}\n{palette.dim}... "
                       f"[{trimmed} chars trimmed; --full to show]{palette.reset}\n"
                       f"{content[-half:]}")
        role_color = palette.role(role)
        print(f"{role_color}{palette.bold}--- [{turn_index}] {role} "
              f"{'-' * max(0, 60 - len(role))}{palette.reset}")
        print(content, end="\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("dump")
    parser.add_argument("--hf-checkpoint", default="/mnt/Qwen3.5-4B")
    parser.add_argument("-e", "--episode", type=int, default=None,
                        help="print the full transcript of episode N (0-based)")
    parser.add_argument("--failed", action="store_true",
                        help="summary table: only episodes with reward == 0")
    parser.add_argument("--max-turn-chars", type=int, default=1500,
                        help="trim each turn to this many chars in transcripts")
    parser.add_argument("--full", action="store_true", help="never trim turns")
    parser.add_argument("--role", default=None, choices=["assistant", "user", "system"],
                        help="transcript: show only this role's turns "
                             "(e.g. assistant = just the model's responses)")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    palette = Palette(enabled=sys.stdout.isatty() and not args.no_color)
    data = torch.load(args.dump, weights_only=False)
    samples = flatten(data["samples"])
    episodes = group_episodes(samples)

    if args.episode is None:
        shown = episodes
        if args.failed:
            shown = [e for e in episodes
                     if max(float(s.get("reward") or 0.0) for s in e) == 0]
        n_good = sum(1 for e in episodes
                     if max(float(s.get("reward") or 0.0) for s in e) > 0)
        print(f"{Path(args.dump).name}: {len(episodes)} episodes "
              f"({n_good} with reward > 0), {len(samples)} samples\n")
        print_table(shown, palette)
        print(f"\nuse -e <#> for a transcript, --failed to filter, | less -R to page")
        return 0

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    print_transcript(
        episodes[args.episode], tokenizer, palette,
        max_turn_chars=0 if args.full else args.max_turn_chars,
        only_role=args.role,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
