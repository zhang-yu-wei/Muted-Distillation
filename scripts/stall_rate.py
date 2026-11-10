"""Stall rate in dumped rollouts: turns that end without closing ``</think>``.

The aod2op-tail objective discards a turn whose think block never closes — it
has no visible tail to score, so it carries NO loss (``opd_rollout._tail_span``
returns None). Under ``PURE_OPD`` no reward term catches it either, leaving
"ramble inside <think> and stop" the one behaviour nothing pushes against. On
the SSB1 q38 tail arm those turns grew from 3.1% of all turns at eval step 0 to
28.4% by step 34, while episodes hitting the step cap went 83/400 -> 208/400.

``officeqa_opd_stall_penalty`` exists to bend that curve, and this script is
its readout. Run it over a run's ``debug_rollout/`` to get, per dump:

    stall%    turns with no </think>          <- the number the penalty targets
    nocmd%    turns with no fenced command    <- the visible symptom
    limits%   episodes ending limits_exceeded <- the cost in pass rate

Usage:
    python scripts/stall_rate.py <checkpoint-dir> [--eval | --train] [--limit N]

``<checkpoint-dir>`` is the run directory (the one holding ``debug_rollout``)
or the ``debug_rollout`` directory itself. Eval dumps carry every turn the
policy produced and are the honest population; train dumps hold only the
samples that survived masking, so their stall rate reads low by construction.
"""
from __future__ import annotations

import argparse
import gc
from collections import Counter
from pathlib import Path


def _dump_paths(root: Path, kind: str, limit: int | None) -> list[Path]:
    directory = root if root.name == "debug_rollout" else root / "debug_rollout"
    if not directory.is_dir():
        raise SystemExit(f"no debug_rollout directory under {root}")
    pattern = "rollout_eval_*.pt" if kind == "eval" else "rollout_[0-9]*.pt"
    paths = sorted(
        directory.glob(pattern),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not paths:
        raise SystemExit(f"no {kind} dumps in {directory}")
    return paths[-limit:] if limit else paths


def summarise(path: Path) -> dict[str, float]:
    import torch

    dump = torch.load(path, map_location="cpu", weights_only=False)
    turns = stalled = fenceless = 0
    episodes: dict[str, str | None] = {}
    for sample in dump["samples"]:
        response = sample["response"]
        turns += 1
        if "</think>" not in response:
            stalled += 1
        if "```" not in response:
            fenceless += 1
        meta = sample.get("metadata") or {}
        if "uid" in meta:
            episodes[meta["uid"]] = meta.get("exit_status")
    del dump
    gc.collect()
    exits = Counter(episodes.values())
    n_ep = max(len(episodes), 1)
    return {
        "step": int(path.stem.rsplit("_", 1)[-1]),
        "turns": turns,
        "stall": 100.0 * stalled / max(turns, 1),
        "nocmd": 100.0 * fenceless / max(turns, 1),
        "episodes": len(episodes),
        "limits": 100.0 * exits.get("limits_exceeded", 0) / n_ep,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--train",
        dest="kind",
        action="store_const",
        const="train",
        default="eval",
        help="read training dumps instead of eval dumps (see caveat above)",
    )
    parser.add_argument("--eval", dest="kind", action="store_const", const="eval")
    parser.add_argument(
        "--limit", type=int, default=None, help="only the last N dumps"
    )
    args = parser.parse_args()

    print(f"{'step':>5s} {'turns':>7s} {'stall%':>7s} {'nocmd%':>7s} "
          f"{'eps':>5s} {'limits%':>8s}")
    for path in _dump_paths(args.run_dir, args.kind, args.limit):
        row = summarise(path)
        print(f"{row['step']:5d} {row['turns']:7d} {row['stall']:6.1f}% "
              f"{row['nocmd']:6.1f}% {row['episodes']:5d} {row['limits']:7.1f}%",
              flush=True)


if __name__ == "__main__":
    main()
