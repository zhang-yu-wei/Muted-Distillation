"""Convert the gated OfficeQA CSV into slime JSONL splits plus a manifest.

Output rows follow the slime data-source format::

    {"prompt": "...", "label": "...", "metadata": {"uid": ..., ...}}

The manifest records the source checksum, split seed, and per-split row counts
and checksums so a run can be tied to the exact dataset revision it used. The
gated benchmark rows themselves are never committed; outputs belong under the
gitignored ``data/`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from officeqa_rl.dataset import read_csv_rows, slime_row_from_csv_row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, required=True)
    parser.add_argument(
        "--exclude-csv",
        type=Path,
        help="Drop rows whose uid appears in this benchmark CSV (e.g. build "
        "full-minus-pro by excluding officeqa_pro.csv)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        default="train=1.0",
        help='Comma-separated name=fraction pairs, e.g. "train=0.8,dev=0.1,test=0.1"',
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--prefix", default="officeqa")
    return parser.parse_args(argv)


def parse_split_spec(spec: str) -> list[tuple[str, float]]:
    splits: list[tuple[str, float]] = []
    for part in spec.split(","):
        name, _, fraction_text = part.partition("=")
        name = name.strip()
        try:
            fraction = float(fraction_text)
        except ValueError:
            raise ValueError(f"Malformed split entry: {part!r}") from None
        if not name or fraction <= 0:
            raise ValueError(f"Malformed split entry: {part!r}")
        if any(existing == name for existing, _ in splits):
            raise ValueError(f"Duplicate split name: {name}")
        splits.append((name, fraction))
    total = sum(fraction for _, fraction in splits)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")
    return splits


def partition_rows(
    rows: list[dict[str, str]],
    splits: list[tuple[str, float]],
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    partitions: dict[str, list[dict[str, str]]] = {}
    cumulative = 0.0
    start = 0
    for index, (name, fraction) in enumerate(splits):
        cumulative += fraction
        end = len(shuffled) if index == len(splits) - 1 else round(cumulative * len(shuffled))
        partitions[name] = shuffled[start:end]
        start = end
    return partitions


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    splits = parse_split_spec(args.split)
    rows = read_csv_rows(args.dataset_csv)
    if not rows:
        raise ValueError(f"{args.dataset_csv} contains no rows")

    excluded = 0
    if args.exclude_csv is not None:
        excluded_uids = {
            (row.get("uid") or "").strip() for row in read_csv_rows(args.exclude_csv)
        }
        kept = [row for row in rows if (row.get("uid") or "").strip() not in excluded_uids]
        excluded = len(rows) - len(kept)
        rows = kept
        if not rows:
            raise ValueError(
                f"Every row of {args.dataset_csv} is excluded by {args.exclude_csv}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    for name, split_rows in partition_rows(rows, splits, args.seed).items():
        output_path = args.output_dir / f"{args.prefix}_{name}.jsonl"
        with output_path.open("w", encoding="utf-8") as file:
            for row in split_rows:
                file.write(json.dumps(slime_row_from_csv_row(row), ensure_ascii=False))
                file.write("\n")
        outputs[name] = {
            "file": output_path.name,
            "rows": len(split_rows),
            "sha256": sha256_of_file(output_path),
        }

    manifest = {
        "source_csv": str(args.dataset_csv.resolve()),
        "source_sha256": sha256_of_file(args.dataset_csv),
        "source_rows": len(rows),
        "exclude_csv": (
            str(args.exclude_csv.resolve()) if args.exclude_csv is not None else None
        ),
        "exclude_sha256": (
            sha256_of_file(args.exclude_csv) if args.exclude_csv is not None else None
        ),
        "excluded_rows": excluded,
        "split": args.split,
        "seed": args.seed,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "outputs": outputs,
    }
    manifest_path = args.output_dir / f"{args.prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    manifest = build(parse_args(argv))
    for name, output in manifest["outputs"].items():
        print(f"{name}: {output['rows']} rows -> {output['file']}")
    print(f"manifest: {manifest['source_rows']} source rows, seed {manifest['seed']}")


if __name__ == "__main__":
    sys.exit(main())
