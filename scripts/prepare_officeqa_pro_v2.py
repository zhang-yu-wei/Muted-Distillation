#!/usr/bin/env python3
"""Download and transform the gated OfficeQA-Pro v2 text corpus.

The full parsed corpus is retained so retrieval sees all 1,435 documents, not
only the gold-source subset. Large files stay under the requested local-SSD
destination. A ``.ready`` marker is written only after all JSONs transform.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import snapshot_download


REPO_ID = "databricks/officeqa-pro-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/localssd/datasets/officeqa-pro-v2"),
    )
    return parser.parse_args()


def _sanitize(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").replace("|", r"\|").strip()


def _headers(columns: Any) -> list[str]:
    if isinstance(columns, pd.MultiIndex):
        headers = [
            " > ".join(
                str(part).strip()
                for part in column
                if part is not None and str(part).strip()
            )
            for column in columns
        ]
    else:
        headers = [str(column).strip() for column in columns]

    counts: dict[str, int] = {}
    unique: list[str] = []
    for index, header in enumerate(headers, start=1):
        name = header or f"col_{index}"
        counts[name] = counts.get(name, 0) + 1
        unique.append(name if counts[name] == 1 else f"{name}_{counts[name]}")
    return unique


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    headers = _headers(frame.columns)
    lines = [
        "| " + " | ".join(_sanitize(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_sanitize(value) for value in row) + " |")
    return "\n".join(lines)


def _element_text(content: str) -> list[str]:
    if "<table" not in content.lower():
        return re.sub(r"\r\n?", "\n", content.strip()).split("\n")
    try:
        return [
            _dataframe_to_markdown(frame)
            for frame in pd.read_html(io.StringIO(content))
        ]
    except Exception:
        return [f"```html\n{content.strip()}\n```"]


def transform_document(source: Path, destination: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    elements = (payload.get("document") or {}).get("elements") or []
    lines: list[str] = []
    for element in elements:
        content = element.get("content") if isinstance(element, dict) else None
        if not isinstance(content, str) or not content.strip():
            continue
        lines.extend(_element_text(content))
        lines.append("")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ready = output_dir / ".ready"
    ready.unlink(missing_ok=True)

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=output_dir,
        allow_patterns=["officeqa_pro_v2.csv", "parsed_corpus/jsons/*.json"],
    )

    dataset_csv = output_dir / "officeqa_pro_v2.csv"
    with dataset_csv.open(newline="", encoding="utf-8") as handle:
        task_count = sum(1 for _ in csv.DictReader(handle))
    if task_count != 90:
        raise RuntimeError(f"Expected 90 OfficeQA-Pro v2 tasks, found {task_count}")

    json_dir = output_dir / "parsed_corpus" / "jsons"
    sources = sorted(json_dir.glob("*.json"))
    if len(sources) < 1_400:
        raise RuntimeError(f"Expected the full parsed corpus, found {len(sources)} JSONs")

    transformed_dir = output_dir / "transformed"
    for index, source in enumerate(sources, start=1):
        destination = transformed_dir / f"{source.stem}.txt"
        if not destination.is_file() or destination.stat().st_size == 0:
            transform_document(source, destination)
        if index % 100 == 0 or index == len(sources):
            print(f"transformed {index}/{len(sources)}", flush=True)

    transformed_count = sum(1 for path in transformed_dir.glob("*.txt") if path.stat().st_size)
    if transformed_count != len(sources):
        raise RuntimeError(
            f"Expected {len(sources)} transformed files, found {transformed_count}"
        )

    ready.write_text(
        json.dumps(
            {
                "repo_id": REPO_ID,
                "tasks": task_count,
                "parsed_documents": len(sources),
                "transformed_documents": transformed_count,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"OfficeQA-Pro v2 ready at {output_dir}")


if __name__ == "__main__":
    main()
