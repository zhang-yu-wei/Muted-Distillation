import json
from pathlib import Path

import pytest

from officeqa_rl.dataset import (
    load_task,
    read_csv_rows,
    slime_row_from_csv_row,
    task_from_csv_row,
    task_from_slime_row,
)
from scripts.build_slime_dataset import build, parse_args, parse_split_spec

FIXTURE = Path(__file__).parent / "fixtures" / "dataset_tiny.csv"
FIXTURE_UIDS = {f"synthetic-00{index}" for index in range(1, 6)}


def build_fixture(tmp_path: Path, name: str, **overrides) -> dict:
    argv = [
        "--dataset-csv",
        str(FIXTURE),
        "--output-dir",
        str(tmp_path / name),
    ]
    for key, value in overrides.items():
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    return build(parse_args(argv))


def test_task_from_csv_row_maps_all_fields():
    task = task_from_csv_row(read_csv_rows(FIXTURE)[0])

    assert task.uid == "synthetic-001"
    assert task.question == "What revenue was reported in the synthetic bulletin?"
    assert task.answer == "123.45 million"
    assert task.source_files == "example_a.txt"
    assert task.difficulty == "easy"


def test_task_from_csv_row_treats_blank_answer_as_none():
    task = task_from_csv_row({"uid": "u1", "question": "Q?", "answer": "  "})

    assert task.answer is None
    assert task.difficulty is None


def test_task_from_csv_row_normalizes_crlf_in_multiline_fields(tmp_path):
    crlf = tmp_path / "crlf.csv"
    crlf.write_bytes(
        b'uid,question,answer\nu1,"First line.\r\nSecond line.",A\n'
    )

    task = task_from_csv_row(read_csv_rows(crlf)[0])

    assert task.question == "First line.\nSecond line."


def test_task_from_csv_row_rejects_missing_uid_or_question():
    with pytest.raises(ValueError, match="uid"):
        task_from_csv_row({"uid": " ", "question": "Q?", "answer": "A"})
    with pytest.raises(ValueError, match="question"):
        task_from_csv_row({"uid": "u1", "question": "", "answer": "A"})


def test_load_task_returns_single_row_and_errors_are_specific(tmp_path):
    task = load_task(FIXTURE, "synthetic-002")
    assert task.answer == "7,500"

    with pytest.raises(KeyError, match="synthetic-999"):
        load_task(FIXTURE, "synthetic-999")

    duplicated = tmp_path / "duplicated.csv"
    duplicated.write_text(
        "uid,question,answer\nu1,Q?,A\nu1,Q again?,B\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not unique"):
        load_task(duplicated, "u1")


def test_read_csv_rows_rejects_missing_columns(tmp_path):
    broken = tmp_path / "broken.csv"
    broken.write_text("uid,question\nu1,Q?\n", encoding="utf-8")

    with pytest.raises(ValueError, match="answer"):
        read_csv_rows(broken)


def test_slime_row_keeps_answer_only_under_label():
    row = read_csv_rows(FIXTURE)[1]
    slime_row = slime_row_from_csv_row(row)

    assert slime_row["label"] == "7,500"
    agent_visible = json.dumps(
        {"prompt": slime_row["prompt"], "metadata": slime_row["metadata"]}
    )
    assert "7,500" not in agent_visible
    assert slime_row["metadata"] == {
        "uid": "synthetic-002",
        "source_files": "example_b.txt",
        "difficulty": "hard",
    }


def test_slime_row_rejects_row_without_answer():
    with pytest.raises(ValueError, match="cannot be graded"):
        slime_row_from_csv_row({"uid": "u1", "question": "Q?", "answer": ""})


def test_csv_and_slime_rows_produce_identical_tasks():
    for row in read_csv_rows(FIXTURE):
        assert task_from_slime_row(slime_row_from_csv_row(row)) == task_from_csv_row(row)


def test_parse_split_spec_validates_entries():
    assert parse_split_spec("train=0.8,dev=0.2") == [("train", 0.8), ("dev", 0.2)]

    with pytest.raises(ValueError, match="sum to 1.0"):
        parse_split_spec("train=0.8,dev=0.1")
    with pytest.raises(ValueError, match="Malformed"):
        parse_split_spec("train")
    with pytest.raises(ValueError, match="Duplicate"):
        parse_split_spec("train=0.5,train=0.5")


def test_build_writes_disjoint_splits_and_accurate_manifest(tmp_path):
    manifest = build_fixture(tmp_path, "out", split="train=0.6,test=0.4", seed=3)

    assert manifest["source_rows"] == 5
    assert manifest["outputs"]["train"]["rows"] == 3
    assert manifest["outputs"]["test"]["rows"] == 2

    uids_by_split = {}
    for name, output in manifest["outputs"].items():
        lines = (tmp_path / "out" / output["file"]).read_text(encoding="utf-8").splitlines()
        assert len(lines) == output["rows"]
        rows = [json.loads(line) for line in lines]
        uids_by_split[name] = {row["metadata"]["uid"] for row in rows}
        for row in rows:
            assert set(row) == {"prompt", "label", "metadata"}

    assert uids_by_split["train"] | uids_by_split["test"] == FIXTURE_UIDS
    assert not uids_by_split["train"] & uids_by_split["test"]

    manifest_on_disk = json.loads(
        (tmp_path / "out" / "officeqa_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_on_disk["outputs"] == manifest["outputs"]
    assert manifest_on_disk["seed"] == 3


def test_build_excludes_uids_from_another_csv(tmp_path):
    exclude = tmp_path / "exclude.csv"
    exclude.write_text(
        "uid,question,answer\n"
        "synthetic-002,ignored,ignored\n"
        "synthetic-004,ignored,ignored\n",
        encoding="utf-8",
    )

    manifest = build_fixture(tmp_path, "out", exclude_csv=exclude)

    assert manifest["source_rows"] == 3
    assert manifest["excluded_rows"] == 2
    assert manifest["exclude_csv"] == str(exclude.resolve())
    lines = (tmp_path / "out" / "officeqa_train.jsonl").read_text(encoding="utf-8").splitlines()
    uids = {json.loads(line)["metadata"]["uid"] for line in lines}
    assert uids == {"synthetic-001", "synthetic-003", "synthetic-005"}


def test_build_is_deterministic_for_a_seed(tmp_path):
    first = build_fixture(tmp_path, "first", split="train=0.6,test=0.4", seed=7)
    second = build_fixture(tmp_path, "second", split="train=0.6,test=0.4", seed=7)
    reshuffled = build_fixture(tmp_path, "third", split="train=0.6,test=0.4", seed=8)

    for name in ("train", "test"):
        assert first["outputs"][name]["sha256"] == second["outputs"][name]["sha256"]
    assert any(
        first["outputs"][name]["sha256"] != reshuffled["outputs"][name]["sha256"]
        for name in ("train", "test")
    )


def test_build_output_loads_through_slime_dataset(tmp_path):
    slime_data = pytest.importorskip(
        "slime.utils.data",
        reason="slime and its dependencies are not installed in this environment",
    )
    manifest = build_fixture(tmp_path, "out")

    dataset = slime_data.Dataset(
        str(tmp_path / "out" / manifest["outputs"]["train"]["file"]),
        None,  # tokenizer
        None,  # processor
        None,  # max_length
        prompt_key="prompt",
        label_key="label",
        metadata_key="metadata",
    )

    assert len(dataset.samples) == 5
    sample = dataset.samples[0]
    assert sample.label
    assert sample.metadata["uid"] in FIXTURE_UIDS
    assert sample.label not in sample.prompt
