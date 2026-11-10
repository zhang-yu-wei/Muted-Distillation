import argparse
import csv
import json
from pathlib import Path

import pytest

from scripts import build_sft_dataset


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["uid", "question", "answer", "source_docs", "source_files", "difficulty"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row(uid: str, question: str, answer: str = "42") -> dict[str, str]:
    return {
        "uid": uid,
        "question": question,
        "answer": answer,
        "source_docs": "synthetic:test",
        "source_files": f"source_{uid}.txt",
        "difficulty": "easy",
    }


def _trace(uid: str, question: str, submission: str = "42") -> dict:
    return {
        "uid": uid,
        "question": question,
        "ground_truth": "42",
        "submission": submission,
        "status": "submitted",
        "steps": 2,
        # Deliberately stale: the builder must recompute agreement.
        "score": 0,
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": "I will inspect the source.\n```bash\nrg value corpus\n```",
                "reasoning_content": "Find the relevant row.",
            },
            {"role": "user", "content": "The value is 42."},
            {
                "role": "assistant",
                "content": "The result is ready.\n```bash\nsubmit \"42\"\n```",
                "reasoning_content": "The observation directly gives the answer.",
            },
            {"role": "exit", "content": "Submitted answer: 42"},
        ],
        "metadata": {
            "environment": {"command_history": [{"command": "rg value corpus"}]},
            "generated_turns": [
                {"usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}},
                {"usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}},
            ],
            "generation_backend": {"model_path": "teacher/model"},
        },
    }


def _args(tmp_path: Path, *, allow_missing: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        trace_dir=tmp_path / "traces",
        dataset_csv=tmp_path / "synthetic.csv",
        benchmark_csv=[tmp_path / "benchmark.csv"],
        provenance_jsonl=None,
        output=tmp_path / "sft.jsonl",
        manifest=tmp_path / "manifest.json",
        episode_policy="filtered",
        max_steps=40,
        max_duplicate_command_ratio=0.30,
        jaccard_threshold=0.50,
        uid_pattern=build_sft_dataset.DEFAULT_UID_PATTERN,
        skip_answer_collision=False,
        allow_missing_traces=allow_missing,
    )


def test_builds_one_masked_sample_per_assistant_turn(tmp_path):
    args = _args(tmp_path)
    args.trace_dir.mkdir()
    question = "What unique synthetic amount appears in the alpha ledger?"
    _write_csv(args.dataset_csv, [_row("SYN0001", question)])
    _write_csv(args.benchmark_csv[0], [_row("UID0001", "How many oranges are in a crate?", "7")])
    (args.trace_dir / "SYN0001.json").write_text(json.dumps(_trace("SYN0001", question)))

    manifest = build_sft_dataset.build(args)

    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert manifest["counts"]["retained_episodes"] == 1
    assert manifest["counts"]["sft_samples"] == 2
    assert manifest["usage"]["retained_traces"] == {
        "prompt_tokens": 30,
        "completion_tokens": 7,
        "total_tokens": 37,
    }
    assert len(rows) == 2
    assert rows[0]["messages"][-1]["content"].startswith("<think>Find the relevant row.</think>")
    assert rows[0]["messages"][-1]["step_loss_mask"] == 1
    assert [item["role"] for item in rows[1]["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert rows[1]["messages"][2]["step_loss_mask"] == 0
    assert "<think>" not in rows[1]["messages"][2]["content"]
    assert rows[1]["messages"][-1]["step_loss_mask"] == 1
    assert rows[1]["metadata"]["teacher_model"] == "teacher/model"


def test_recomputes_answer_agreement_instead_of_using_cached_score(tmp_path):
    args = _args(tmp_path)
    args.trace_dir.mkdir()
    good_question = "What unique synthetic amount appears in the alpha ledger?"
    bad_question = "What unique synthetic amount appears in the beta ledger?"
    _write_csv(
        args.dataset_csv,
        [_row("SYN0001", good_question), _row("SYN0002", bad_question)],
    )
    _write_csv(args.benchmark_csv[0], [_row("UID0001", "How many oranges are in a crate?", "7")])
    good = _trace("SYN0001", good_question)
    bad = _trace("SYN0002", bad_question, submission="99")
    bad["score"] = 1
    (args.trace_dir / "SYN0001.json").write_text(json.dumps(good))
    (args.trace_dir / "SYN0002.json").write_text(json.dumps(bad))

    manifest = build_sft_dataset.build(args)

    assert manifest["rejections"] == {"answer_disagreement": 1}
    decision = next(item for item in manifest["trace_decisions"] if item["uid"] == "SYN0002")
    assert decision["gemini_answer"] == "42"
    assert decision["teacher_submission"] == "99"
    assert decision["agreement_score"] == 0.0


def test_accepts_direct_command_turn_without_reasoning(tmp_path):
    args = _args(tmp_path)
    args.trace_dir.mkdir()
    question = "What unique synthetic amount appears in the direct-command ledger?"
    _write_csv(args.dataset_csv, [_row("SYN0001", question)])
    _write_csv(args.benchmark_csv[0], [_row("UID0001", "Unrelated benchmark question", "7")])
    trace = _trace("SYN0001", question)
    trace["messages"][2].pop("reasoning_content")
    trace["messages"][2]["content"] = "```bash\nrg value corpus\n```"
    (args.trace_dir / "SYN0001.json").write_text(json.dumps(trace))

    manifest = build_sft_dataset.build(args)

    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert manifest["counts"]["direct_command_sft_samples"] == 1
    assert rows[0]["messages"][-1]["content"] == "```bash\nrg value corpus\n```"
    assert rows[0]["metadata"]["target_has_reasoning"] is False


def test_rejects_unreasoned_turn_with_narration(tmp_path):
    args = _args(tmp_path)
    args.trace_dir.mkdir()
    good_question = "What unique synthetic amount appears in the alpha ledger?"
    bad_question = "What unique synthetic amount appears in the narrated ledger?"
    _write_csv(
        args.dataset_csv,
        [_row("SYN0001", good_question), _row("SYN0002", bad_question)],
    )
    _write_csv(args.benchmark_csv[0], [_row("UID0001", "Unrelated benchmark question", "7")])
    good = _trace("SYN0001", good_question)
    bad = _trace("SYN0002", bad_question)
    bad["messages"][2].pop("reasoning_content")
    (args.trace_dir / "SYN0001.json").write_text(json.dumps(good))
    (args.trace_dir / "SYN0002.json").write_text(json.dumps(bad))

    manifest = build_sft_dataset.build(args)

    assert manifest["rejections"] == {"unreasoned_non_direct_turn": 1}


def test_all_policy_keeps_failed_episode_and_raw_turns(tmp_path):
    args = _args(tmp_path)
    args.episode_policy = "all"
    args.trace_dir.mkdir()
    question = "What unique synthetic amount appears in the failed ledger?"
    _write_csv(args.dataset_csv, [_row("SYN0001", question)])
    _write_csv(args.benchmark_csv[0], [_row("UID0001", "Unrelated benchmark question", "7")])
    trace = _trace("SYN0001", question, submission="TIMEOUT")
    trace["status"] = "limits_exceeded"
    trace["steps"] = 50
    trace["messages"][2].pop("reasoning_content")
    trace["messages"][2]["content"] = "Narration without a command."
    trace["messages"][-1:-1] = [{"role": "user", "content": "Step limit reached."}]
    (args.trace_dir / "SYN0001.json").write_text(json.dumps(trace))

    manifest = build_sft_dataset.build(args)

    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert manifest["episode_policy"] == "all"
    assert manifest["counts"]["retained_episodes"] == 1
    assert manifest["rejections"] == {}
    assert rows[0]["messages"][-1]["content"] == "Narration without a command."
    assert rows[0]["metadata"]["score"] == 0.0
    assert rows[0]["metadata"]["status"] == "limits_exceeded"


def test_incomplete_trace_directory_fails_closed(tmp_path):
    args = _args(tmp_path)
    args.trace_dir.mkdir()
    _write_csv(
        args.dataset_csv,
        [_row("SYN0001", "Question one unique text"), _row("SYN0002", "Question two unique text")],
    )
    _write_csv(args.benchmark_csv[0], [_row("UID0001", "Unrelated benchmark question", "7")])
    trace = _trace("SYN0001", "Question one unique text")
    (args.trace_dir / "SYN0001.json").write_text(json.dumps(trace))

    with pytest.raises(ValueError, match="Trace directory is incomplete: 1 of 2"):
        build_sft_dataset.build(args)


def test_uid_pattern_flag_supports_benchmark_uids(tmp_path):
    args = _args(tmp_path)
    args.uid_pattern = r"UID\d+"
    args.trace_dir.mkdir()
    question = "What benchmark amount appears in the treasury ledger?"
    _write_csv(args.dataset_csv, [_row("UID0001", question)])
    _write_csv(args.benchmark_csv[0], [_row("UID9001", "Unrelated benchmark question", "7")])
    (args.trace_dir / "UID0001.json").write_text(json.dumps(_trace("UID0001", question)))
    # Non-trace JSON beside the traces (the runner's summary.json) is ignored.
    (args.trace_dir / "summary.json").write_text("{}")

    manifest = build_sft_dataset.build(args)

    assert manifest["counts"]["retained_episodes"] == 1
    assert manifest["filters"]["uid_pattern"] == r"UID\d+"


def test_dataset_uid_not_matching_pattern_fails_closed(tmp_path):
    args = _args(tmp_path)
    args.trace_dir.mkdir()
    _write_csv(args.dataset_csv, [_row("UID0001", "Some benchmark question")])
    _write_csv(args.benchmark_csv[0], [_row("UID9001", "Unrelated benchmark question", "7")])

    with pytest.raises(ValueError, match="must fully match"):
        build_sft_dataset.build(args)


def test_multi_sample_attempts_are_independent_episodes(tmp_path):
    args = _args(tmp_path)
    args.trace_dir.mkdir()
    question = "What unique synthetic amount appears in the alpha ledger?"
    _write_csv(
        args.dataset_csv,
        [_row("SYN0001", question), _row("SYN0002", question)],
    )
    _write_csv(args.benchmark_csv[0], [_row("UID0001", "Unrelated benchmark question", "7")])
    # SYN0001: attempts s0 and s10 are correct, s2 disagrees with the label.
    (args.trace_dir / "SYN0001.s0.json").write_text(json.dumps(_trace("SYN0001", question)))
    (args.trace_dir / "SYN0001.s2.json").write_text(
        json.dumps(_trace("SYN0001", question, submission="99"))
    )
    (args.trace_dir / "SYN0001.s10.json").write_text(json.dumps(_trace("SYN0001", question)))
    # SYN0002 duplicates SYN0001's question: still rejected across uids.
    (args.trace_dir / "SYN0002.s0.json").write_text(json.dumps(_trace("SYN0002", question)))

    manifest = build_sft_dataset.build(args)

    assert manifest["counts"]["found_traces"] == 2
    assert manifest["counts"]["found_trace_files"] == 4
    assert manifest["counts"]["retained_episodes"] == 2
    assert manifest["counts"]["retained_uids"] == 1
    assert manifest["counts"]["sft_samples"] == 4
    assert manifest["rejections"] == {
        "answer_disagreement": 1,
        "synthetic_question_exact": 1,
    }
    syn1_files = [
        item["trace_file"] for item in manifest["trace_decisions"] if item["uid"] == "SYN0001"
    ]
    assert syn1_files == ["SYN0001.s0.json", "SYN0001.s2.json", "SYN0001.s10.json"]
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    trace_files = {row["metadata"]["trace_file"] for row in rows}
    assert {Path(item).name for item in trace_files} == {"SYN0001.s0.json", "SYN0001.s10.json"}


def test_skip_answer_collision_flag(tmp_path):
    args = _args(tmp_path)
    args.trace_dir.mkdir()
    question = "What unique synthetic amount appears in the alpha ledger?"
    row = _row("SYN0001", question)
    benchmark = _row("UID0001", "A completely different benchmark question", "42")
    benchmark["source_files"] = row["source_files"]
    _write_csv(args.dataset_csv, [row])
    _write_csv(args.benchmark_csv[0], [benchmark])
    (args.trace_dir / "SYN0001.json").write_text(json.dumps(_trace("SYN0001", question)))

    with pytest.raises(ValueError, match="No traces passed"):
        build_sft_dataset.build(args)

    args.skip_answer_collision = True
    manifest = build_sft_dataset.build(args)

    assert manifest["counts"]["retained_episodes"] == 1
    assert manifest["filters"]["benchmark_answer_collision_enabled"] is False


def test_parse_args_validates_uid_pattern(tmp_path):
    argv = [
        "--trace-dir", str(tmp_path / "traces"),
        "--dataset-csv", str(tmp_path / "synthetic.csv"),
        "--benchmark-csv", str(tmp_path / "benchmark.csv"),
        "--output", str(tmp_path / "sft.jsonl"),
    ]
    args = build_sft_dataset.parse_args(argv)
    assert args.uid_pattern == build_sft_dataset.DEFAULT_UID_PATTERN
    assert args.skip_answer_collision is False

    with pytest.raises(SystemExit):
        build_sft_dataset.parse_args(argv + ["--uid-pattern", "["])


@pytest.mark.parametrize(
    ("content", "valid"),
    [
        ("text\n```bash\nrg x\n```", True),
        ("```python\nprint(1)\n```", False),
        ("```bash\nrg x\n```\ntrailing text", False),
        ("```bash\nrg x\n```\n```bash\nrg y\n```", False),
        ("```python\nprint(1)\n```\n```bash\nrg x\n```", False),
    ],
)
def test_valid_bash_content(content, valid):
    assert build_sft_dataset._valid_bash_content(content) is valid


@pytest.mark.parametrize(
    ("content", "valid"),
    [
        ("```bash\nrg x\n```", True),
        ("  \n```bash\nrg x\n```\n", True),
        ("I will search.\n```bash\nrg x\n```", False),
    ],
)
def test_valid_direct_bash_content(content, valid):
    assert build_sft_dataset._valid_direct_bash_content(content) is valid
