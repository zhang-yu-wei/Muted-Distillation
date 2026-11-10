"""Summary contract of scripts/run_jobbench_eval.py.

The runner lives in ``scripts/`` (not the package), so importing it needs that
directory on the path — the same arrangement the launchers use when they run
``python3 scripts/run_jobbench_eval.py``.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

run_jobbench_eval = pytest.importorskip(
    "run_jobbench_eval", reason="requires scripts/ on sys.path"
)


def _result(uid: str, score: float, **overrides):
    values = dict(
        uid=uid,
        occupation="statisticians",
        score=score,
        pass_rate=score,
        graded=True,
        exit_status="Submitted",
        submitted=True,
        steps=5,
        output_files=["memo.md"],
        judge_model="grok-4.3",
        token_usage={"prompt_tokens": 100, "completion_tokens": 20},
    )
    values.update(overrides)
    return values


def _args(**overrides):
    values = dict(
        split="main",
        model="m",
        search_file_policy="omit",
        step_limit=60,
        temperature=0.0,
        samples_per_task=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_offline_eval_defaults_to_staged_search_files(tmp_path: Path):
    args = run_jobbench_eval.parse_args(
        ["--model", "m", "--output-dir", str(tmp_path / "results")]
    )
    assert args.search_file_policy == "stage"


def test_resume_refuses_results_from_another_search_file_policy(tmp_path: Path):
    task = SimpleNamespace(uid="JB__main__statisticians__task1")
    result_path = tmp_path / f"{task.uid}.json"
    result_path.write_text(
        '{"protocol": {"search_file_policy": "omit"}}\n', encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="Refusing to mix search-file policies"):
        run_jobbench_eval._validate_resumed_result_protocol(
            tmp_path, [task], [None], search_file_policy="stage"
        )


def test_summary_reports_mean_score_and_failure_shape():
    results = [
        _result("a", 1.0),
        _result("b", 0.0, exit_status="LimitsExceeded", submitted=False, output_files=[]),
        _result("c", 0.5, graded=False),
    ]
    summary = run_jobbench_eval.summarize(results, _args())
    assert summary["tasks"] == 3
    assert summary["mean_score"] == pytest.approx(0.5)
    assert summary["submitted_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["empty_output_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary["attempts_with_judge_failures"] == 1
    assert summary["exit_status_counts"]["LimitsExceeded"] == 1
    assert summary["score_by_occupation"]["statisticians"] == pytest.approx(0.5)
    # Token usage accumulates across attempts rather than keeping only the last.
    assert summary["token_usage_total"]["prompt_tokens"] == 300


def test_summary_reports_pass_at_k_for_multi_sample_runs():
    results = [
        _result("a", 1.0),
        _result("a", 0.0),
        _result("b", 0.4),
        _result("b", 0.6),
    ]
    summary = run_jobbench_eval.summarize(results, _args(samples_per_task=2))
    # pass@k counts only a fully passed rubric as a success: task a once, b never.
    assert summary["pass_at_k"]["pass@1"] == pytest.approx(0.25)
    assert summary["pass_at_k"]["pass@2"] == pytest.approx(0.5)
    assert summary["best_of_n_mean_score"] == pytest.approx((1.0 + 0.6) / 2)


def test_judge_config_prefers_explicit_flags():
    args = _args(
        judge_model="my-judge",
        judge_api_base="http://localhost:9000/v1",
        judge_timeout_sec=10.0,
        judge_max_retries=1,
        judge_concurrency=2,
        judge_max_completion_tokens=256,
        judge_images=False,
    )
    config = run_jobbench_eval.judge_config_from_args(args)
    assert config.model == "my-judge"
    assert config.api_base == "http://localhost:9000/v1"
    assert config.max_concurrent_rubrics == 2
    assert config.attach_images is False


def test_shard_flags_partition_the_task_list():
    parse = run_jobbench_eval.parse_args
    base = ["--model", "m", "--output-dir", "out"]

    args = parse(base)
    assert (args.num_shards, args.shard_index) == (1, 0)

    args = parse(base + ["--num-shards", "8", "--shard-index", "3"])
    assert (args.num_shards, args.shard_index) == (8, 3)

    with pytest.raises(SystemExit):
        parse(base + ["--num-shards", "0"])
    with pytest.raises(SystemExit):
        parse(base + ["--num-shards", "4", "--shard-index", "4"])

    # Position-modulo sharding partitions the ordered list exactly.
    tasks = [SimpleNamespace(uid=f"JB__{i}") for i in range(65)]
    shards = [
        [task for pos, task in enumerate(tasks) if pos % 8 == index]
        for index in range(8)
    ]
    flattened = sorted(task.uid for shard in shards for task in shard)
    assert flattened == sorted(task.uid for task in tasks)
    assert {len(shard) for shard in shards} == {8, 9}
