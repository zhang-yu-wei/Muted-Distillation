import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from officeqa_rl.backends.vllm import VLLMGenerationBackend
from officeqa_rl.types import GeneratedTurn
import scripts.run_full_corpus_inference as inference_module
from scripts.run_full_corpus_inference import (
    load_task,
    parse_args,
    run_full_eval,
    run_inference,
)


FIXTURE = Path(__file__).parent / "fixtures" / "inference_smoke.csv"


class ScriptedTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


class ScriptedLLM:
    instances = []

    def __init__(self, engine_args):
        self.init_kwargs = engine_args.kwargs
        self.tokenizer = ScriptedTokenizer()
        self.responses = iter(
            [
                '```bash\ngrep -n "revenue" officeqa_corpus/example.txt\n```',
                '```bash\nsubmit "123.45 million"\n```',
            ]
        )
        self.data_parallel_ranks = []
        self.instances.append(self)

    @classmethod
    def from_engine_args(cls, engine_args):
        return cls(engine_args)

    def get_tokenizer(self):
        return self.tokenizer

    async def generate(
        self,
        prompt,
        sampling_params,
        request_id,
        *,
        data_parallel_rank=None,
    ):
        del prompt, sampling_params, request_id
        self.data_parallel_ranks.append(data_parallel_rank)
        text = next(self.responses)
        completion = SimpleNamespace(
            text=text,
            token_ids=list(range(len(text))),
            finish_reason="stop",
        )
        yield SimpleNamespace(prompt_token_ids=[1, 2], outputs=[completion])


class ScriptedEngineArgs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class ScriptedSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class ScriptedGeminiBackend:
    instances = []

    def __init__(self, model_name, **kwargs):
        self.model_path = model_name
        self.init_kwargs = kwargs
        self.responses = iter(
            [
                '```bash\ngrep -n "revenue" officeqa_corpus/example.txt\n```',
                '```bash\nsubmit "123.45 million"\n```',
            ]
        )
        self.n_calls = 0
        self.instances.append(self)

    async def generate(self, messages, sampling_params):
        del messages, sampling_params
        self.n_calls += 1
        return GeneratedTurn(text=next(self.responses), finish_reason="STOP")

    def get_stats(self):
        return {"n_calls": self.n_calls, "cost": 0.0}


def install_fake_docker(tmp_path: Path, monkeypatch) -> Path:
    log = tmp_path / "docker.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        """#!/bin/bash
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
case "$1" in
  run) printf 'fake-inference-container\\n' ;;
  exec) printf '7:The reported revenue was 123.45 million.\\n' ;;
  rm) exit 0 ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


def test_load_task_selects_one_synthetic_uid():
    task = load_task(FIXTURE, "synthetic-inference-001")

    assert task.question == "What revenue was reported?"
    assert task.answer == "123.45 million"
    assert task.source_files == "example.txt"


@pytest.mark.asyncio
async def test_inference_only_pipeline_is_deterministic(tmp_path, monkeypatch):
    ScriptedLLM.instances = []
    fake_vllm = SimpleNamespace(
        AsyncLLMEngine=ScriptedLLM,
        AsyncEngineArgs=ScriptedEngineArgs,
        SamplingParams=ScriptedSamplingParams,
    )
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    docker_log = install_fake_docker(tmp_path, monkeypatch)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "example.txt").write_text(
        "The reported revenue was 123.45 million.\n",
        encoding="utf-8",
    )
    output = tmp_path / "result.json"
    args = parse_args(
        [
            "--dataset-csv",
            str(FIXTURE),
            "--uid",
            "synthetic-inference-001",
            "--corpus-dir",
            str(corpus_dir),
            "--model-path",
            "fake/model",
            "--workspace-root",
            str(tmp_path / "workspaces"),
            "--sandbox-image",
            "officeqa-sandbox:test",
            "--run-id",
            "inference-smoke-run",
            "--output",
            str(output),
            "--step-limit",
            "4",
            "--max-tokens",
            "128",
            "--data-parallel-size",
            "8",
        ]
    )

    result = await run_inference(args)

    assert args.enable_visual_pdf_tool is False
    assert args.page_text_dir is None
    assert args.pdf_dir is None
    assert result["submission"] == "123.45 million"
    assert result["status"] == "submitted"
    assert result["steps"] == 2
    assert result["score"] == 1.0
    assert result["metadata"]["generation_backend"]["stats"]["n_calls"] == 2
    assert "visual_pdf_tool_enabled" not in result["metadata"]["environment"]
    assert "automatic_page_retrievals" not in result["metadata"]["environment"]
    assert json.loads(output.read_text(encoding="utf-8"))["score"] == 1.0
    docker_calls = docker_log.read_text(encoding="utf-8")
    assert "run -d" in docker_calls
    assert "exec -w /workspace" in docker_calls
    assert "rm -f fake-inference-container" in docker_calls
    assert "officeqa_pages" not in docker_calls
    assert len(set(ScriptedLLM.instances[0].data_parallel_ranks)) == 1
    assert ScriptedLLM.instances[0].data_parallel_ranks[0] is not None


@pytest.mark.asyncio
async def test_gemini_inference_pipeline_uses_same_docker_evaluation(
    tmp_path,
    monkeypatch,
):
    ScriptedGeminiBackend.instances = []
    monkeypatch.setattr(
        inference_module,
        "GeminiGenerationBackend",
        ScriptedGeminiBackend,
    )
    docker_log = install_fake_docker(tmp_path, monkeypatch)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    args = parse_args(
        [
            "--generation-backend",
            "gemini",
            "--model",
            "gemini-3.5-flash",
            "--dataset-csv",
            str(FIXTURE),
            "--uid",
            "synthetic-inference-001",
            "--corpus-dir",
            str(corpus_dir),
            "--workspace-root",
            str(tmp_path / "workspaces"),
            "--sandbox-image",
            "officeqa-sandbox:test",
            "--run-id",
            "gemini-inference-smoke",
            "--gemini-api-version",
            "v1beta",
            "--gemini-thinking-budget",
            "0",
        ]
    )

    result = await run_inference(args)

    assert result["submission"] == "123.45 million"
    assert result["score"] == 1.0
    assert result["metadata"]["generation_backend"]["kind"] == "gemini"
    backend = ScriptedGeminiBackend.instances[0]
    assert backend.model_path == "gemini-3.5-flash"
    assert backend.init_kwargs["api_version"] == "v1beta"
    assert backend.init_kwargs["thinking_budget"] == 0
    docker_calls = docker_log.read_text(encoding="utf-8")
    assert "exec -w /workspace" in docker_calls
    assert "rm -f fake-inference-container" in docker_calls


class SubmitByQuestionBackend:
    """Answer '11' to the 'first' question and '22' to everything else."""

    instances = []

    def __init__(self, model_name, **kwargs):
        self.model_path = model_name
        self.init_kwargs = kwargs
        self.n_calls = 0
        type(self).instances.append(self)

    async def generate(self, messages, sampling_params):
        del sampling_params
        self.n_calls += 1
        question = messages[1]["content"]
        answer = "11" if "first" in question else "22"
        return GeneratedTurn(
            text=f'```bash\nsubmit "{answer}"\n```',
            usage={
                "prompt_tokens": 400,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 300},
            },
        )

    def get_stats(self):
        return {"n_calls": self.n_calls}


def write_multi_row_csv(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text(
        "uid,question,answer,source_files,difficulty\n"
        "multi-001,What is the first synthetic value?,11,example.txt,easy\n"
        "multi-002,What is the second synthetic value?,99,example.txt,hard\n",
        encoding="utf-8",
    )
    return dataset


def full_eval_args(tmp_path: Path, dataset: Path) -> list[str]:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    return [
        "--generation-backend",
        "gemini",
        "--model",
        "gemini-3.5-flash",
        "--dataset-csv",
        str(dataset),
        "--corpus-dir",
        str(corpus_dir),
        "--workspace-root",
        str(tmp_path / "workspaces"),
        "--sandbox-runtime",
        "local",
        "--output-dir",
        str(tmp_path / "results"),
        "--concurrency",
        "2",
        "--step-limit",
        "3",
    ]


@pytest.mark.asyncio
async def test_full_eval_writes_per_uid_results_and_summary(tmp_path, monkeypatch, caplog):
    SubmitByQuestionBackend.instances = []
    monkeypatch.setattr(
        inference_module,
        "GeminiGenerationBackend",
        SubmitByQuestionBackend,
    )
    dataset = write_multi_row_csv(tmp_path)

    with caplog.at_level("INFO", logger="officeqa_rl.inference"):
        summary = await run_full_eval(parse_args(full_eval_args(tmp_path, dataset)))

    progress_lines = [line for line in caplog.messages if line.startswith("progress:")]
    assert len(progress_lines) == 2  # one per finished trajectory
    assert "2/2 trajectories this run (0 failed) | overall 2/2 (100%)" in progress_lines[-1]
    assert "mean score this run 0.500 over 2 scored" in progress_lines[-1]

    assert summary["total_rows"] == 2
    assert summary["completed"] == 2
    assert summary["resumed"] == 0
    assert summary["failed"] == {}
    assert summary["mean_score"] == 0.5  # 11 is right, 22 != 99 is wrong
    assert summary["status_counts"] == {"submitted": 2}

    # One model call per attempt, two attempts: 800 prompt tokens of which 600
    # were cache hits, and 40 generated.
    assert summary["token_usage"]["prompt_tokens"] == 800
    assert summary["token_usage"]["cached_input_tokens"] == 600
    assert summary["token_usage"]["input_tokens"] == 200
    assert summary["token_usage"]["output_tokens"] == 40
    assert summary["token_usage"]["episodes"] == 2
    assert summary["token_usage"]["output_tokens_per_task"] == 20.0

    results_dir = tmp_path / "results"
    per_uid = json.loads((results_dir / "multi-001.json").read_text(encoding="utf-8"))
    assert per_uid["submission"] == "11"
    assert per_uid["score"] == 1.0
    assert per_uid["token_usage"]["prompt_tokens"] == 400
    assert per_uid["token_usage"]["cached_input_tokens"] == 300
    assert per_uid["token_usage"]["model_calls"] == 1
    summary_on_disk = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_on_disk["results"]["multi-002"]["score"] == 0.0
    assert SubmitByQuestionBackend.instances[0].n_calls == 2


@pytest.mark.asyncio
async def test_full_eval_resumes_from_existing_results(tmp_path, monkeypatch):
    SubmitByQuestionBackend.instances = []
    monkeypatch.setattr(
        inference_module,
        "GeminiGenerationBackend",
        SubmitByQuestionBackend,
    )
    dataset = write_multi_row_csv(tmp_path)
    argv = full_eval_args(tmp_path, dataset)

    first = await run_full_eval(parse_args(argv))
    assert first["completed"] == 2

    second = await run_full_eval(parse_args(argv))

    assert second["resumed"] == 2
    assert second["completed"] == 2
    assert second["mean_score"] == first["mean_score"]
    # A fully resumed run only rebuilds the summary and does not create a
    # generation backend.
    assert len(SubmitByQuestionBackend.instances) == 1


class FailsOncePerQuestionBackend(SubmitByQuestionBackend):
    """Simulate an exhausted-retries API error on each question's first turn."""

    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        self.seen_questions = set()

    async def generate(self, messages, sampling_params):
        # Key on the question word, not the whole task message: the message
        # embeds the workspace path, which changes on every retry attempt.
        question_key = "first" if "first" in messages[1]["content"] else "second"
        if question_key not in self.seen_questions:
            self.seen_questions.add(question_key)
            raise RuntimeError("Gemini call failed after retries: 429 RESOURCE_EXHAUSTED")
        return await super().generate(messages, sampling_params)


class ContextOverflowBackend(SubmitByQuestionBackend):
    """Reject every request with SGLang's actual context-overflow wording."""

    async def generate(self, messages, sampling_params):
        del messages, sampling_params
        self.n_calls += 1
        raise RuntimeError(
            "Input length (243493 tokens) exceeds the maximum allowed length "
            "(243450 tokens)."
        )


@pytest.mark.asyncio
async def test_full_eval_retry_passes_recover_failed_trajectories(tmp_path, monkeypatch):
    SubmitByQuestionBackend.instances = []
    monkeypatch.setattr(
        inference_module,
        "GeminiGenerationBackend",
        FailsOncePerQuestionBackend,
    )
    dataset = write_multi_row_csv(tmp_path)
    argv = [
        *full_eval_args(tmp_path, dataset),
        "--failure-retries",
        "1",
        "--failure-retry-delay-sec",
        "0",
    ]

    summary = await run_full_eval(parse_args(argv))

    assert summary["failed"] == {}
    assert summary["completed"] == 2
    assert summary["status_counts"] == {"submitted": 2}


@pytest.mark.asyncio
async def test_full_eval_reports_failures_when_retries_are_exhausted(tmp_path, monkeypatch):
    SubmitByQuestionBackend.instances = []
    monkeypatch.setattr(
        inference_module,
        "GeminiGenerationBackend",
        FailsOncePerQuestionBackend,
    )
    dataset = write_multi_row_csv(tmp_path)
    argv = [*full_eval_args(tmp_path, dataset), "--failure-retries", "0"]

    summary = await run_full_eval(parse_args(argv))

    assert summary["completed"] == 0
    assert set(summary["failed"]) == {"multi-001", "multi-002"}
    assert "RESOURCE_EXHAUSTED" in summary["failed"]["multi-001"]
    assert summary["failure_counts"] == {"resource_exhausted": 2}


@pytest.mark.asyncio
async def test_full_eval_persists_context_overflow_as_zero_without_retry(
    tmp_path, monkeypatch
):
    ContextOverflowBackend.instances = []
    monkeypatch.setattr(
        inference_module,
        "GeminiGenerationBackend",
        ContextOverflowBackend,
    )
    dataset = write_multi_row_csv(tmp_path)
    argv = [
        *full_eval_args(tmp_path, dataset),
        "--failure-retries",
        "2",
        "--failure-retry-delay-sec",
        "0",
    ]

    first = await run_full_eval(parse_args(argv))

    backend = ContextOverflowBackend.instances[0]
    assert backend.n_calls == 2  # one request per row, no backend or eval retry
    assert first["failed"] == {}
    assert first["completed"] == 2
    assert first["scored"] == 2
    assert first["mean_score"] == 0.0
    assert first["status_counts"] == {"context_length_exceeded": 2}
    for uid in ("multi-001", "multi-002"):
        payload = json.loads(
            (tmp_path / "results" / f"{uid}.json").read_text(encoding="utf-8")
        )
        assert payload["score"] == 0.0
        assert payload["submission"] is None
        assert payload["status"] == "context_length_exceeded"
        assert payload["metadata"]["terminal_failure"]["type"] == "context_length"

    second = await run_full_eval(parse_args(argv))

    assert second["resumed"] == 2
    assert second["mean_score"] == 0.0
    assert backend.n_calls == 2
    assert len(ContextOverflowBackend.instances) == 1


def test_classify_failure_buckets():
    classify = inference_module._classify_failure
    assert classify("Gemini call failed after retries: 429 RESOURCE_EXHAUSTED") == "resource_exhausted"
    assert classify("503 UNAVAILABLE. high demand") == "unavailable"
    assert (
        classify(
            "This model's maximum context length is 262144 tokens. However, "
            "you requested 0 output tokens and your prompt contains at least "
            "262145 input tokens"
        )
        == "context_length"
    )
    assert (
        classify(
            "Input length (243493 tokens) exceeds the maximum allowed length "
            "(243450 tokens)."
        )
        == "context_length"
    )
    # A token count containing the digits 429/503 must not match the rate buckets.
    assert classify("RuntimeError: prompt has 14290 tokens") == "other"
    assert classify("TimeoutError: sandbox boot") == "other"


def test_parse_args_rejects_mismatched_mode_flags(tmp_path, capsys):
    base = [
        "--dataset-csv",
        "d.csv",
        "--corpus-dir",
        "corpus",
        "--model",
        "m",
    ]
    with pytest.raises(SystemExit):
        parse_args([*base, "--output", "out.json"])
    with pytest.raises(SystemExit):
        parse_args([*base, "--uid", "u1", "--output-dir", "results"])
    capsys.readouterr()


def test_pass_at_k_estimator():
    # All attempts succeed / fail.
    assert inference_module._pass_at_k(4, 4, 1) == 1.0
    assert inference_module._pass_at_k(4, 0, 3) == 0.0
    # 2 successes out of 4: pass@1 is the success rate, pass@4 is certain.
    assert inference_module._pass_at_k(4, 2, 1) == pytest.approx(0.5)
    assert inference_module._pass_at_k(4, 2, 4) == 1.0
    # 1 of 3, k=2: 1 - C(2,2)/C(3,2) = 2/3.
    assert inference_module._pass_at_k(3, 1, 2) == pytest.approx(2 / 3)


def test_samples_per_task_argument_validation(tmp_path, capsys):
    base = [
        "--dataset-csv", "d.csv",
        "--corpus-dir", str(tmp_path),
        "--model", "m",
        "--output-dir", str(tmp_path / "results"),
    ]
    # Temperature now defaults to 1.0, so pass@k works without extra flags.
    args = parse_args([*base, "--samples-per-task", "4"])
    assert args.samples_per_task == 4
    assert args.temperature == 1.0
    # Greedy decoding still cannot produce independent attempts.
    with pytest.raises(SystemExit):
        parse_args([*base, "--samples-per-task", "4", "--temperature", "0"])
    with pytest.raises(SystemExit):
        parse_args([*base, "--samples-per-task", "0"])
    args = parse_args([*base, "--samples-per-task", "4", "--temperature", "0.7"])
    assert args.samples_per_task == 4
    capsys.readouterr()


def test_data_parallel_argument_validation(tmp_path, capsys):
    base = [
        "--dataset-csv",
        "d.csv",
        "--corpus-dir",
        str(tmp_path),
        "--model",
        "m",
        "--output-dir",
        str(tmp_path / "results"),
    ]
    with pytest.raises(SystemExit):
        parse_args([*base, "--data-parallel-size", "0"])
    args = parse_args([*base, "--data-parallel-size", "8"])
    assert args.data_parallel_size == 8
    assert args.enable_expert_parallel is False
    assert args.enable_prefix_caching is True
    assert args.concurrency == 32
    four_b_args = parse_args([*base, "--model", "Qwen/Qwen3.5-4B"])
    assert four_b_args.concurrency == 48
    moe_args = parse_args([*base, "--model", "google/gemma-4-26B-A4B-it"])
    assert moe_args.concurrency == 32
    explicit_args = parse_args(
        [*base, "--model", "Qwen/Qwen3.5-4B", "--concurrency", "7"]
    )
    assert explicit_args.concurrency == 7
    args = parse_args([*base, "--enable-prefix-caching"])
    assert args.enable_prefix_caching is True
    args = parse_args([*base, "--no-enable-prefix-caching"])
    assert args.enable_prefix_caching is False
    api_args = parse_args([*base, "--generation-backend", "gemini"])
    assert api_args.concurrency == 4
    capsys.readouterr()


class FlakySubmitBackend(SubmitByQuestionBackend):
    """Answer correctly on every second generate call, per question."""

    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        self.calls_per_question = {}

    async def generate(self, messages, sampling_params):
        question_key = "first" if "first" in messages[1]["content"] else "second"
        n = self.calls_per_question.get(question_key, 0)
        self.calls_per_question[question_key] = n + 1
        if n % 2 == 1:
            return GeneratedTurn(text='```bash\nsubmit "wrong"\n```')
        return await super().generate(messages, sampling_params)


@pytest.mark.asyncio
async def test_full_eval_pass_at_k(tmp_path, monkeypatch):
    SubmitByQuestionBackend.instances = []
    monkeypatch.setattr(inference_module, "GeminiGenerationBackend", FlakySubmitBackend)
    dataset = write_multi_row_csv(tmp_path)
    argv = [
        *full_eval_args(tmp_path, dataset),
        "--samples-per-task", "2",
        "--temperature", "0.7",
    ]

    summary = await run_full_eval(parse_args(argv))

    # multi-001 ("11" is correct): attempt 0 right, attempt 1 wrong -> c=1/2.
    # multi-002 (needs "99", model says "22"/"wrong"): both attempts wrong.
    assert summary["samples_per_task"] == 2
    assert summary["completed"] == 2
    entry = summary["results"]["multi-001"]
    assert entry["n_attempts"] == 2
    assert entry["n_success"] == 1
    assert summary["results"]["multi-002"]["n_success"] == 0
    assert summary["pass_at_k"]["1"] == pytest.approx(0.25)  # (0.5 + 0) / 2
    assert summary["pass_at_k"]["2"] == pytest.approx(0.5)  # (1.0 + 0) / 2
    assert summary["mean_score"] == pytest.approx(0.25)

    results_dir = tmp_path / "results"
    assert (results_dir / "multi-001.s0.json").is_file()
    assert (results_dir / "multi-001.s1.json").is_file()
    assert not (results_dir / "multi-001.json").is_file()

    # Resume: a second run skips every attempt.
    second = await run_full_eval(parse_args(argv))
    assert second["resumed"] == 4
    assert second["pass_at_k"] == summary["pass_at_k"]


def test_gemini_api_keys_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS_FILE", raising=False)
    assert inference_module._gemini_api_keys_from_env() is None

    monkeypatch.setenv("GEMINI_API_KEYS", "key-a, key-b\nkey-c")
    assert inference_module._gemini_api_keys_from_env() == ["key-a", "key-b", "key-c"]

    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("# project keys\nkey-1\n\n  key-2  \n")
    monkeypatch.setenv("GEMINI_API_KEYS_FILE", str(keys_file))
    # The file takes precedence over GEMINI_API_KEYS.
    assert inference_module._gemini_api_keys_from_env() == ["key-1", "key-2"]

    keys_file.write_text("# only comments\n\n")
    with pytest.raises(SystemExit):
        inference_module._gemini_api_keys_from_env()


def test_build_kimi_backend_uses_single_key_configuration(monkeypatch):
    ScriptedGeminiBackend.instances = []
    monkeypatch.setattr(
        inference_module,
        "KimiGenerationBackend",
        ScriptedGeminiBackend,
    )
    args = parse_args(
        [
            "--generation-backend",
            "kimi",
            "--model",
            "kimi-k3",
            "--dataset-csv",
            "dataset.csv",
            "--corpus-dir",
            "corpus",
            "--kimi-reasoning-effort",
            "high",
        ]
    )

    backend = inference_module.build_generation_backend(args)

    assert backend.model_path == "kimi-k3"
    assert "api_keys" not in backend.init_kwargs
    assert backend.init_kwargs["reasoning_effort"] == "high"
    assert backend.init_kwargs["base_url"] == "https://api.moonshot.ai/v1"


def test_build_openai_server_backend_uses_base_url(monkeypatch):
    ScriptedGeminiBackend.instances = []
    monkeypatch.setattr(
        inference_module,
        "OpenAIServerGenerationBackend",
        ScriptedGeminiBackend,
    )
    args = parse_args(
        [
            "--generation-backend",
            "openai-server",
            "--model",
            "/mnt/local-ssd/GLM-5.2",
            "--dataset-csv",
            "dataset.csv",
            "--corpus-dir",
            "corpus",
            "--openai-base-url",
            "http://10.0.0.1:8000/v1",
        ]
    )

    backend = inference_module.build_generation_backend(args)

    assert backend.model_path == "/mnt/local-ssd/GLM-5.2"
    assert backend.init_kwargs["base_url"] == "http://10.0.0.1:8000/v1"
    assert backend.init_kwargs["request_timeout_sec"] == 600.0
    assert backend.init_kwargs["data_parallel_size"] == 1


def test_build_vllm_backend_uses_data_parallel_replicas(monkeypatch):
    ScriptedGeminiBackend.instances = []
    monkeypatch.setattr(
        inference_module,
        "VLLMGenerationBackend",
        ScriptedGeminiBackend,
    )
    args = parse_args(
        [
            "--generation-backend",
            "vllm",
            "--model",
            "/mnt/local-ssd/Qwen3.5-4B",
            "--dataset-csv",
            "dataset.csv",
            "--corpus-dir",
            "corpus",
            "--data-parallel-size",
            "8",
            "--tensor-parallel-size",
            "1",
            "--enable-expert-parallel",
            "--enable-prefix-caching",
        ]
    )

    backend = inference_module.build_generation_backend(args)

    assert backend.init_kwargs["data_parallel_size"] == 8
    assert backend.init_kwargs["tensor_parallel_size"] == 1
    assert backend.init_kwargs["enable_expert_parallel"] is True
    assert backend.init_kwargs["enable_prefix_caching"] is True


def test_build_vllm_backend_enables_thinking_for_gemma4_12b(monkeypatch):
    monkeypatch.setattr(
        inference_module,
        "VLLMGenerationBackend",
        ScriptedGeminiBackend,
    )
    args = parse_args(
        [
            "--generation-backend",
            "vllm",
            "--model",
            "/mnt/local-ssd/gemma-4-12B-it",
            "--dataset-csv",
            "dataset.csv",
            "--corpus-dir",
            "corpus",
        ]
    )

    backend = inference_module.build_generation_backend(args)

    assert backend.init_kwargs["enable_thinking"] is True


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_real_vllm_generation_smoke():
    model_path = os.environ.get("OFFICEQA_TEST_VLLM_MODEL")
    if not model_path:
        pytest.skip("Set OFFICEQA_TEST_VLLM_MODEL to a local checkpoint")

    backend = VLLMGenerationBackend(model_path, max_tokens=16)
    turn = await backend.generate(
        [{"role": "user", "content": "Reply with the single word OK."}],
        {"temperature": 0.0},
    )

    assert turn.text.strip()
