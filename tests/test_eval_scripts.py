import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script_name",
    [
        "eval_qwen35_4b.sh",
        "eval_gemini35_flash.sh",
        "eval_vllm_full.sh",
        "eval_kimi_k3.sh",
        "eval_openai_server.sh",
        "generate_sft_traces.sh",
    ],
)
def test_eval_script_has_valid_bash_syntax(script_name):
    subprocess.run(
        ["bash", "-n", str(ROOT / "scripts" / script_name)],
        check=True,
    )


@pytest.mark.parametrize(
    ("script_name", "backend", "model_flag"),
    [
        ("eval_qwen35_4b.sh", "vllm", "/models/Qwen3.5-4B"),
        ("eval_gemini35_flash.sh", "gemini", "gemini-3.5-flash"),
        ("eval_vllm_full.sh", "vllm", "/models/Qwen3.5-4B"),
        ("eval_kimi_k3.sh", "kimi", "kimi-k3"),
        ("eval_openai_server.sh", "openai-server", "/models/GLM-5.2"),
    ],
)
def test_eval_script_selects_model_and_docker_runtime(
    tmp_path,
    script_name,
    backend,
    model_flag,
):
    args_log = tmp_path / "args.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/bash\nprintf '%s\\n' \"$*\" > \"$EVAL_ARGS_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "EVAL_ARGS_LOG": str(args_log),
        "PYTHON_BIN": str(fake_python),
        "OFFICEQA_DATASET_CSV": "/data/officeqa_pro.csv",
        "OFFICEQA_UID": "test-uid",
        "OFFICEQA_CORPUS_DIR": "/data/corpus",
        "OFFICEQA_SANDBOX_IMAGE": "officeqa-sandbox:test",
        "OFFICEQA_RUN_ID": "test-run",
        "QWEN35_4B_MODEL": "/models/Qwen3.5-4B",
        "VLLM_MODEL": "/models/Qwen3.5-4B",
        "GEMINI_MODEL": "gemini-3.5-flash",
        "GOOGLE_API_KEY": "fake-key",
        "KIMI_MODEL": "kimi-k3",
        "MOONSHOT_API_KEY": "fake-key",
        "OPENAI_MODEL": "/models/GLM-5.2",
        "OPENAI_BASE_URL": "http://10.0.0.1:8000/v1",
    }
    env.pop("OFFICEQA_ENABLE_VISUAL_PDF_TOOL", None)
    env.pop("OFFICEQA_PAGE_TEXT_DIR", None)
    env.pop("OFFICEQA_PDF_DIR", None)

    subprocess.run(
        ["bash", str(ROOT / "scripts" / script_name)],
        check=True,
        env=env,
    )

    args = args_log.read_text(encoding="utf-8")
    assert f"--generation-backend {backend}" in args
    assert f"--model {model_flag}" in args
    assert "--sandbox-runtime docker" in args
    assert "--sandbox-image officeqa-sandbox:test" in args
    assert "--uid test-uid" in args
    assert "--enable-visual-pdf-tool" not in args


def test_vllm_full_script_runs_full_dataset_with_batched_generation(tmp_path):
    args_log = tmp_path / "args.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/bash\nprintf '%s\\n' \"$*\" > \"$EVAL_ARGS_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    results_dir = tmp_path / "results"
    env = {
        **{
            key: value
            for key, value in os.environ.items()
            if key not in {"OFFICEQA_UID", "OFFICEQA_ENABLE_VISUAL_PDF_TOOL"}
        },
        "EVAL_ARGS_LOG": str(args_log),
        "PYTHON_BIN": str(fake_python),
        "VLLM_MODEL": "/models/Qwen3.5-4B",
        "VLLM_DATA_PARALLEL_SIZE": "4",
        "VLLM_ENABLE_EXPERT_PARALLEL": "1",
        "OFFICEQA_OUTPUT_DIR": str(results_dir),
        "OFFICEQA_CONCURRENCY": "3",
    }

    subprocess.run(
        ["bash", str(ROOT / "scripts" / "eval_vllm_full.sh")],
        check=True,
        env=env,
    )

    args = args_log.read_text(encoding="utf-8")
    assert "--generation-backend vllm" in args
    assert "--uid" not in args
    assert f"--output-dir {results_dir}" in args
    assert "--concurrency 3" in args
    assert "--data-parallel-size 4" in args
    assert "--enable-expert-parallel" in args
    assert "--serialize-generation-calls" not in args
    log_files = list(results_dir.glob("eval-*.log"))
    assert len(log_files) == 1


# The standard 8-GPU recipe uses one flat concurrency default for every model
# (commit 0e60334); larger models are expected to adjust TP/DP and set
# OFFICEQA_CONCURRENCY explicitly instead of relying on name sniffing.
@pytest.mark.parametrize(
    ("model", "expected_concurrency"),
    [
        ("/models/Qwen3.5-4B", 48),
        ("/models/Qwen3.6-35B-A3B", 48),
        ("/models/gemma-4-26B-A4B-it", 48),
    ],
)
def test_vllm_full_script_uses_standard_concurrency_default(
    tmp_path,
    model,
    expected_concurrency,
):
    args_log = tmp_path / "args.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/bash\nprintf '%s\\n' \"$*\" > \"$EVAL_ARGS_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = {
        **{
            key: value
            for key, value in os.environ.items()
            if key not in {"OFFICEQA_UID", "OFFICEQA_CONCURRENCY"}
        },
        "EVAL_ARGS_LOG": str(args_log),
        "PYTHON_BIN": str(fake_python),
        "VLLM_MODEL": model,
        "OFFICEQA_OUTPUT_DIR": str(tmp_path / "results"),
    }

    subprocess.run(
        ["bash", str(ROOT / "scripts" / "eval_vllm_full.sh")],
        check=True,
        env=env,
    )

    args = args_log.read_text(encoding="utf-8")
    assert f"--concurrency {expected_concurrency}" in args


def test_qwen_script_enables_visual_pdf_page_inputs(tmp_path):
    args_log = tmp_path / "args.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/bash\nprintf '%s\\n' \"$*\" > \"$EVAL_ARGS_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "EVAL_ARGS_LOG": str(args_log),
        "PYTHON_BIN": str(fake_python),
        "OFFICEQA_DATASET_CSV": "/data/officeqa_pro.csv",
        "OFFICEQA_UID": "test-uid",
        "OFFICEQA_CORPUS_DIR": "/data/corpus",
        "OFFICEQA_SANDBOX_IMAGE": "officeqa-sandbox:test",
        "OFFICEQA_ENABLE_VISUAL_PDF_TOOL": "1",
        "OFFICEQA_PAGE_TEXT_DIR": "/data/pages",
        "OFFICEQA_PDF_DIR": "/data/pdfs",
        "OFFICEQA_PDF_RENDER_DPI": "180",
    }

    subprocess.run(
        ["bash", str(ROOT / "scripts" / "eval_qwen35_4b.sh")],
        check=True,
        env=env,
    )

    args = args_log.read_text(encoding="utf-8")
    assert "--enable-visual-pdf-tool" in args
    assert "--page-text-dir /data/pages" in args
    assert "--pdf-dir /data/pdfs" in args
    assert "--pdf-render-dpi 180" in args


def test_kimi_script_requires_api_key(tmp_path):
    env = {key: value for key, value in os.environ.items() if key != "MOONSHOT_API_KEY"}
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "eval_kimi_k3.sh")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "MOONSHOT_API_KEY" in result.stderr


def test_kimi_script_runs_full_dataset_when_uid_is_unset(tmp_path):
    args_log = tmp_path / "args.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/bash\nprintf '%s\\n' \"$*\" > \"$EVAL_ARGS_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    results_dir = tmp_path / "results"
    env = {
        **{key: value for key, value in os.environ.items() if key != "OFFICEQA_UID"},
        "EVAL_ARGS_LOG": str(args_log),
        "PYTHON_BIN": str(fake_python),
        "OFFICEQA_DATASET_CSV": "/data/officeqa_pro.csv",
        "OFFICEQA_CORPUS_DIR": "/data/corpus",
        "OFFICEQA_SANDBOX_IMAGE": "officeqa-sandbox:test",
        "OFFICEQA_OUTPUT_DIR": str(results_dir),
        "OFFICEQA_CONCURRENCY": "3",
        "MOONSHOT_API_KEY": "fake-key",
        "KIMI_REASONING_EFFORT": "high",
    }

    subprocess.run(
        ["bash", str(ROOT / "scripts" / "eval_kimi_k3.sh")],
        check=True,
        env=env,
    )

    args = args_log.read_text(encoding="utf-8")
    assert "--generation-backend kimi" in args
    assert "--model kimi-k3" in args
    assert "--kimi-reasoning-effort high" in args
    assert "--uid" not in args
    assert f"--output-dir {results_dir}" in args
    assert "--concurrency 3" in args
    log_files = list(results_dir.glob("eval-*.log"))
    assert len(log_files) == 1


def test_openai_server_script_runs_full_dataset_with_explicit_model(tmp_path):
    args_log = tmp_path / "args.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/bash\nprintf '%s\\n' \"$*\" > \"$EVAL_ARGS_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    results_dir = tmp_path / "results"
    env = {
        **{key: value for key, value in os.environ.items() if key != "OFFICEQA_UID"},
        "EVAL_ARGS_LOG": str(args_log),
        "PYTHON_BIN": str(fake_python),
        "OFFICEQA_OUTPUT_DIR": str(results_dir),
        "OFFICEQA_CONCURRENCY": "3",
        # An explicit model skips /models discovery, which needs a live server.
        "OPENAI_MODEL": "/mnt/local-ssd/GLM-5.2",
        "OPENAI_BASE_URL": "http://10.0.0.1:8000/v1",
    }

    subprocess.run(
        ["bash", str(ROOT / "scripts" / "eval_openai_server.sh")],
        check=True,
        env=env,
    )

    args = args_log.read_text(encoding="utf-8")
    assert "--generation-backend openai-server" in args
    assert "--model /mnt/local-ssd/GLM-5.2" in args
    assert "--openai-base-url http://10.0.0.1:8000/v1" in args
    assert "--uid" not in args
    assert f"--output-dir {results_dir}" in args
    assert "--concurrency 3" in args
    log_files = list(results_dir.glob("eval-*.log"))
    assert len(log_files) == 1


def test_sft_trace_script_defaults_to_synthetic_dataset(tmp_path):
    args_log = tmp_path / "args.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/bash\nprintf '%s\\n' \"$*\" > \"$EVAL_ARGS_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    results_dir = tmp_path / "traces"
    env = {
        **{key: value for key, value in os.environ.items() if key != "OFFICEQA_UID"},
        "EVAL_ARGS_LOG": str(args_log),
        "PYTHON_BIN": str(fake_python),
        "OFFICEQA_OUTPUT_DIR": str(results_dir),
        # An explicit model skips /models discovery, which needs a live server.
        "OPENAI_MODEL": "/mnt/local-ssd/GLM-5.2",
        "OPENAI_BASE_URL": "http://10.0.0.1:8000/v1",
    }

    subprocess.run(
        ["bash", str(ROOT / "scripts" / "generate_sft_traces.sh")],
        check=True,
        env=env,
    )

    args = args_log.read_text(encoding="utf-8")
    assert "--generation-backend openai-server" in args
    assert "--dataset-csv" in args
    assert "data/synthetic/officeqa_synth_author_v1.csv" in args
    assert "--openai-base-url http://10.0.0.1:8000/v1" in args
    assert f"--output-dir {results_dir}" in args
    assert "--uid" not in args
    log_files = list(results_dir.glob("traces-*.log"))
    assert len(log_files) == 1


def test_gemini_script_runs_full_dataset_when_uid_is_unset(tmp_path):
    args_log = tmp_path / "args.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/bash\nprintf '%s\\n' \"$*\" > \"$EVAL_ARGS_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    results_dir = tmp_path / "results"
    env = {
        **{key: value for key, value in os.environ.items() if key != "OFFICEQA_UID"},
        "EVAL_ARGS_LOG": str(args_log),
        "PYTHON_BIN": str(fake_python),
        "OFFICEQA_DATASET_CSV": "/data/officeqa_pro.csv",
        "OFFICEQA_CORPUS_DIR": "/data/corpus",
        "OFFICEQA_SANDBOX_IMAGE": "officeqa-sandbox:test",
        "OFFICEQA_OUTPUT_DIR": str(results_dir),
        "OFFICEQA_CONCURRENCY": "3",
        "GOOGLE_API_KEY": "fake-key",
    }

    subprocess.run(
        ["bash", str(ROOT / "scripts" / "eval_gemini35_flash.sh")],
        check=True,
        env=env,
    )

    args = args_log.read_text(encoding="utf-8")
    assert "--uid" not in args
    assert f"--output-dir {results_dir}" in args
    assert "--concurrency 3" in args
    log_files = list(results_dir.glob("eval-*.log"))
    assert len(log_files) == 1
