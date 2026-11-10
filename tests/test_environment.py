import asyncio
import json
import os
import re
from pathlib import Path

import pytest

from officeqa_rl.full_corpus_env import FullCorpusEnvironment
from officeqa_rl.types import EnvironmentConfig, OfficeQATask, Submitted


def make_config(tmp_path: Path, corpus_dir: Path, **overrides) -> EnvironmentConfig:
    values = {
        "corpus_dir": corpus_dir,
        "pdf_dir": None,
        "workspace_root": tmp_path / "workspaces",
        "sandbox_image": None,
        "command_timeout_sec": 5,
        "max_observation_chars": 200,
        "sandbox_runtime": "local",
    }
    values.update(overrides)
    return EnvironmentConfig(**values)


def test_environment_config_defaults_to_docker_runtime(tmp_path):
    config = EnvironmentConfig(
        corpus_dir=tmp_path / "corpus",
        pdf_dir=None,
        workspace_root=tmp_path / "workspaces",
        sandbox_image="officeqa-sandbox:test",
        command_timeout_sec=5,
        max_observation_chars=200,
    )

    assert config.sandbox_runtime == "docker"


def test_environment_config_preserves_existing_optional_positional_order(tmp_path):
    config = EnvironmentConfig(
        tmp_path / "corpus",
        None,
        tmp_path / "workspaces",
        None,
        5,
        200,
        "local",
        False,
        7,
    )

    assert config.enable_visual_pdf_tool is False
    assert config.max_concurrent_sandbox_boots == 7
    assert config.page_text_dir is None
    assert config.pdf_render_dpi == 144


def test_visual_pdf_tool_requires_pdf_directory(tmp_path):
    with pytest.raises(ValueError, match="pdf_dir is required"):
        FullCorpusEnvironment(
            make_config(
                tmp_path,
                tmp_path / "corpus",
                enable_visual_pdf_tool=True,
            ),
            OfficeQATask(uid="visual", question="Question?", answer=None),
            run_id="run",
        )


@pytest.mark.asyncio
async def test_text_search_observation_attaches_relevant_pdf_page(
    tmp_path,
    monkeypatch,
):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "bulletin.txt").write_text(
        "National defense expenditures were reported in this table.\n",
        encoding="utf-8",
    )
    page_text_dir = tmp_path / "pages"
    page_text_dir.mkdir()
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (page_text_dir / "bulletin.txt").write_text(
        "--- PAGE 2 ---\n\nCurrency in circulation by denomination.\n\n"
        "--- PAGE 3 ---\n\n"
        "Table: National defense expenditures by calendar year.\n",
        encoding="utf-8",
    )
    pdf_path = pdf_dir / "bulletin.pdf"
    pdf_path.write_bytes(b"placeholder")
    rendered_image = object()
    render_calls = []

    def fake_render(path, page_number, dpi):
        render_calls.append((path, page_number, dpi))
        return rendered_image

    monkeypatch.setattr(
        FullCorpusEnvironment,
        "_render_pdf_page",
        staticmethod(fake_render),
    )
    config = make_config(
        tmp_path,
        corpus_dir,
        pdf_dir=pdf_dir,
        page_text_dir=page_text_dir,
        enable_visual_pdf_tool=True,
        pdf_render_dpi=180,
        max_observation_chars=1_000,
    )
    task = OfficeQATask(
        uid="automatic-visual",
        question="What were the national defense expenditures?",
        answer=None,
        source_files="bulletin.txt",
    )

    async with FullCorpusEnvironment(config, task, run_id="run") as env:
        message = env.get_task_message()
        assert message == env.get_task_message()
        assert isinstance(message["content"], str)
        assert render_calls == []

        observation = await env.execute(
            'grep -ni "national defense" officeqa_corpus/bulletin.txt'
        )
        assert observation.image is rendered_image
        assert "National defense expenditures were reported" in observation.output
        assert "Automatically rendered PDF evidence" in observation.output
        assert "PDF page: 3" in observation.output
        assert "Table: National defense expenditures" in observation.output
        retrieval = env.get_info()["automatic_page_retrievals"]
        assert retrieval[0]["retrieval_reason"] == "lexical_match"
        assert retrieval[0]["search_query"] == "national defense"

    assert render_calls == [(pdf_path.resolve(), 3, 180)]


@pytest.mark.asyncio
async def test_automatic_page_retrieval_stays_inside_searched_documents(
    tmp_path,
    monkeypatch,
):
    """The attached page must come from a document the search matched.

    The task's own ``source_files`` holds the better-scoring page, so a
    retrieval that consulted it would tell the model where the answer lives.
    """
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "searched.txt").write_text(
        "National defense expenditures are discussed here.\n",
        encoding="utf-8",
    )
    (corpus_dir / "gold.txt").write_text(
        "National defense expenditures table for the calendar year.\n",
        encoding="utf-8",
    )
    page_text_dir = tmp_path / "pages"
    page_text_dir.mkdir()
    (page_text_dir / "searched.txt").write_text(
        "--- PAGE 4 ---\n\nNational defense expenditures are discussed here.\n",
        encoding="utf-8",
    )
    (page_text_dir / "gold.txt").write_text(
        "--- PAGE 7 ---\n\n"
        "Table: national defense expenditures by calendar year, 1940.\n",
        encoding="utf-8",
    )
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "searched.pdf").write_bytes(b"placeholder")
    (pdf_dir / "gold.pdf").write_bytes(b"placeholder")
    rendered_image = object()
    monkeypatch.setattr(
        FullCorpusEnvironment,
        "_render_pdf_page",
        staticmethod(lambda path, page_number, dpi: rendered_image),
    )
    config = make_config(
        tmp_path,
        corpus_dir,
        pdf_dir=pdf_dir,
        page_text_dir=page_text_dir,
        enable_visual_pdf_tool=True,
        max_observation_chars=1_000,
    )
    task = OfficeQATask(
        uid="scoped-visual",
        question="What were national defense expenditures in calendar year 1940?",
        answer=None,
        source_files="gold.txt",
    )

    async with FullCorpusEnvironment(config, task, run_id="run") as env:
        observation = await env.execute(
            'grep -ni "national defense" officeqa_corpus/searched.txt'
        )
        assert observation.image is rendered_image
        assert "Document: searched" in observation.output
        assert "PDF page: 4" in observation.output
        retrieval = env.get_info()["automatic_page_retrievals"][0]
        assert retrieval["document"] == "searched"
        assert retrieval["candidate_documents"] == ["searched"]


@pytest.mark.asyncio
async def test_automatic_page_retrieval_declines_when_no_document_matched(
    tmp_path,
    monkeypatch,
):
    """A search that names no document attaches nothing."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "gold.txt").write_text(
        "National defense expenditures table.\n",
        encoding="utf-8",
    )
    page_text_dir = tmp_path / "pages"
    page_text_dir.mkdir()
    (page_text_dir / "gold.txt").write_text(
        "--- PAGE 7 ---\n\nNational defense expenditures table.\n",
        encoding="utf-8",
    )
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "gold.pdf").write_bytes(b"placeholder")
    monkeypatch.setattr(
        FullCorpusEnvironment,
        "_render_pdf_page",
        staticmethod(lambda path, page_number, dpi: object()),
    )
    config = make_config(
        tmp_path,
        corpus_dir,
        pdf_dir=pdf_dir,
        page_text_dir=page_text_dir,
        enable_visual_pdf_tool=True,
        max_observation_chars=1_000,
    )
    task = OfficeQATask(
        uid="unscoped-visual",
        question="What were national defense expenditures?",
        answer=None,
        source_files="gold.txt",
    )

    async with FullCorpusEnvironment(config, task, run_id="run") as env:
        observation = await env.execute(
            'grep -rhi "national defense" officeqa_corpus/'
        )
        assert observation.image is None
        assert "Automatically rendered PDF evidence" not in observation.output
        assert env.get_info()["automatic_page_retrievals"] == []


def test_retrieval_documents_read_search_output_then_command_paths(tmp_path):
    page_text_dir = tmp_path / "pages"
    page_text_dir.mkdir()
    for name in ("bulletin_1940_01", "bulletin_1940_02", "bulletin_1955_01"):
        (page_text_dir / f"{name}.txt").write_text(
            "--- PAGE 1 ---\n\ntext\n",
            encoding="utf-8",
        )
    config = make_config(
        tmp_path,
        tmp_path / "corpus",
        pdf_dir=tmp_path / "pdfs",
        page_text_dir=page_text_dir,
        enable_visual_pdf_tool=True,
    )
    env = FullCorpusEnvironment(
        config,
        OfficeQATask(uid="u", question="q", answer=None, source_files="secret.txt"),
        run_id="run",
    )

    matched = env._retrieval_documents(
        'grep -Rni "defense" officeqa_corpus/',
        "officeqa_corpus/bulletin_1940_02.txt:3:defense\n"
        "officeqa_corpus/bulletin_1940_01.txt:9:defense\n",
    )
    assert matched == ["bulletin_1940_02", "bulletin_1940_01"]

    # grep prints no file name for a single-file search, so fall back to the
    # paths and globs the command itself named.
    from_command = env._retrieval_documents(
        'grep -ni "defense" officeqa_corpus/bulletin_1940_*.txt',
        "3:defense\n",
    )
    assert from_command == ["bulletin_1940_01", "bulletin_1940_02"]
    assert env._retrieval_documents('grep -rhi "defense" officeqa_corpus/', "") == []


def test_explicit_printed_page_and_contents_reference_are_resolved():
    candidates = [
        ("bulletin", 8, "Contents\n| TF-G chart highway trust fund | 148.0 |"),
        ("bulletin", 19, "5 ECONOMIC POLICY\nA line chart."),
        ("bulletin", 158, "148 TRUST FUND REPORTS\nHighway trust fund chart."),
    ]
    scores = FullCorpusEnvironment._rank_pages(
        "On page 5, how many local maxima are on the line chart?",
        [candidate[2] for candidate in candidates],
    )
    assert FullCorpusEnvironment._select_explicit_page(
        "On page 5, how many local maxima are on the line chart?",
        candidates,
        scores,
    ) == 1
    assert FullCorpusEnvironment._resolve_contents_reference(
        "According to chart TF-G, when did highway trust fund outlays rise?",
        candidates,
        0,
    ) == 2
    assert FullCorpusEnvironment._search_query_from_command(
        'grep -Rni -A 3 -B 3 "national defense" officeqa_corpus/bulletin.txt '
        "| head -50"
    ) == "national defense"
    assert FullCorpusEnvironment._search_query_from_command(
        'rg -ni "national defense" unrelated_directory/'
    ) is None


@pytest.mark.asyncio
async def test_view_page_returns_marked_text_and_rendered_image(
    tmp_path,
    monkeypatch,
):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    page_text_dir = tmp_path / "pages"
    page_text_dir.mkdir()
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (page_text_dir / "treasury_bulletin_1954_01.txt").write_text(
        "--- PAGE 12 ---\n\nprevious page\n\n"
        "--- PAGE 13 ---\n\nThe requested table value is 391.\n\n"
        "--- PAGE 14 ---\n\nnext page\n",
        encoding="utf-8",
    )
    pdf_path = pdf_dir / "treasury_bulletin_1954_01.pdf"
    pdf_path.write_bytes(b"placeholder")
    rendered_image = object()
    render_calls = []

    def fake_render(path, page_number, dpi):
        render_calls.append((path, page_number, dpi))
        return rendered_image

    monkeypatch.setattr(
        FullCorpusEnvironment,
        "_render_pdf_page",
        staticmethod(fake_render),
    )
    config = make_config(
        tmp_path,
        corpus_dir,
        pdf_dir=pdf_dir,
        page_text_dir=page_text_dir,
        enable_visual_pdf_tool=True,
        pdf_render_dpi=180,
    )
    task = OfficeQATask(uid="visual", question="Question?", answer=None)

    async with FullCorpusEnvironment(config, task, run_id="run") as env:
        observation = await env.execute("view_page treasury_bulletin_1954_01 13")
        assert (env.workspace / "officeqa_pages").is_symlink()
        assert "The requested table value is 391." in observation.output
        assert "previous page" not in observation.output
        assert "next page" not in observation.output
        assert observation.image is rendered_image
        assert "view_page" in env.get_task_message()["content"]

    assert render_calls == [(pdf_path.resolve(), 13, 180)]


@pytest.mark.asyncio
async def test_view_page_rejects_page_not_in_page_text(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    page_text_dir = tmp_path / "pages"
    page_text_dir.mkdir()
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (page_text_dir / "bulletin.txt").write_text(
        "--- PAGE 2 ---\n\nonly page\n",
        encoding="utf-8",
    )
    (pdf_dir / "bulletin.pdf").write_bytes(b"placeholder")
    config = make_config(
        tmp_path,
        corpus_dir,
        pdf_dir=pdf_dir,
        page_text_dir=page_text_dir,
        enable_visual_pdf_tool=True,
    )

    async with FullCorpusEnvironment(
        config,
        OfficeQATask(uid="visual", question="Question?", answer=None),
        run_id="run",
    ) as env:
        observation = await env.execute("view_page bulletin 99")

    assert "Page 99 is not present" in observation.output
    assert observation.image is None


@pytest.mark.asyncio
async def test_search_then_submit_without_leaking_answer(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "example.txt").write_text(
        "The reported revenue was 17 million dollars.\n",
        encoding="utf-8",
    )
    task = OfficeQATask(
        uid="question/1",
        question="What revenue was reported?",
        answer="17 million dollars",
        source_files="example.txt",
    )
    env = FullCorpusEnvironment(
        make_config(tmp_path, corpus_dir),
        task,
        run_id="rollout-3/sample-7",
    )

    async with env:
        assert env.is_active
        observation = await env.execute(
            {
                "type": "command",
                "command": 'grep -n "revenue" officeqa_corpus/example.txt',
            }
        )
        assert "17 million dollars" in observation.output
        assert observation.output.startswith("Exit code: 0")
        assert observation.image is None
        assert "automatic_page_retrievals" not in env.get_info()

        task_message = env.get_task_message()
        assert isinstance(task_message["content"], str)
        assert "officeqa_pages" not in task_message["content"]
        assert task.answer not in json.dumps(task_message)
        manifest = json.loads(
            (env.workspace / "manifest.json").read_text(encoding="utf-8")
        )
        assert set(manifest) == {
            "uid",
            "question",
            "source_files",
            "run_id",
            "workspace",
            "corpus_dir",
            "workspace_corpus_path",
            "sandbox_runtime",
            "pdf_dir",
            "visual_pdf_tool_enabled",
        }
        assert manifest["visual_pdf_tool_enabled"] is False
        assert set(env._command_env(docker=False)) == {
            "OFFICEQA_WORKSPACE",
            "OFFICEQA_CORPUS_DIR",
            "OFFICEQA_PDF_DIR",
            "OFFICEQA_UID",
            "OFFICEQA_RUN_ID",
        }
        assert set(env.get_info()) == {
            "uid",
            "run_id",
            "workspace",
            "corpus_dir",
            "pdf_dir",
            "source_files",
            "sandbox_runtime",
            "sandbox_image",
            "command_container_id",
            "command_history",
        }
        disabled_visual = await env.execute("view_page example 1")
        assert disabled_visual.output == (
            "Unknown command. This environment is text-only; use shell commands "
            'or `submit "<answer>"`.'
        )
        assert disabled_visual.image is None

        with pytest.raises(Submitted) as submitted:
            await env.execute('submit "17 million dollars"')
        assert submitted.value.submission == "17 million dollars"

        agent_files = [
            env.workspace / "question.txt",
            env.workspace / "source_files.txt",
            env.workspace / "manifest.json",
        ]
        assert all(
            task.answer not in path.read_text(encoding="utf-8")
            for path in agent_files
        )
        visible_workspace_text = "".join(
            path.read_text(encoding="utf-8")
            for path in env.workspace.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        assert task.answer not in visible_workspace_text
        assert "answer" not in json.loads(
            (env.workspace / "manifest.json").read_text(encoding="utf-8")
        )

    assert not env.is_active
    assert env.command_container_id is None


@pytest.mark.asyncio
async def test_docker_runtime_mounts_corpus_read_only_and_cleans_up(
    tmp_path,
    monkeypatch,
):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    docker_log = tmp_path / "docker.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        """#!/bin/bash
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
case "$1" in
  run) printf 'fake-container-id\\n' ;;
  exec) printf 'docker-exec-ok\\n' ;;
  rm) exit 0 ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(docker_log))
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    task = OfficeQATask(
        uid="docker-task",
        question="What value was reported?",
        answer="private-ground-truth",
    )
    config = make_config(
        tmp_path,
        corpus_dir,
        sandbox_runtime="docker",
        sandbox_image="officeqa-sandbox:test",
    )
    env = FullCorpusEnvironment(config, task, run_id="docker-rollout-1")

    async with env:
        assert env.command_container_id == "fake-container-id"
        assert not (env.workspace / "officeqa_corpus").exists()
        observation = await env.execute("printf ignored-by-fake-docker")
        assert "docker-exec-ok" in observation.output

    invocations = docker_log.read_text(encoding="utf-8")
    corpus_mount = f"{corpus_dir.resolve()}:/workspace/officeqa_corpus:ro"
    assert "run -d" in invocations
    assert "--read-only" in invocations
    assert corpus_mount in invocations
    assert "officeqa-sandbox:test" in invocations
    assert "exec -w /workspace" in invocations
    assert "rm -f fake-container-id" in invocations
    assert task.answer not in invocations


@pytest.mark.asyncio
async def test_run_id_isolates_workspaces_for_the_same_task(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    task = OfficeQATask(uid="same-uid", question="Question?", answer=None)
    config = make_config(tmp_path, corpus_dir)
    first = FullCorpusEnvironment(config, task, run_id="group-1-sample-1")
    second = FullCorpusEnvironment(config, task, run_id="group-1-sample-2")

    assert first.workspace != second.workspace
    async with first, second:
        assert first.workspace.is_dir()
        assert second.workspace.is_dir()


@pytest.mark.asyncio
async def test_output_truncation_keeps_head_and_tail(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    config = make_config(tmp_path, corpus_dir, max_observation_chars=20)
    task = OfficeQATask(uid="truncate", question="Question?", answer=None)

    async with FullCorpusEnvironment(config, task, run_id="run") as env:
        observation = await env.execute(
            "python3 -c 'print(\"abcdefghij0123456789ABCDEFGHIJ\", end=\"\")'"
        )

    assert "abcdefghij" in observation.output
    assert "ABCDEFGHIJ" in observation.output
    assert "omitted 10 characters" in observation.output


@pytest.mark.asyncio
async def test_context_manager_cleans_up_after_exception(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    task = OfficeQATask(uid="cleanup", question="Question?", answer=None)
    env = FullCorpusEnvironment(
        make_config(tmp_path, corpus_dir),
        task,
        run_id="exception-run",
    )

    with pytest.raises(RuntimeError, match="boom"):
        async with env:
            raise RuntimeError("boom")

    assert not env.is_active
    assert env.command_container_id is None


async def wait_for_process_exit(pid: int) -> None:
    for _ in range(200):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"process {pid} is still running")


async def read_command_pid(pid_file: Path) -> int:
    for _ in range(500):
        if pid_file.exists():
            text = pid_file.read_text(encoding="utf-8").strip()
            if text:
                return int(text)
        await asyncio.sleep(0.01)
    pytest.fail(f"command never wrote {pid_file}")


@pytest.mark.asyncio
async def test_concurrent_docker_boots_are_limited_by_semaphore(
    tmp_path,
    monkeypatch,
):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    docker_log = tmp_path / "docker.log"
    state_dir = tmp_path / "docker-state"
    state_dir.mkdir()
    (state_dir / "lock").touch()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        """#!/bin/bash
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
case "$1" in
  run)
    flock "$FAKE_DOCKER_STATE/lock" bash -c '
      active=$(( $(cat "$FAKE_DOCKER_STATE/active" 2>/dev/null || echo 0) + 1 ))
      echo "$active" > "$FAKE_DOCKER_STATE/active"
      peak=$(cat "$FAKE_DOCKER_STATE/peak" 2>/dev/null || echo 0)
      if [ "$active" -gt "$peak" ]; then echo "$active" > "$FAKE_DOCKER_STATE/peak"; fi
    '
    sleep 0.2
    flock "$FAKE_DOCKER_STATE/lock" bash -c '
      active=$(( $(cat "$FAKE_DOCKER_STATE/active") - 1 ))
      echo "$active" > "$FAKE_DOCKER_STATE/active"
    '
    printf 'container-%s\\n' "$$"
    ;;
  rm) exit 0 ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(docker_log))
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(state_dir))
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    task = OfficeQATask(uid="same-uid", question="Question?", answer=None)
    config = make_config(
        tmp_path,
        corpus_dir,
        sandbox_runtime="docker",
        sandbox_image="officeqa-sandbox:test",
        max_concurrent_sandbox_boots=1,
    )

    async def boot(index: int) -> tuple[str, str]:
        env = FullCorpusEnvironment(config, task, run_id=f"boot-{index}")
        async with env:
            return env._container_name, env.command_container_id

    results = await asyncio.wait_for(
        asyncio.gather(*(boot(index) for index in range(4))),
        timeout=30,
    )

    peak = int((state_dir / "peak").read_text(encoding="utf-8"))
    assert peak == 1, f"semaphore allowed {peak} concurrent boots"

    container_names = [name for name, _ in results]
    container_ids = [container_id for _, container_id in results]
    assert len(set(container_names)) == 4
    assert len(set(container_ids)) == 4

    invocations = docker_log.read_text(encoding="utf-8")
    assert invocations.count("run -d") == 4
    removed = re.findall(r"rm -f (\S+)", invocations)
    assert sorted(removed) == sorted(container_ids)


@pytest.mark.asyncio
async def test_concurrent_environments_do_not_block_the_event_loop(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    config = make_config(tmp_path, corpus_dir, command_timeout_sec=10)
    task = OfficeQATask(uid="loop-check", question="Question?", answer=None)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    async with FullCorpusEnvironment(config, task, run_id="loop-run") as env:
        ticker_task = asyncio.create_task(ticker())
        try:
            observation = await env.execute("sleep 0.5")
        finally:
            ticker_task.cancel()

    assert observation.output.startswith("Exit code: 0")
    assert ticks >= 10, "event loop was starved while the command ran"


@pytest.mark.asyncio
async def test_cancelled_command_is_killed_and_cleanup_runs(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    config = make_config(tmp_path, corpus_dir, command_timeout_sec=30)
    task = OfficeQATask(uid="same-uid", question="Question?", answer=None)
    env = FullCorpusEnvironment(config, task, run_id="cancel-run")

    async def run_forever() -> None:
        async with env:
            await env.execute("echo $$ > cmd.pid; sleep 30")

    runner = asyncio.create_task(run_forever())
    pid = await read_command_pid(env.workspace / "cmd.pid")

    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner

    assert not env.is_active
    await wait_for_process_exit(pid)


@pytest.mark.asyncio
async def test_timed_out_command_cleanup_kills_process_group(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    config = make_config(tmp_path, corpus_dir, command_timeout_sec=1)
    task = OfficeQATask(uid="timeout", question="Question?", answer=None)

    async with FullCorpusEnvironment(config, task, run_id="timeout-run") as env:
        observation = await env.execute("echo $$ > cmd.pid; sleep 30; echo done")
        pid = await read_command_pid(env.workspace / "cmd.pid")

    assert "timed out after 1s" in observation.output
    assert "done" not in observation.output
    await wait_for_process_exit(pid)
