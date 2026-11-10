import asyncio
import copy
import os
from pathlib import Path

import pytest

from officeqa_rl.agent import OfficeQAAgent
from officeqa_rl.full_corpus_env import FullCorpusEnvironment
from officeqa_rl.prompts import build_system_prompt
from officeqa_rl.types import (
    AgentConfig,
    EnvironmentConfig,
    GeneratedTurn,
    Message,
    OfficeQATask,
)


class FakeGenerationBackend:
    def __init__(self, turns: list[GeneratedTurn]):
        self.turns = iter(turns)
        self.calls: list[tuple[list[Message], dict]] = []

    async def generate(self, messages, sampling_params):
        self.calls.append((copy.deepcopy(messages), dict(sampling_params)))
        return next(self.turns)


def install_fake_docker(tmp_path: Path, monkeypatch) -> Path:
    docker_log = tmp_path / "docker.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        """#!/bin/bash
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
case "$1" in
  run) printf 'fake-agent-container\\n' ;;
  exec) printf '7:The reported revenue was 123.45 million.\\n' ;;
  rm) exit 0 ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(docker_log))
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    return docker_log


@pytest.mark.asyncio
async def test_agent_searches_then_submits_through_docker_runtime(
    tmp_path,
    monkeypatch,
):
    docker_log = install_fake_docker(tmp_path, monkeypatch)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    task = OfficeQATask(
        uid="agent-task",
        question="What revenue was reported?",
        answer="123.45 million",
    )
    env_config = EnvironmentConfig(
        corpus_dir=corpus_dir,
        pdf_dir=None,
        workspace_root=tmp_path / "workspaces",
        sandbox_image="officeqa-sandbox:test",
        command_timeout_sec=5,
        max_observation_chars=1_000,
    )
    agent_config = AgentConfig(step_limit=4)
    search_turn = GeneratedTurn(
        text='```bash\ngrep -n "revenue" officeqa_corpus/example.txt\n```',
        reasoning_content="I should search the corpus for the reported revenue.",
        prompt_ids=[1, 2],
        output_ids=[3, 4],
        output_log_probs=[-0.1, -0.2],
        finish_reason="stop",
    )
    submit_turn = GeneratedTurn(
        text='```bash\nsubmit "123.45 million"\n```',
        reasoning_content="The observation contains the requested value.",
        prompt_ids=[1, 2, 3, 4],
        output_ids=[5, 6],
        output_log_probs=[-0.3, -0.4],
        finish_reason="stop",
    )
    backend = FakeGenerationBackend([search_turn, submit_turn])
    agent = OfficeQAAgent(
        backend,
        agent_config,
        sampling_params={"temperature": 0.0},
    )

    async with FullCorpusEnvironment(
        env_config,
        task,
        run_id="rollout-1-sample-2",
    ) as env:
        result = await agent.run(env)

    assert result.submission == "123.45 million"
    assert result.status == "submitted"
    assert result.steps == 2
    assert [message["role"] for message in result.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "exit",
    ]
    assert result.messages[0]["content"] == build_system_prompt(agent_config)
    assert result.messages[2]["content"] == search_turn.text
    assert result.messages[2]["reasoning_content"] == search_turn.reasoning_content
    assert "The reported revenue was 123.45 million" in result.messages[3]["content"]
    assert result.messages[4]["content"] == submit_turn.text
    assert result.messages[5]["content"] == "Submitted answer: 123.45 million"

    assert [message["role"] for message in backend.calls[0][0]] == [
        "system",
        "user",
    ]
    assert [message["role"] for message in backend.calls[1][0]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert (
        backend.calls[1][0][2]["reasoning_content"]
        == "I should search the corpus for the reported revenue."
    )
    assert backend.calls[0][1] == {"temperature": 0.0}
    assert result.metadata["generated_turns"][0]["output_ids"] == [3, 4]
    assert (
        result.metadata["generated_turns"][0]["reasoning_content"]
        == search_turn.reasoning_content
    )
    assert "rm -f fake-agent-container" in docker_log.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_context_exhausted_length_turn_ends_episode_as_limit(tmp_path):
    # An empty "length" turn means the context window ran out before the turn
    # could generate anything. The episode must end as limits_exceeded (its
    # completed turns stay valid training samples) instead of looping on
    # format errors or aborting.
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    env_config = EnvironmentConfig(
        corpus_dir=corpus_dir,
        pdf_dir=None,
        workspace_root=tmp_path / "workspaces",
        sandbox_image=None,
        command_timeout_sec=5,
        max_observation_chars=1_000,
        sandbox_runtime="local",
    )
    task = OfficeQATask(uid="ctxwall", question="Question?", answer=None)
    backend = FakeGenerationBackend(
        [
            GeneratedTurn(text="```bash\nls\n```"),
            GeneratedTurn(
                text="", output_ids=[], output_log_probs=[], finish_reason="length"
            ),
            GeneratedTurn(text="```bash\nnever reached\n```"),
        ]
    )
    agent = OfficeQAAgent(backend, AgentConfig(step_limit=10))

    async with FullCorpusEnvironment(env_config, task, run_id="ctxwall-run") as env:
        result = await agent.run(env)

    assert result.status == "limits_exceeded"
    assert result.submission == "TIMEOUT"
    assert len(backend.calls) == 2  # the third canned turn is never requested
    exit_message = result.messages[-1]
    assert exit_message["role"] == "exit"
    assert "Context window exhausted after 1 steps" in exit_message["content"]
    # No empty assistant message is appended for the exhausted turn.
    assert result.messages[-2]["role"] != "assistant"


async def test_per_turn_length_cap_does_not_end_episode(tmp_path):
    # finish_reason="length" with actual output is the per-turn token cap,
    # not context exhaustion — the episode must continue.
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    env_config = EnvironmentConfig(
        corpus_dir=corpus_dir,
        pdf_dir=None,
        workspace_root=tmp_path / "workspaces",
        sandbox_image=None,
        command_timeout_sec=5,
        max_observation_chars=1_000,
        sandbox_runtime="local",
    )
    task = OfficeQATask(uid="turncap", question="Question?", answer=None)
    backend = FakeGenerationBackend(
        [
            GeneratedTurn(
                text="```bash\nls\n```",
                output_ids=[1, 2, 3],
                finish_reason="length",
            ),
            GeneratedTurn(text='```bash\nsubmit "42"\n```'),
        ]
    )
    agent = OfficeQAAgent(backend, AgentConfig(step_limit=10))

    async with FullCorpusEnvironment(env_config, task, run_id="turncap-run") as env:
        result = await agent.run(env)

    assert result.status == "submitted"
    assert result.submission == "42"


async def test_format_error_is_observed_before_step_limit(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    env_config = EnvironmentConfig(
        corpus_dir=corpus_dir,
        pdf_dir=None,
        workspace_root=tmp_path / "workspaces",
        sandbox_image=None,
        command_timeout_sec=5,
        max_observation_chars=1_000,
        sandbox_runtime="local",
    )
    task = OfficeQATask(uid="format", question="Question?", answer=None)
    backend = FakeGenerationBackend([GeneratedTurn(text="submit without a fence")])
    agent = OfficeQAAgent(backend, AgentConfig(step_limit=1))

    async with FullCorpusEnvironment(env_config, task, run_id="format-run") as env:
        result = await agent.run(env)

    assert result.submission == "TIMEOUT"
    assert result.status == "limits_exceeded"
    assert result.steps == 1
    assert [message["role"] for message in result.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "exit",
    ]
    assert result.messages[3]["extra"]["error"]["type"] == "FormatError"
    assert result.messages[-1]["content"] == "Step limit reached: 1"


@pytest.mark.asyncio
async def test_agent_returns_pdf_image_and_text_together_to_backend(
    tmp_path,
    monkeypatch,
):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pages_dir / "bulletin.txt").write_text(
        "--- PAGE 3 ---\n\nParsed value: 42\n",
        encoding="utf-8",
    )
    (pdf_dir / "bulletin.pdf").write_bytes(b"placeholder")

    class FakeImage:
        size = (100, 200)
        mode = "RGB"

    rendered_image = FakeImage()
    monkeypatch.setattr(
        FullCorpusEnvironment,
        "_render_pdf_page",
        staticmethod(lambda path, page, dpi: rendered_image),
    )
    backend = FakeGenerationBackend(
        [
            GeneratedTurn(text="```bash\nview_page bulletin 3\n```"),
            GeneratedTurn(text='```bash\nsubmit "42"\n```'),
        ]
    )
    agent = OfficeQAAgent(backend, AgentConfig(step_limit=3))
    config = EnvironmentConfig(
        corpus_dir=corpus_dir,
        pdf_dir=pdf_dir,
        page_text_dir=pages_dir,
        workspace_root=tmp_path / "workspaces",
        sandbox_image=None,
        command_timeout_sec=5,
        max_observation_chars=1_000,
        sandbox_runtime="local",
        enable_visual_pdf_tool=True,
    )

    async with FullCorpusEnvironment(
        config,
        OfficeQATask(uid="visual", question="What value?", answer="42"),
        run_id="run",
    ) as env:
        result = await agent.run(env)

    multimodal = backend.calls[1][0][-1]["content"]
    assert multimodal[0]["type"] == "image"
    assert multimodal[0]["image"] is not None
    assert multimodal[1] == {
        "type": "text",
        "text": "Document: bulletin\nPDF page: 3\n\nParsed page text:\nParsed value: 42",
    }
    assert result.submission == "42"
    # Completed trajectories remain JSON serializable without embedding a
    # potentially multi-megabyte Pillow object.
    image_trace = result.messages[3]["content"][0]["image"]
    assert image_trace == {
        "kind": "rendered_pdf_page",
        "size": [100, 200],
        "mode": "RGB",
    }


@pytest.mark.asyncio
async def test_agent_text_search_triggers_pdf_page_for_next_generation(
    tmp_path,
    monkeypatch,
):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "bulletin.txt").write_text(
        "National defense expenditures: 42.\n",
        encoding="utf-8",
    )
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pages_dir / "bulletin.txt").write_text(
        "--- PAGE 2 ---\n\nUnrelated currency table.\n\n"
        "--- PAGE 7 ---\n\nNational defense expenditures: 42.\n",
        encoding="utf-8",
    )
    (pdf_dir / "bulletin.pdf").write_bytes(b"placeholder")

    class FakeImage:
        size = (100, 200)
        mode = "RGB"

    rendered_image = FakeImage()
    monkeypatch.setattr(
        FullCorpusEnvironment,
        "_render_pdf_page",
        staticmethod(lambda path, page, dpi: rendered_image),
    )
    backend = FakeGenerationBackend(
        [
            GeneratedTurn(
                    text=(
                        '```bash\ngrep -ni "national defense" '
                    "officeqa_corpus/bulletin.txt\n```"
                )
            ),
            GeneratedTurn(text='```bash\nsubmit "42"\n```'),
        ]
    )
    agent = OfficeQAAgent(backend, AgentConfig(step_limit=2))
    config = EnvironmentConfig(
        corpus_dir=corpus_dir,
        pdf_dir=pdf_dir,
        page_text_dir=pages_dir,
        workspace_root=tmp_path / "workspaces",
        sandbox_image=None,
        command_timeout_sec=5,
        max_observation_chars=1_000,
        sandbox_runtime="local",
        enable_visual_pdf_tool=True,
    )
    task = OfficeQATask(
        uid="automatic-visual",
        question="What were the national defense expenditures?",
        answer="42",
        source_files="bulletin.txt",
    )

    async with FullCorpusEnvironment(config, task, run_id="run") as env:
        result = await agent.run(env)

    initial_content = backend.calls[0][0][1]["content"]
    assert isinstance(initial_content, str)
    search_observation = backend.calls[1][0][-1]["content"]
    assert search_observation[0]["type"] == "image"
    assert search_observation[0]["image"] is not None
    assert "National defense expenditures: 42." in search_observation[1]["text"]
    assert len(backend.calls) == 2
    assert result.submission == "42"
    assert result.steps == 2
    assert result.metadata["environment"]["automatic_page_retrievals"][0][
        "page_number"
    ] == 7


class RendezvousBackend(FakeGenerationBackend):
    """Force the first turns of two agents to be in flight simultaneously."""

    def __init__(
        self,
        turns: list[GeneratedTurn],
        *,
        ready: asyncio.Event,
        peer_ready: asyncio.Event,
    ):
        super().__init__(turns)
        self._ready = ready
        self._peer_ready = peer_ready

    async def generate(self, messages, sampling_params):
        self._ready.set()
        await asyncio.wait_for(self._peer_ready.wait(), timeout=30)
        return await super().generate(messages, sampling_params)


@pytest.mark.asyncio
async def test_concurrent_agents_with_same_uid_stay_isolated(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    task = OfficeQATask(uid="same-uid", question="Question?", answer=None)
    env_config = EnvironmentConfig(
        corpus_dir=corpus_dir,
        pdf_dir=None,
        workspace_root=tmp_path / "workspaces",
        sandbox_image=None,
        command_timeout_sec=5,
        max_observation_chars=1_000,
        sandbox_runtime="local",
    )
    agent_config = AgentConfig(step_limit=3)

    def turns(answer: str) -> list[GeneratedTurn]:
        return [
            GeneratedTurn(text='```bash\nprintf "%s" "$OFFICEQA_RUN_ID"\n```'),
            GeneratedTurn(text=f'```bash\nsubmit "{answer}"\n```'),
        ]

    ready_a = asyncio.Event()
    ready_b = asyncio.Event()
    backend_a = RendezvousBackend(
        turns("answer-a"), ready=ready_a, peer_ready=ready_b
    )
    backend_b = RendezvousBackend(
        turns("answer-b"), ready=ready_b, peer_ready=ready_a
    )

    async def run_one(backend, run_id):
        agent = OfficeQAAgent(backend, agent_config)
        env = FullCorpusEnvironment(env_config, task, run_id=run_id)
        async with env:
            result = await agent.run(env)
        return env, result

    (env_a, result_a), (env_b, result_b) = await asyncio.wait_for(
        asyncio.gather(
            run_one(backend_a, "group-1-sample-1"),
            run_one(backend_b, "group-1-sample-2"),
        ),
        timeout=60,
    )

    assert env_a.workspace != env_b.workspace
    assert env_a._container_name != env_b._container_name
    assert result_a.status == result_b.status == "submitted"
    assert result_a.submission == "answer-a"
    assert result_b.submission == "answer-b"

    observation_a = result_a.messages[3]["content"]
    observation_b = result_b.messages[3]["content"]
    assert "group-1-sample-1" in observation_a
    assert "group-1-sample-2" not in observation_a
    assert "group-1-sample-2" in observation_b
    assert "group-1-sample-1" not in observation_b

    assert [message["role"] for message in backend_a.calls[1][0]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "answer-b" not in str(result_a.messages)
    assert "answer-a" not in str(result_b.messages)


@pytest.mark.asyncio
async def test_cancelled_agent_run_removes_container(tmp_path, monkeypatch):
    docker_log = install_fake_docker(tmp_path, monkeypatch)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    task = OfficeQATask(uid="cancel-agent", question="Question?", answer=None)
    env_config = EnvironmentConfig(
        corpus_dir=corpus_dir,
        pdf_dir=None,
        workspace_root=tmp_path / "workspaces",
        sandbox_image="officeqa-sandbox:test",
        command_timeout_sec=5,
        max_observation_chars=1_000,
    )

    class HangingBackend:
        def __init__(self):
            self.started = asyncio.Event()

        async def generate(self, messages, sampling_params):
            self.started.set()
            await asyncio.Event().wait()

    backend = HangingBackend()
    agent = OfficeQAAgent(backend, AgentConfig(step_limit=3))
    env = FullCorpusEnvironment(env_config, task, run_id="cancel-docker-run")

    async def run_agent() -> None:
        async with env:
            await agent.run(env)

    runner = asyncio.create_task(run_agent())
    await asyncio.wait_for(backend.started.wait(), timeout=10)

    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner

    assert not env.is_active
    assert env.command_container_id is None
    assert "rm -f fake-agent-container" in docker_log.read_text(encoding="utf-8")
