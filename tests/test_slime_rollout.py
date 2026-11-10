"""Integration tests for the slime per-sample rollout hook.

Everything runs without a GPU, Docker, or slime training: a scripted local
fake SGLang router serves exact token ids, the real agent/environment run
against a temporary local-sandbox corpus, and the hook exports slime Samples.
"""

import asyncio
from types import SimpleNamespace

import pytest
from aiohttp import web

pytest.importorskip("slime", reason="requires the vendored slime on PYTHONPATH")

from slime.utils.types import Sample

from officeqa_rl.training import rollout
from tests.sglang_fakes import ByteTokenizer, FakeSGLangServer, sglang_response

SEARCH_TURN = 'Let me check the corpus.\n```bash\ncat officeqa_corpus/example.txt\n```'
SUBMIT_TURN = 'Found it.\n```bash\nsubmit "42"\n```'


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch):
    monkeypatch.setattr(rollout, "_get_tokenizer", lambda args: ByteTokenizer())


@pytest.fixture
def corpus_dir(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "example.txt").write_text("revenue: 42\n", encoding="utf-8")
    return corpus


def scripted_responder(texts):
    queue = list(texts)

    async def responder(payload):
        return sglang_response(queue.pop(0))

    return responder


def make_args(server, corpus_dir, tmp_path, **overrides):
    values = dict(
        hf_checkpoint="fake-actor",
        sglang_router_ip=server.host,
        sglang_router_port=server.port,
        rollout_max_context_len=8192,
        eval_max_context_len=8192,
        officeqa_corpus_dir=str(corpus_dir),
        officeqa_workspace_root=str(tmp_path / "workspaces"),
        officeqa_sandbox_runtime="local",
        officeqa_sandbox_image=None,
        officeqa_step_limit=4,
        officeqa_command_timeout_sec=10,
        officeqa_max_observation_chars=4000,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_base_sample(**overrides):
    values = dict(
        index=3,
        group_index=1,
        rollout_id=7,
        prompt="What is the revenue?",
        label="42",
        metadata={"uid": "UID0001", "source_files": "example.txt", "difficulty": "easy"},
    )
    values.update(overrides)
    return Sample(**values)


async def test_submit_rollout_exports_exact_tokens_and_masks(corpus_dir, tmp_path):
    async with FakeSGLangServer(scripted_responder([SEARCH_TURN, SUBMIT_TURN])) as server:
        args = make_args(server, corpus_dir, tmp_path)
        sample = await rollout.generate(
            args,
            make_base_sample(),
            {"temperature": 0.7, "max_new_tokens": 512},
        )

    assert isinstance(sample, Sample)  # a linear trajectory stays one sample
    assert sample.status == Sample.Status.COMPLETED
    assert sample.reward == 1.0
    assert sample.rollout_id == 7

    # Exact-token contract: the training sequence is exactly the served ids,
    # never a re-tokenization of decoded text.
    prompt_1 = server.generate_payloads[0]["input_ids"]
    output_1 = [ord(ch) for ch in SEARCH_TURN]
    prompt_2 = server.generate_payloads[1]["input_ids"]
    output_2 = [ord(ch) for ch in SUBMIT_TURN]
    assert prompt_2[: len(prompt_1) + len(output_1)] == prompt_1 + output_1
    assert sample.tokens == prompt_2 + output_2

    observation_len = len(prompt_2) - len(prompt_1) - len(output_1)
    assert sample.response_length == len(sample.tokens) - len(prompt_1)
    assert sample.loss_mask == (
        [1] * len(output_1) + [0] * observation_len + [1] * len(output_2)
    )
    assert sample.rollout_log_probs == (
        [-0.5 * (i + 1) for i in range(len(output_1))]
        + [0.0] * observation_len
        + [-0.5 * (i + 1) for i in range(len(output_2))]
    )

    # The masked span is the environment observation of the executed command.
    response_tokens = sample.tokens[-sample.response_length :]
    masked_text = "".join(
        chr(token)
        for token, mask in zip(response_tokens, sample.loss_mask, strict=True)
        if mask == 0
    )
    assert "Exit code: 0" in masked_text
    assert "revenue: 42" in masked_text
    assert sample.response == "".join(chr(token) for token in response_tokens)

    # Sampling params flow through to the router unchanged (minus clamping).
    assert server.generate_payloads[0]["sampling_params"]["temperature"] == 0.7
    assert server.generate_payloads[0]["return_logprob"] is True

    assert sample.metadata["uid"] == "UID0001"
    assert sample.metadata["submission"] == "42"
    assert sample.metadata["exit_status"] == "submitted"
    assert sample.metadata["steps"] == 2
    assert sample.metadata["truncated"] is False
    assert "idx3-grp1-rid7-" in sample.metadata["run_id"]
    assert "abort_reason" not in sample.metadata


async def test_wrong_submission_scores_zero(corpus_dir, tmp_path):
    wrong = 'Done.\n```bash\nsubmit "999"\n```'
    async with FakeSGLangServer(scripted_responder([wrong])) as server:
        args = make_args(server, corpus_dir, tmp_path)
        sample = await rollout.generate(args, make_base_sample(), {"max_new_tokens": 512})

    assert sample.status == Sample.Status.COMPLETED
    assert sample.reward == 0.0
    assert sample.metadata["submission"] == "999"


async def test_tolerance_metadata_override(corpus_dir, tmp_path):
    near_miss = 'Done.\n```bash\nsubmit "43"\n```'  # ~2.4% off 42
    async with FakeSGLangServer(scripted_responder([near_miss])) as server:
        args = make_args(server, corpus_dir, tmp_path)
        base = make_base_sample(
            metadata={"uid": "UID0001", "officeqa_reward_tolerance": 0.05}
        )
        sample = await rollout.generate(args, base, {"max_new_tokens": 512})

    assert sample.reward == 1.0


async def test_format_error_turn_is_trained_and_feedback_is_masked(corpus_dir, tmp_path):
    invalid = "I should look around first."  # no fenced command
    async with FakeSGLangServer(scripted_responder([invalid, SUBMIT_TURN])) as server:
        args = make_args(server, corpus_dir, tmp_path)
        sample = await rollout.generate(args, make_base_sample(), {"max_new_tokens": 512})

    assert sample.status == Sample.Status.COMPLETED
    assert sample.reward == 1.0
    assert sample.metadata["steps"] == 2

    prompt_1 = server.generate_payloads[0]["input_ids"]
    output_1 = [ord(ch) for ch in invalid]
    prompt_2 = server.generate_payloads[1]["input_ids"]
    output_2 = [ord(ch) for ch in SUBMIT_TURN]
    assert sample.tokens == prompt_2 + output_2
    feedback_len = len(prompt_2) - len(prompt_1) - len(output_1)
    # The ill-formed model turn still trains; the format-error feedback does not.
    assert sample.loss_mask == (
        [1] * len(output_1) + [0] * feedback_len + [1] * len(output_2)
    )
    masked_text = "".join(
        chr(token)
        for token, mask in zip(
            sample.tokens[-sample.response_length :], sample.loss_mask, strict=True
        )
        if mask == 0
    )
    assert "Format error" in masked_text


async def test_step_limit_yields_truncated_zero_reward(corpus_dir, tmp_path):
    looping = "Still looking.\n```bash\necho hi\n```"
    async with FakeSGLangServer(scripted_responder([looping, looping])) as server:
        args = make_args(server, corpus_dir, tmp_path, officeqa_step_limit=2)
        sample = await rollout.generate(args, make_base_sample(), {"max_new_tokens": 512})

    assert sample.status == Sample.Status.TRUNCATED
    assert sample.reward == 0.0
    assert sample.metadata["exit_status"] == "limits_exceeded"
    assert sample.metadata["submission"] == "TIMEOUT"
    assert sample.metadata["steps"] == 2
    assert not sample.remove_sample


async def test_invalid_task_returns_removable_aborted_sample(corpus_dir, tmp_path):
    async with FakeSGLangServer(scripted_responder([])) as server:
        args = make_args(server, corpus_dir, tmp_path)
        base = make_base_sample(metadata={})  # no uid
        sample = await rollout.generate(args, base, {"max_new_tokens": 512})

    assert sample is base
    assert sample.status == Sample.Status.ABORTED
    assert sample.remove_sample is True
    assert sample.reward == 0.0
    assert sample.metadata["abort_reason"].startswith("invalid_task:")
    assert server.generate_payloads == []


async def test_missing_label_aborts_before_generation(corpus_dir, tmp_path):
    async with FakeSGLangServer(scripted_responder([])) as server:
        args = make_args(server, corpus_dir, tmp_path)
        sample = await rollout.generate(
            args, make_base_sample(label=None), {"max_new_tokens": 512}
        )

    assert sample.status == Sample.Status.ABORTED
    assert "no label" in sample.metadata["abort_reason"]
    assert server.generate_payloads == []


async def test_missing_rollout_config_aborts(corpus_dir, tmp_path):
    async with FakeSGLangServer(scripted_responder([])) as server:
        args = make_args(server, corpus_dir, tmp_path, officeqa_corpus_dir=None)
        sample = await rollout.generate(args, make_base_sample(), {"max_new_tokens": 512})

    assert sample.status == Sample.Status.ABORTED
    assert sample.remove_sample is True
    assert sample.metadata["abort_reason"].startswith("invalid_rollout_config:")
    assert "officeqa_corpus_dir" in sample.metadata["abort_reason"]


async def test_environment_failure_aborts_without_raising(tmp_path):
    async with FakeSGLangServer(scripted_responder([])) as server:
        args = make_args(server, tmp_path / "missing-corpus", tmp_path)
        sample = await rollout.generate(args, make_base_sample(), {"max_new_tokens": 512})

    assert sample.status == Sample.Status.ABORTED
    assert sample.remove_sample is True
    assert sample.metadata["abort_reason"].startswith("environment_failure:")
    assert server.generate_payloads == []


async def test_router_error_aborts_with_agent_failure(corpus_dir, tmp_path):
    async def responder(payload):
        return web.Response(status=503, text="router overloaded")

    async with FakeSGLangServer(responder) as server:
        args = make_args(server, corpus_dir, tmp_path)
        sample = await rollout.generate(args, make_base_sample(), {"max_new_tokens": 512})

    assert sample.status == Sample.Status.ABORTED
    assert sample.remove_sample is True
    assert sample.metadata["abort_reason"].startswith("agent_failure: error: RuntimeError")


async def test_exhausted_context_yields_empty_trajectory_abort(corpus_dir, tmp_path):
    async with FakeSGLangServer(scripted_responder([])) as server:
        args = make_args(
            server, corpus_dir, tmp_path, rollout_max_context_len=8, officeqa_step_limit=2
        )
        sample = await rollout.generate(args, make_base_sample(), {"max_new_tokens": 512})

    # Every turn is an empty finish_reason="length" turn: nothing trainable.
    assert sample.status == Sample.Status.ABORTED
    assert sample.metadata["abort_reason"].startswith("empty_trajectory:")
    assert server.generate_payloads == []


async def test_evaluation_uses_eval_context_budget(corpus_dir, tmp_path):
    async with FakeSGLangServer(scripted_responder([SUBMIT_TURN])) as server:
        args = make_args(
            server, corpus_dir, tmp_path, rollout_max_context_len=8192, eval_max_context_len=3000
        )
        sample = await rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 100_000}, evaluation=True
        )

    assert sample.status == Sample.Status.COMPLETED
    payload = server.generate_payloads[0]
    assert payload["sampling_params"]["max_new_tokens"] == 3000 - len(payload["input_ids"])


async def test_routing_key_header_follows_consistent_hashing(corpus_dir, tmp_path):
    async with FakeSGLangServer(scripted_responder([SUBMIT_TURN])) as server:
        args = make_args(server, corpus_dir, tmp_path, router_policy="consistent_hashing")
        base = make_base_sample(session_id="session-abc")
        await rollout.generate(args, base, {"max_new_tokens": 512})

    assert server.generate_headers[0]["X-SMG-Routing-Key"] == "session-abc"


async def test_concurrent_runs_of_same_uid_get_unique_run_ids(corpus_dir, tmp_path):
    async with FakeSGLangServer(
        scripted_responder([SUBMIT_TURN, SUBMIT_TURN])
    ) as server:
        args = make_args(server, corpus_dir, tmp_path)
        first, second = await asyncio.gather(
            rollout.generate(args, make_base_sample(), {"max_new_tokens": 512}),
            rollout.generate(args, make_base_sample(), {"max_new_tokens": 512}),
        )

    assert first.status == Sample.Status.COMPLETED
    assert second.status == Sample.Status.COMPLETED
    assert first.metadata["run_id"] != second.metadata["run_id"]

    workspaces = list((tmp_path / "workspaces").iterdir())
    assert len(workspaces) == 2


def test_sock_read_timeout_is_configurable_and_defaults_to_900():
    """A generate call that merely queues behind long episodes must not be
    abandoned: the backend read timeout is a knob, not a constant."""
    import types
    from officeqa_rl.training.rollout import _create_backend

    base = dict(
        sglang_router_ip="127.0.0.1", sglang_router_port=1,
        rollout_max_context_len=1024, eval_max_context_len=1024,
    )
    d = _create_backend(types.SimpleNamespace(**base), evaluation=False, routing_key=None)
    assert d.sock_read_timeout_sec == 900.0
    r = _create_backend(
        types.SimpleNamespace(**base, officeqa_sock_read_timeout_sec=2700),
        evaluation=False, routing_key=None,
    )
    assert r.sock_read_timeout_sec == 2700.0
