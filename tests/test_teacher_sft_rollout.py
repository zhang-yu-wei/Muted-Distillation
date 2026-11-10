"""Tests for the online teacher-SFT rollout function.

The refill/queue loop is tested with a scripted ``_run_groups`` and a fake
data source (no network, no GPU). The episode-to-samples conversion is tested
against the real Qwen3.5 tokenizer when it is available on this machine.
"""

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("slime", reason="requires the vendored slime on PYTHONPATH")

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.types import Sample

from officeqa_rl.sft_samples import per_turn_training_messages
from officeqa_rl.training import teacher_sft_rollout as tsr

QWEN_CHECKPOINT = Path("/mnt/Qwen3.5-4B")


def _loop_args(**overrides):
    defaults = dict(
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        officeqa_sft_filter="correct",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class FakeDataSource:
    def __init__(self, n_samples_per_prompt: int) -> None:
        self.n = n_samples_per_prompt
        self.calls: list[int] = []
        self._index = 0

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        self.calls.append(num_samples)
        groups = []
        for _ in range(num_samples):
            group = []
            for _ in range(self.n):
                group.append(
                    Sample(index=self._index, prompt="q", label="42", metadata={"uid": "U"})
                )
                self._index += 1
            groups.append(group)
        return groups


def _episode(rollout_id: int) -> list[Sample]:
    return [Sample(rollout_id=rollout_id, tokens=[1, 2], response_length=1, loss_mask=[1])]


def _scripted_state_and_rounds(monkeypatch, rounds: list[list[list[Sample]]]):
    """Route _get_state to a bare queue-holding stub and _run_groups to a
    script that pops one round of episodes per call."""
    state = SimpleNamespace(episode_queue=[])
    monkeypatch.setattr(tsr, "_get_state", lambda args: state)
    script = list(rounds)

    async def fake_run_groups(args, state_arg, groups, runner=None):
        assert state_arg is state
        # OfficeQA's own entry point leaves the runner unset; the SSB1 twin
        # passes its episode runner through this argument.
        assert runner is None
        episodes = script.pop(0)
        attempts = sum(len(group) for group in groups)
        stats = {
            "run": attempts,
            "kept": len(episodes),
            "filtered": attempts - len(episodes),
            "failed": 0,
            "passed": len(episodes),
        }
        return episodes, stats

    monkeypatch.setattr(tsr, "_run_groups", fake_run_groups)
    return state


def test_emits_exact_episode_count_and_carries_surplus(monkeypatch):
    args = _loop_args()  # target = 2 * 2 = 4 episodes per call
    state = _scripted_state_and_rounds(
        monkeypatch,
        rounds=[
            [_episode(0), _episode(1), _episode(2)],  # 3 < 4 -> refill
            [_episode(3), _episode(4), _episode(5)],  # queue 6 >= 4
            [_episode(6), _episode(7)],  # second call: 2 carried + 2 new
        ],
    )
    source = FakeDataSource(args.n_samples_per_prompt)

    output = tsr.generate_rollout(args, rollout_id=0, data_source=source)

    assert isinstance(output, RolloutFnTrainOutput)
    assert [episode[0].rollout_id for episode in output.samples] == [0, 1, 2, 3]
    assert [episode[0].rollout_id for episode in state.episode_queue] == [4, 5]
    assert source.calls == [2, 2]
    assert output.metrics["rollout/teacher_sft/episodes_run"] == 8.0
    assert output.metrics["rollout/teacher_sft/queue_carryover"] == 2.0

    output = tsr.generate_rollout(args, rollout_id=1, data_source=source)
    assert [episode[0].rollout_id for episode in output.samples] == [4, 5, 6, 7]
    assert state.episode_queue == []
    assert source.calls == [2, 2, 2]


def test_consecutive_empty_rounds_raise(monkeypatch):
    args = _loop_args()
    _scripted_state_and_rounds(monkeypatch, rounds=[[], [], []])
    source = FakeDataSource(args.n_samples_per_prompt)

    with pytest.raises(RuntimeError, match="no usable episodes"):
        tsr.generate_rollout(args, rollout_id=0, data_source=source)
    assert source.calls == [2, 2, 2]


def test_one_empty_round_between_productive_rounds_is_fine(monkeypatch):
    args = _loop_args(rollout_batch_size=1)  # target = 2
    _scripted_state_and_rounds(
        monkeypatch, rounds=[[_episode(0)], [], [_episode(1)]]
    )
    source = FakeDataSource(args.n_samples_per_prompt)

    output = tsr.generate_rollout(args, rollout_id=0, data_source=source)
    assert [episode[0].rollout_id for episode in output.samples] == [0, 1]


def test_invalid_filter_mode_rejected():
    args = _loop_args(officeqa_sft_filter="bogus")
    with pytest.raises(ValueError, match="officeqa_sft_filter"):
        tsr.generate_rollout(args, rollout_id=0, data_source=FakeDataSource(2))


def test_evaluation_branch_asserts():
    with pytest.raises(AssertionError, match="debug-train-only"):
        tsr.generate_rollout(_loop_args(), rollout_id=0, data_source=None, evaluation=True)


def test_per_turn_training_messages_masks_only_target():
    cleaned = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "How many?"},
        {
            "role": "assistant",
            "content": "```bash\nrg value corpus\n```",
            "reasoning_content": "Find the row.",
        },
        {"role": "user", "content": "The value is 42."},
        {"role": "assistant", "content": "```bash\nsubmit \"42\"\n```"},
    ]

    turns = per_turn_training_messages(cleaned)

    assert len(turns) == 2
    target1, messages1 = turns[0]
    assert target1 is cleaned[2]
    assert messages1[-1]["step_loss_mask"] == 1
    assert messages1[-1]["content"].startswith("<think>Find the row.</think>\n")
    target2, messages2 = turns[1]
    assert [message["role"] for message in messages2] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert messages2[2]["step_loss_mask"] == 0
    assert "<think>" not in messages2[2]["content"]
    # An unreasoned direct command stays a command-only target.
    assert messages2[-1] == {
        "role": "assistant",
        "content": "```bash\nsubmit \"42\"\n```",
        "step_loss_mask": 1,
    }


def _agent_result(status="submitted", finish_reasons=("stop", "stop")):
    return SimpleNamespace(
        status=status,
        submission="42",
        steps=2,
        messages=[
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "How many?"},
            {
                "role": "assistant",
                "content": "```bash\nrg value corpus\n```",
                "reasoning_content": "Find the row.",
            },
            {"role": "user", "content": "The value is 42."},
            {
                "role": "assistant",
                "content": "```bash\nsubmit \"42\"\n```",
                "reasoning_content": "The observation gives the answer.",
            },
            {"role": "exit", "content": "Submitted answer: 42"},
        ],
        metadata={
            "generated_turns": [{"finish_reason": reason} for reason in finish_reasons]
        },
    )


def test_persist_trace_writes_offline_builder_format(tmp_path):
    args = SimpleNamespace(officeqa_sft_trace_dir=str(tmp_path / "traces"))
    task = SimpleNamespace(uid="UID0001", question="How many?", answer="42")
    result = _agent_result()

    asyncio.run(tsr._persist_trace(args, Sample(index=5), task, result, 1.0))

    path = tmp_path / "traces" / "UID0001.s5.json"
    trace = json.loads(path.read_text(encoding="utf-8"))
    assert trace["uid"] == "UID0001"
    assert trace["question"] == "How many?"
    assert trace["ground_truth"] == "42"
    assert trace["submission"] == "42"
    assert trace["status"] == "submitted"
    assert trace["score"] == 1.0
    # The offline builder's contract: an exit-terminated message list and the
    # <uid>.s<i> sibling stem shape.
    assert trace["messages"][-1]["role"] == "exit"
    assert re.fullmatch(r"(UID\d+)(?:\.s(\d+))?", path.stem)


def test_persist_trace_disabled_without_dir(tmp_path):
    result = _agent_result()
    asyncio.run(
        tsr._persist_trace(
            SimpleNamespace(),
            Sample(index=5),
            SimpleNamespace(uid="UID0001", question="q", answer="1"),
            result,
            0.0,
        )
    )
    assert not list(tmp_path.iterdir())


def test_non_text_content_drops_episode():
    result = _agent_result()
    result.messages[3] = {"role": "user", "content": [{"type": "text", "text": "x"}]}
    samples = tsr._episode_to_samples(
        SimpleNamespace(rollout_max_context_len=0),
        SimpleNamespace(),  # guard fires before any state use
        Sample(index=1),
        SimpleNamespace(uid="UID0001"),
        result,
        1.0,
        "run-1",
        SimpleNamespace(model_name="teacher-x"),
    )
    assert samples == []


@pytest.fixture(scope="module")
def qwen_state():
    if not QWEN_CHECKPOINT.is_dir():
        pytest.skip(f"Qwen3.5 checkpoint not available at {QWEN_CHECKPOINT}")
    from slime.utils.mask_utils import MultiTurnLossMaskGenerator
    from slime.utils.processing_utils import load_tokenizer

    tokenizer = load_tokenizer(str(QWEN_CHECKPOINT), trust_remote_code=True)
    return SimpleNamespace(
        tokenizer=tokenizer,
        mask_generator=MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_5"),
    )


def test_episode_to_samples_builds_per_turn_masked_samples(qwen_state):
    args = SimpleNamespace(rollout_max_context_len=32_768)
    base = Sample(index=7, group_index=3, prompt="orig", label="42", metadata={"uid": "UID0001"})
    result = _agent_result(finish_reasons=("stop", "length"))

    samples = tsr._episode_to_samples(
        args,
        qwen_state,
        base,
        SimpleNamespace(uid="UID0001"),
        result,
        1.0,
        "run-1",
        SimpleNamespace(model_name="teacher-x"),
    )

    assert len(samples) == 2
    cleaned = result.messages[:-1]
    expected_messages = [messages for _, messages in per_turn_training_messages(cleaned)]
    for turn_number, sample in enumerate(samples, start=1):
        assert sample.rollout_id == 7
        assert sample.prompt == expected_messages[turn_number - 1]
        response_length = int(sample.response_length)
        assert response_length > 0
        assert len(sample.loss_mask) == response_length
        assert len(sample.tokens) >= response_length
        assert sum(sample.loss_mask) > 0
        metadata = sample.metadata
        assert metadata["uid"] == "UID0001"
        assert metadata["run_id"] == "run-1"
        assert metadata["exit_status"] == "submitted"
        assert metadata["assistant_turn"] == turn_number
        assert metadata["teacher_model"] == "teacher-x"
        assert metadata["target_has_reasoning"] is True
        assert sample.reward == 1.0

    trained_first = qwen_state.tokenizer.decode(
        [
            token
            for token, mask in zip(
                samples[0].tokens[-int(samples[0].response_length) :],
                samples[0].loss_mask,
            )
            if mask
        ]
    )
    assert "rg value corpus" in trained_first
    assert "Find the row." in trained_first
    assert "submit" not in trained_first

    assert samples[0].status == Sample.Status.COMPLETED
    assert samples[0].metadata["truncated"] is False
    # Turn 2's generation finished with "length": its sample is TRUNCATED.
    assert samples[1].status == Sample.Status.TRUNCATED
    assert samples[1].metadata["truncated"] is True


def test_episode_to_samples_drops_overlong_turns(qwen_state):
    args = SimpleNamespace(rollout_max_context_len=10)
    samples = tsr._episode_to_samples(
        args,
        qwen_state,
        Sample(index=1),
        SimpleNamespace(uid="UID0001"),
        _agent_result(),
        1.0,
        "run-1",
        SimpleNamespace(model_name="teacher-x"),
    )
    assert samples == []


def test_episode_to_samples_limits_exceeded_status(qwen_state):
    args = SimpleNamespace(rollout_max_context_len=32_768)
    result = _agent_result(status="limits_exceeded")
    samples = tsr._episode_to_samples(
        args,
        qwen_state,
        Sample(index=2),
        SimpleNamespace(uid="UID0001"),
        result,
        0.0,
        "run-2",
        SimpleNamespace(model_name="teacher-x"),
    )
    assert samples
    assert all(sample.status == Sample.Status.TRUNCATED for sample in samples)
    assert all(sample.reward == 0.0 for sample in samples)


def _pool(n_endpoints: int, per_endpoint: int) -> tsr._EndpointPool:
    return tsr._EndpointPool(
        [SimpleNamespace(model_name=f"teacher-{i}") for i in range(n_endpoints)],
        per_endpoint,
    )


def test_endpoint_pool_caps_concurrency_per_endpoint():
    """The Vertex 397B deployments tolerate only a few concurrent requests
    each, so the cap has to bind per endpoint, not over the pool."""
    pool = _pool(3, 2)
    live: dict[str, int] = {}
    peak: dict[str, int] = {}
    release = asyncio.Event()

    async def episode():
        async with pool.lease() as backend:
            name = backend.model_name
            live[name] = live.get(name, 0) + 1
            peak[name] = max(peak.get(name, 0), live[name])
            await release.wait()
            live[name] -= 1

    async def main():
        tasks = [asyncio.create_task(episode()) for _ in range(30)]
        # Let every admissible episode reach its endpoint before releasing.
        for _ in range(50):
            await asyncio.sleep(0)
        in_flight = sum(live.values())
        release.set()
        await asyncio.gather(*tasks)
        return in_flight

    in_flight = asyncio.run(main())
    assert in_flight == 6, "pool admits per_endpoint x endpoints episodes at once"
    assert peak == {"teacher-0": 2, "teacher-1": 2, "teacher-2": 2}


def test_endpoint_pool_balances_across_endpoints():
    """Assignment counts the episode before awaiting, so simultaneous leases
    spread instead of all reading the same minimum."""
    pool = _pool(2, 4)
    seen: list[str] = []

    async def episode():
        async with pool.lease() as backend:
            seen.append(backend.model_name)
            await asyncio.sleep(0)

    asyncio.run(_gather(episode, 8))
    assert seen.count("teacher-0") == 4
    assert seen.count("teacher-1") == 4


async def _gather(factory, count: int):
    await asyncio.gather(*(factory() for _ in range(count)))


def test_endpoint_pool_single_endpoint_is_a_plain_semaphore():
    pool = _pool(1, 3)
    assert pool.size == 1
    peak = 0
    live = 0
    release = asyncio.Event()

    async def episode():
        nonlocal peak, live
        async with pool.lease():
            live += 1
            peak = max(peak, live)
            await release.wait()
            live -= 1

    async def main():
        tasks = [asyncio.create_task(episode()) for _ in range(10)]
        for _ in range(20):
            await asyncio.sleep(0)
        release.set()
        await asyncio.gather(*tasks)

    asyncio.run(main())
    assert peak == 3


# --- SSB1 twin ---------------------------------------------------------------


def test_ssb1_entry_point_passes_its_runner_to_the_shared_driver(monkeypatch):
    """The SSB1 module reuses this batch driver with its own episode runner,
    so the queue/replacement behaviour cannot drift between benchmarks."""
    from officeqa_rl.training import ssb1_teacher_sft_rollout as ssb1_sft

    seen = {}

    def fake_generate_rollout(args, rollout_id, data_source, evaluation=False, *, runner=None):
        seen["runner"] = runner
        seen["evaluation"] = evaluation
        return "sentinel"

    monkeypatch.setattr(tsr, "generate_rollout", fake_generate_rollout)
    result = ssb1_sft.generate_rollout(object(), 0, object())

    assert result == "sentinel"
    assert seen["runner"] is ssb1_sft._run_episode
    assert seen["evaluation"] is False


def test_ssb1_runner_drops_rows_from_another_benchmark():
    """A non-SSB1 row must not reach the spreadsheet sandbox: the RL path
    forwards such rows to the OfficeQA generate, but an SFT batch has no
    fallback, so the episode is dropped instead of booting a container."""
    from officeqa_rl.training import ssb1_teacher_sft_rollout as ssb1_sft

    sample = Sample(index=0, prompt=[], label=None, metadata={"benchmark": "officeqa"})
    samples, outcome, passed = asyncio.run(
        ssb1_sft._run_episode(object(), object(), sample)
    )

    assert samples is None
    assert outcome == "failed"
    assert passed is False


def test_ssb1_trace_is_written_in_the_eval_schema(tmp_path):
    """One JSON per episode, named <uid>.s<index>.json and shaped like the
    offline eval's payload, so analysis tooling cannot tell them apart."""
    from officeqa_rl.training import ssb1_teacher_sft_rollout as ssb1_sft

    task = SimpleNamespace(
        uid="SSB1__7",
        task_id="7",
        instruction_type="cell-filling",
        instruction="Fill column Z",
        cases=[object(), object(), object()],
    )
    result = SimpleNamespace(
        submission="done",
        status="submitted",
        steps=11,
        messages=[{"role": "assistant", "content": "```bash\nls\n```"}],
        metadata={"token_usage": {"output_tokens": 4096, "model_calls": 11}},
    )
    metrics = {"hard_restriction": 0.0, "soft_restriction": 0.5, "cell_accuracy": 0.75}
    sample = Sample(index=3, prompt=[], label=None, metadata={"benchmark": "ssb1"})
    args = SimpleNamespace(officeqa_sft_trace_dir=str(tmp_path))

    asyncio.run(
        ssb1_sft._persist_trace(args, sample, task, result, metrics, "run-1", "teacher")
    )

    written = list(tmp_path.glob("*.json"))
    assert [p.name for p in written] == ["SSB1__7.s3.json"]
    payload = json.loads(written[0].read_text())
    # Same keys the eval writes, so the two are interchangeable downstream.
    for key in ("uid", "task_id", "instruction_type", "question", "submission",
                "status", "steps", "score", "token_usage", "metrics", "messages"):
        assert key in payload, key
    assert payload["question"] == "Fill column Z"
    assert payload["score"] == metrics["hard_restriction"]
    assert payload["metrics"]["soft_restriction"] == 0.5
    assert payload["metadata"]["case_count"] == 3
    assert payload["metadata"]["sample_index"] == 3


def test_ssb1_trace_is_skipped_when_no_directory_is_configured(tmp_path):
    """No trace dir means no writes and no crash."""
    from officeqa_rl.training import ssb1_teacher_sft_rollout as ssb1_sft

    args = SimpleNamespace(officeqa_sft_trace_dir="")
    asyncio.run(
        ssb1_sft._persist_trace(args, Sample(index=0, prompt=[], label=None), None,
                                None, None, "run-1", "teacher")
    )
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# gemma-4 reasoning style: reasoning_content field + thinking-on teacher prompt
# ---------------------------------------------------------------------------
GEMMA_CHECKPOINT = Path("/mnt/gemma-4-12B-it")


def test_reasoning_style_follows_loss_mask_type():
    from officeqa_rl.sft_samples import reasoning_style_for_loss_mask

    assert reasoning_style_for_loss_mask("qwen3_5") == "qwen"
    assert reasoning_style_for_loss_mask("gemma4") == "gemma"
    assert reasoning_style_for_loss_mask("gemma4_think") == "gemma"


def test_per_turn_training_messages_gemma_style_uses_reasoning_content():
    cleaned = _agent_result().messages[:-1]

    turns = per_turn_training_messages(cleaned, "gemma")

    assert len(turns) == 2
    _, messages1 = turns[0]
    assert messages1[-1]["step_loss_mask"] == 1
    assert messages1[-1]["reasoning_content"] == "Find the row."
    assert messages1[-1]["content"] == "```bash\nrg value corpus\n```"
    assert "<think>" not in messages1[-1]["content"]
    _, messages2 = turns[1]
    # Earlier turns carry neither loss nor reasoning.
    assert messages2[2] == {
        "role": "assistant",
        "content": "```bash\nrg value corpus\n```",
        "step_loss_mask": 0,
    }
    assert messages2[-1]["reasoning_content"] == "The observation gives the answer."
    with pytest.raises(ValueError):
        per_turn_training_messages(cleaned, "llama")


def test_rollout_state_passes_enable_thinking_to_the_teacher_backend(monkeypatch):
    captured = []

    class FakeBackend:
        def __init__(self, model_name, **kwargs):
            captured.append(kwargs)
            self.model_name = model_name

    monkeypatch.setattr(tsr, "VertexSGLangGenerationBackend", FakeBackend)
    monkeypatch.setattr(tsr, "load_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr(tsr, "MultiTurnLossMaskGenerator", lambda tok, tokenizer_type: tokenizer_type)

    def state(**overrides):
        args = SimpleNamespace(
            hf_checkpoint="x", loss_mask_type="gemma4_think", rollout_max_response_len=8,
            officeqa_sft_teacher_url="http://a/generate,http://b/generate",
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        captured.clear()
        return tsr._RolloutState(args)

    s = state()
    assert s.reasoning_style == "gemma"
    assert [k["enable_thinking"] for k in captured] == [None, None]
    state(officeqa_sft_enable_thinking=True)
    assert [k["enable_thinking"] for k in captured] == [True, True]
    state(officeqa_sft_enable_thinking="false")
    assert [k["enable_thinking"] for k in captured] == [False, False]
    assert state(loss_mask_type="qwen3_5").reasoning_style == "qwen"
    with pytest.raises(ValueError):
        state(officeqa_sft_enable_thinking="maybe")


@pytest.fixture(scope="module")
def gemma_tokenizer():
    if not GEMMA_CHECKPOINT.is_dir():
        pytest.skip(f"gemma-4 checkpoint not available at {GEMMA_CHECKPOINT}")
    from slime.utils.processing_utils import load_tokenizer

    return load_tokenizer(str(GEMMA_CHECKPOINT), trust_remote_code=True)


def test_gemma4_think_mask_trains_thought_and_visible_text(gemma_tokenizer):
    from slime.utils.mask_utils import MultiTurnLossMaskGenerator

    cleaned = _agent_result().messages[:-1]
    _, messages = per_turn_training_messages(cleaned, "gemma")[1]
    gen = MultiTurnLossMaskGenerator(gemma_tokenizer, tokenizer_type="gemma4_think")
    token_ids, loss_mask = gen.get_loss_mask(messages)
    assert len(token_ids) == len(loss_mask)
    response_length = gen.get_response_lengths([loss_mask])[0]
    prompt = gemma_tokenizer.decode(token_ids[:-response_length], skip_special_tokens=False)
    response = gemma_tokenizer.decode(token_ids[-response_length:], skip_special_tokens=False)
    # Thinking-on prompt, first turn's thought stripped and untrained.
    assert "<|turn>system\n<|think|>\n" in prompt
    assert "Find the row." not in prompt
    assert prompt.endswith("<|turn>model\n")
    # The whole target turn is supervised: opener, reasoning, close, action, end.
    assert response == (
        "<|channel>thought\nThe observation gives the answer.\n<channel|>"
        "```bash\nsubmit \"42\"\n```<turn|>\n"
    )
    assert all(loss_mask[-response_length:])
    assert not any(loss_mask[:-response_length])
    # The visible-only gemma4 mask keeps the thought out of the loss.
    plain = MultiTurnLossMaskGenerator(gemma_tokenizer, tokenizer_type="gemma4")
    plain_ids, plain_mask = plain.get_loss_mask(messages)
    plain_len = plain.get_response_lengths([plain_mask])[0]
    assert gemma_tokenizer.decode(plain_ids[-plain_len:], skip_special_tokens=False) == (
        "```bash\nsubmit \"42\"\n```<turn|>\n"
    )
    assert "<|think|>" not in gemma_tokenizer.decode(plain_ids, skip_special_tokens=False)
