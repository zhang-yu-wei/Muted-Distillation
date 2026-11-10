import asyncio
import sys
from types import SimpleNamespace

import pytest

from officeqa_rl.backends.vllm import VLLMGenerationBackend


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeEngineArgs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered prompt"


class FakeAsyncLLM:
    def __init__(self, engine_args):
        self.init_kwargs = engine_args.kwargs
        self.tokenizer = FakeTokenizer()
        self.generate_calls = []
        self.data_parallel_ranks = []

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
        await asyncio.sleep(0.05)
        self.generate_calls.append((prompt, sampling_params, request_id))
        self.data_parallel_ranks.append(data_parallel_rank)
        completion = SimpleNamespace(
            text="plain response with no action parsing",
            token_ids=[31, 32],
            finish_reason="stop",
        )
        yield SimpleNamespace(
            prompt_token_ids=[11, 12, 13],
            outputs=[completion],
        )


def fake_vllm_module():
    return SimpleNamespace(
        AsyncLLMEngine=FakeAsyncLLM,
        AsyncEngineArgs=FakeEngineArgs,
        SamplingParams=FakeSamplingParams,
    )


@pytest.mark.asyncio
async def test_vllm_backend_renders_and_generates_without_parsing(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm_module())
    backend = VLLMGenerationBackend(
        "fake/model",
        max_tokens=128,
        tensor_parallel_size=2,
    )
    messages = [
        {"role": "system", "content": "system", "extra": {"ignored": True}},
        {"role": "tool", "content": "observation"},
        {"role": "exit", "content": "not sent"},
    ]

    generation = asyncio.create_task(
        backend.generate(messages, {"temperature": 0.2, "top_p": 0.9})
    )
    await asyncio.sleep(0.01)
    assert not generation.done()
    turn = await generation

    assert backend.llm.init_kwargs == {
        "model": "fake/model",
        "tensor_parallel_size": 2,
    }
    assert backend.tokenizer.calls == [
        (
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "observation"},
            ],
            {"tokenize": False, "add_generation_prompt": True},
        )
    ]
    prompt, params, request_id = backend.llm.generate_calls[0]
    assert prompt == {"prompt": "rendered prompt"}
    assert params.kwargs == {
        "temperature": 0.2,
        "max_tokens": 128,
        "top_p": 0.9,
    }
    assert request_id == "officeqa-0"
    assert turn.text == "plain response with no action parsing"
    assert turn.prompt_ids == [11, 12, 13]
    assert turn.output_ids == [31, 32]
    assert turn.output_log_probs is None
    assert turn.finish_reason == "stop"
    assert backend.get_stats() == {"n_calls": 1, "cost": 0.0}


@pytest.mark.asyncio
async def test_vllm_backend_forwards_enable_thinking_to_chat_template(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm_module())
    backend = VLLMGenerationBackend(
        "google/gemma-4-12B-it",
        max_tokens=16,
        enable_thinking=True,
    )

    await backend.generate([{"role": "user", "content": "question"}], {})

    assert backend.tokenizer.calls[0][1] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": True,
    }
    assert "enable_thinking" not in backend.llm.init_kwargs


@pytest.mark.asyncio
async def test_vllm_backend_batches_concurrent_calls(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm_module())
    backend = VLLMGenerationBackend("fake/model", max_tokens=16)

    started = asyncio.get_running_loop().time()
    turns = await asyncio.gather(
        backend.generate([{"role": "user", "content": "q1"}], {}),
        backend.generate([{"role": "user", "content": "q2"}], {}),
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert [turn.text for turn in turns] == [
        "plain response with no action parsing",
        "plain response with no action parsing",
    ]
    request_ids = {call[2] for call in backend.llm.generate_calls}
    assert request_ids == {"officeqa-0", "officeqa-1"}
    # Both fake generations sleep 0.05s; overlapping them proves the calls
    # were in flight together rather than serialized.
    assert elapsed < 0.09


@pytest.mark.asyncio
async def test_vllm_backend_pins_episode_turns_to_one_data_parallel_rank(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm_module())
    backend = VLLMGenerationBackend(
        "fake/model",
        max_tokens=16,
        data_parallel_size=8,
    )

    episode = backend.for_episode("UID0001.s0-run")
    await episode.generate([{"role": "user", "content": "turn 1"}], {})
    await episode.generate([{"role": "user", "content": "turn 2"}], {})

    assert len(set(backend.llm.data_parallel_ranks)) == 1
    assert 0 <= backend.llm.data_parallel_ranks[0] < 8


def test_vllm_backend_places_new_episodes_round_robin(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm_module())
    backend = VLLMGenerationBackend(
        "fake/model",
        max_tokens=16,
        data_parallel_size=4,
    )

    ranks = [
        backend.for_episode(f"episode-{index}")._data_parallel_rank
        for index in range(8)
    ]

    assert ranks == [0, 1, 2, 3, 0, 1, 2, 3]


@pytest.mark.asyncio
async def test_vllm_backend_sends_multimodal_image_with_matching_placeholder(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm_module())
    backend = VLLMGenerationBackend("fake/model", max_tokens=16)
    image = object()

    await backend.generate(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Read this PDF page."},
                ],
            }
        ],
        {},
    )

    assert backend.tokenizer.calls[0][0] == [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Read this PDF page."},
            ],
        }
    ]
    model_input = backend.llm.generate_calls[0][0]
    assert model_input["prompt"] == "rendered prompt"
    assert model_input["multi_modal_data"] == {"image": image}
