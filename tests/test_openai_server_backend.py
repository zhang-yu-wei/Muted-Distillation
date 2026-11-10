import asyncio

import pytest
from PIL import Image

from officeqa_rl.backends.openai_server import (
    OpenAIServerAPIError,
    OpenAIServerGenerationBackend,
)


def server_response(*, content="391", reasoning=None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "model": "/mnt/local-ssd/GLM-5.2",
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 135,
            "total_tokens": 160,
        },
    }


def make_backend(**kwargs):
    return OpenAIServerGenerationBackend(
        "/mnt/local-ssd/GLM-5.2",
        base_url="http://10.0.0.1:8000/v1",
        **kwargs,
    )


def test_splits_inline_think_reasoning_out_of_content(monkeypatch):
    backend = make_backend()
    inline = (
        "Plan: run\n```bash\nls /data\n```\nto see the files.</think>"
        "Listing first.\n```bash\nls /workspace\n```"
    )
    monkeypatch.setattr(backend, "_post_json", lambda payload: server_response(content=inline))

    turn = backend._generate_sync([{"role": "user", "content": "go"}], {})

    # The fenced block inside reasoning must not reach the action parser.
    assert turn.text == "Listing first.\n```bash\nls /workspace\n```"
    assert turn.reasoning_content.startswith("Plan: run")
    assert "</think>" not in turn.text


def test_unclosed_think_is_all_reasoning(monkeypatch):
    backend = make_backend()
    truncated = "<think>Maybe I should run\n```bash\nrm -rf /\n```\nno wait"
    monkeypatch.setattr(
        backend, "_post_json", lambda payload: server_response(content=truncated)
    )

    turn = backend._generate_sync([{"role": "user", "content": "go"}], {})

    assert turn.text == ""
    assert turn.reasoning_content.startswith("Maybe I should run")


def test_prefers_server_parsed_reasoning_field(monkeypatch):
    backend = make_backend()
    monkeypatch.setattr(
        backend,
        "_post_json",
        lambda payload: server_response(content="391", reasoning="17*23=391."),
    )

    turn = backend._generate_sync([{"role": "user", "content": "17*23?"}], {})

    assert turn.text == "391"
    assert turn.reasoning_content == "17*23=391."


def test_passes_sampling_params_and_never_replays_reasoning(monkeypatch):
    backend = make_backend(max_tokens=256)
    payloads = []

    def fake_post(payload):
        payloads.append(payload)
        return server_response()

    monkeypatch.setattr(backend, "_post_json", fake_post)
    backend._generate_sync(
        [
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": "```bash\nls\n```",
                "reasoning_content": "I should inspect the file.",
            },
            {"role": "user", "content": "file.txt"},
        ],
        {"temperature": 0.0},
    )

    payload = payloads[0]
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 256
    assert payload["messages"][1] == {"role": "assistant", "content": "```bash\nls\n```"}


def test_converts_rendered_page_image_to_openai_vision_content(monkeypatch):
    backend = make_backend()
    payloads = []

    def fake_post(payload):
        payloads.append(payload)
        return server_response()

    monkeypatch.setattr(backend, "_post_json", fake_post)
    image = Image.new("RGB", (3, 2), "white")

    backend._generate_sync(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Parsed page text"},
                ],
            }
        ],
        {},
    )

    content = payloads[0]["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "Parsed page text"}


def test_works_without_api_key_and_sends_bearer_when_configured(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert make_backend()._api_key is None
    assert make_backend(api_key="secret")._api_key == "secret"
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert make_backend()._api_key == "from-env"


def test_episode_affinity_rotates_and_stays_on_one_vllm_dp_rank(monkeypatch):
    backend = make_backend(data_parallel_size=2)
    ranks = []

    def fake_post(payload, *, data_parallel_rank=None):
        ranks.append(data_parallel_rank)
        return server_response()

    monkeypatch.setattr(backend, "_post_json", fake_post)
    episode0 = backend.for_episode("episode-0")
    episode1 = backend.for_episode("episode-1")

    asyncio.run(episode0.generate([{"role": "user", "content": "first"}], {}))
    asyncio.run(episode0.generate([{"role": "user", "content": "second"}], {}))
    asyncio.run(episode1.generate([{"role": "user", "content": "third"}], {}))

    assert ranks == [0, 0, 1]


def test_capacity_error_retries(monkeypatch):
    backend = make_backend(quota_max_retries=2)
    attempts = []

    def fake_post(payload):
        attempts.append(payload)
        if len(attempts) == 1:
            raise OpenAIServerAPIError(503, "loading")
        return server_response()

    monkeypatch.setattr(backend, "_post_json", fake_post)
    monkeypatch.setattr("officeqa_rl.backends.openai_server.time.sleep", lambda _: None)

    turn = backend._generate_sync([{"role": "user", "content": "hello"}], {})

    assert turn.text == "391"
    assert len(attempts) == 2
    assert backend.get_stats()["usage_totals"]["total_tokens"] == 160
