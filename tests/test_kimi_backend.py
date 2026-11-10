import pytest

from officeqa_rl.backends.kimi import KimiAPIError, KimiGenerationBackend


def kimi_response(*, answer="9", reasoning="Nine remain."):
    return {
        "model": "kimi-k3",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "reasoning_content": reasoning,
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def test_uses_single_api_key(monkeypatch):
    backend = KimiGenerationBackend("kimi-k3", api_key="key-a")
    used_keys = []

    def fake_post(api_key, payload):
        del payload
        used_keys.append(api_key)
        return kimi_response()

    monkeypatch.setattr(backend, "_post_json", fake_post)

    first = backend._generate_sync([{"role": "user", "content": "one"}], {})
    second = backend._generate_sync([{"role": "user", "content": "two"}], {})

    assert used_keys == ["key-a", "key-a"]
    assert first.reasoning_content == "Nine remain."
    assert second.text == "9"
    assert backend.get_stats()["n_calls"] == 2


def test_capacity_error_retries_same_key(monkeypatch):
    backend = KimiGenerationBackend(
        "kimi-k3",
        api_key="key-a",
        quota_max_retries=2,
    )
    used_keys = []

    def fake_post(api_key, payload):
        del payload
        used_keys.append(api_key)
        if len(used_keys) == 1:
            raise KimiAPIError(429, "quota exhausted")
        return kimi_response()

    monkeypatch.setattr(backend, "_post_json", fake_post)
    monkeypatch.setattr("officeqa_rl.backends.kimi.time.sleep", lambda _: None)

    turn = backend._generate_sync([{"role": "user", "content": "hello"}], {})

    assert turn.text == "9"
    assert used_keys == ["key-a", "key-a"]


def test_replays_reasoning_content_and_omits_sampling_parameters(monkeypatch):
    backend = KimiGenerationBackend(
        "kimi-k3",
        api_key="key-a",
        reasoning_effort="high",
        max_tokens=256,
    )
    payloads = []

    def fake_post(api_key, payload):
        del api_key
        payloads.append(payload)
        return kimi_response()

    monkeypatch.setattr(backend, "_post_json", fake_post)
    backend._generate_sync(
        [
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "reasoning_content": "I should inspect the file.",
                "content": "```bash\nls\n```",
            },
            {"role": "user", "content": "file.txt"},
        ],
        {"temperature": 0.0, "top_p": 0.1},
    )

    payload = payloads[0]
    assert payload["messages"][1]["reasoning_content"] == "I should inspect the file."
    assert payload["reasoning_effort"] == "high"
    assert payload["max_completion_tokens"] == 256
    assert "temperature" not in payload
    assert "top_p" not in payload


def test_requires_at_least_one_api_key(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MOONSHOT_API_KEY"):
        KimiGenerationBackend("kimi-k3")


def test_stats_never_contain_key_material(monkeypatch):
    backend = KimiGenerationBackend("kimi-k3", api_key="secret-a")
    monkeypatch.setattr(backend, "_post_json", lambda *_: kimi_response())
    backend._generate_sync([{"role": "user", "content": "hello"}], {})

    stats = backend.get_stats()

    assert "secret" not in repr(stats)
