import asyncio
import os
import sys
import time
from types import SimpleNamespace

import pytest

from officeqa_rl.backends.gemini import GeminiGenerationBackend


class FakeHttpOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeThinkingConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeGenerateContentConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakePart:
    @classmethod
    def from_text(cls, *, text):
        return SimpleNamespace(kind="text", text=text)

    @classmethod
    def from_bytes(cls, *, data, mime_type):
        return SimpleNamespace(kind="bytes", data=data, mime_type=mime_type)


class FakeContent:
    def __init__(self, *, role, parts):
        self.role = role
        self.parts = parts


class FakeUsage:
    def model_dump(self):
        return {"prompt_token_count": 10, "candidates_token_count": 4}


class FakeQuotaError(Exception):
    def __init__(self):
        super().__init__(
            "429 RESOURCE_EXHAUSTED. Quota exceeded for quota metric "
            "'GenerateContent request limit'. {'retryDelay': '0s'}"
        )
        self.code = 429
        self.status = "RESOURCE_EXHAUSTED"


class FakeOverloadedError(Exception):
    """503 load shedding, shaped like the google-genai ServerError."""

    def __init__(self):
        super().__init__(
            "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model "
            "is currently experiencing high demand.', 'status': 'UNAVAILABLE'}}"
        )
        self.code = 503
        self.status = "UNAVAILABLE"


class FakeModels:
    def __init__(self):
        self.calls = []
        self.failures_remaining = 0
        self.quota_failures_remaining = 0
        self.overload_failures_remaining = 0

    def generate_content(self, **kwargs):
        time.sleep(0.05)
        self.calls.append(kwargs)
        if self.quota_failures_remaining:
            self.quota_failures_remaining -= 1
            raise FakeQuotaError()
        if self.overload_failures_remaining:
            self.overload_failures_remaining -= 1
            raise FakeOverloadedError()
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("transient API failure")
        finish_reason = SimpleNamespace(name="STOP")
        candidate = SimpleNamespace(
            finish_reason=finish_reason,
            finish_message=None,
            avg_logprobs=-0.25,
            content=None,
        )
        return SimpleNamespace(
            text="plain Gemini response without action parsing",
            usage_metadata=FakeUsage(),
            candidates=[candidate],
            prompt_feedback=None,
        )


class FakeClient:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.models = FakeModels()


def install_fake_genai(monkeypatch):
    client = FakeClient()
    fake_genai = SimpleNamespace(Client=lambda **kwargs: _configure_client(client, kwargs))
    fake_types = SimpleNamespace(
        Content=FakeContent,
        GenerateContentConfig=FakeGenerateContentConfig,
        HttpOptions=FakeHttpOptions,
        Part=FakePart,
        ThinkingConfig=FakeThinkingConfig,
    )
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    return client


def _configure_client(client, kwargs):
    client.init_kwargs = kwargs
    return client


@pytest.mark.asyncio
async def test_gemini_backend_converts_messages_and_collects_usage(monkeypatch):
    client = install_fake_genai(monkeypatch)
    backend = GeminiGenerationBackend(
        "gemini-test-model",
        api_version="v1beta",
        max_tokens=128,
        thinking_budget=32,
        include_thoughts=True,
    )
    messages = [
        {"role": "system", "content": "system one"},
        {"role": "system", "content": [{"type": "text", "text": "system two"}]},
        {"role": "user", "content": "question"},
        {"role": "tool", "content": "observation"},
        {"role": "assistant", "content": "analysis one"},
        {"role": "assistant", "content": "analysis two"},
        {"role": "exit", "content": "not sent"},
    ]

    generation = asyncio.create_task(
        backend.generate(messages, {"temperature": 0.3, "top_p": 0.8})
    )
    await asyncio.sleep(0.01)
    assert not generation.done()
    turn = await generation

    assert client.init_kwargs["http_options"].kwargs == {"api_version": "v1beta"}
    call = client.models.calls[0]
    assert call["model"] == "gemini-test-model"
    assert [content.role for content in call["contents"]] == ["user", "model"]
    assert [part.text for part in call["contents"][0].parts] == [
        "question",
        "observation",
    ]
    assert [part.text for part in call["contents"][1].parts] == [
        "analysis one",
        "analysis two",
    ]
    config = call["config"].kwargs
    assert config["system_instruction"] == "system one\n\nsystem two"
    assert config["temperature"] == 0.3
    assert config["max_output_tokens"] == 128
    assert config["top_p"] == 0.8
    assert config["thinking_config"].kwargs == {
        "thinking_budget": 32,
        "include_thoughts": True,
    }

    assert turn.text == "plain Gemini response without action parsing"
    assert turn.prompt_ids is None
    assert turn.output_ids is None
    assert turn.output_log_probs is None
    assert turn.finish_reason == "STOP"
    stats = backend.get_stats()
    assert stats["n_calls"] == 1
    assert stats["usage_totals"] == {
        "prompt_token_count": 10,
        "candidates_token_count": 4,
    }
    assert stats["response_metadata"][0]["candidates"][0]["finish_reason"] == "STOP"


@pytest.mark.asyncio
async def test_gemini_36_uses_provider_managed_sampling(monkeypatch):
    client = install_fake_genai(monkeypatch)
    backend = GeminiGenerationBackend(
        "gemini-3.6-flash",
        max_tokens=128,
        thinking_budget=0,
    )

    await backend.generate(
        [{"role": "user", "content": "question"}],
        {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
    )

    config = client.models.calls[0]["config"].kwargs
    assert config["max_output_tokens"] == 128
    assert "temperature" not in config
    assert "top_p" not in config
    assert "top_k" not in config
    assert "thinking_config" not in config


@pytest.mark.asyncio
async def test_gemini_backend_retries_transient_failures(monkeypatch):
    client = install_fake_genai(monkeypatch)
    client.models.failures_remaining = 1
    monkeypatch.setattr("officeqa_rl.backends.gemini.time.sleep", lambda _: None)
    backend = GeminiGenerationBackend("gemini-test-model", max_retries=1)

    turn = await backend.generate(
        [{"role": "user", "content": "question"}],
        {"temperature": 0.0},
    )

    assert turn.text == "plain Gemini response without action parsing"
    assert backend.get_stats()["n_calls"] == 2


@pytest.mark.asyncio
async def test_quota_errors_use_their_own_retry_budget(monkeypatch):
    client = install_fake_genai(monkeypatch)
    client.models.quota_failures_remaining = 5
    monkeypatch.setattr("officeqa_rl.backends.gemini.time.sleep", lambda _: None)
    # max_retries=0 would fail immediately for transient errors; quota errors
    # must survive on their separate budget.
    backend = GeminiGenerationBackend("gemini-test-model", max_retries=0)

    turn = await backend.generate(
        [{"role": "user", "content": "question"}],
        {"temperature": 0.0},
    )

    assert turn.text == "plain Gemini response without action parsing"
    assert backend.get_stats()["n_calls"] == 6  # 5 quota failures + 1 success


@pytest.mark.asyncio
async def test_exhausted_quota_budget_raises(monkeypatch):
    client = install_fake_genai(monkeypatch)
    client.models.quota_failures_remaining = 5
    monkeypatch.setattr("officeqa_rl.backends.gemini.time.sleep", lambda _: None)
    backend = GeminiGenerationBackend(
        "gemini-test-model",
        max_retries=3,
        quota_max_retries=2,
    )

    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        await backend.generate(
            [{"role": "user", "content": "question"}],
            {"temperature": 0.0},
        )

    assert len(client.models.calls) == 3  # initial call + 2 quota retries


def install_fake_genai_multi(monkeypatch):
    """Fake genai module that creates a fresh client per Client(...) call,
    recording each one so key-pool rotation can be asserted."""
    clients = []

    def make_client(**kwargs):
        client = FakeClient(**kwargs)
        clients.append(client)
        return client

    fake_genai = SimpleNamespace(Client=make_client)
    fake_types = SimpleNamespace(
        Content=FakeContent,
        GenerateContentConfig=FakeGenerateContentConfig,
        HttpOptions=FakeHttpOptions,
        Part=FakePart,
        ThinkingConfig=FakeThinkingConfig,
    )
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    return clients


@pytest.mark.asyncio
async def test_api_key_pool_round_robins_calls(monkeypatch):
    clients = install_fake_genai_multi(monkeypatch)
    backend = GeminiGenerationBackend(
        "gemini-test-model",
        api_keys=["key-a", "key-b"],
    )

    assert [client.init_kwargs["api_key"] for client in clients] == ["key-a", "key-b"]
    for _ in range(3):
        await backend.generate(
            [{"role": "user", "content": "question"}],
            {"temperature": 0.0},
        )
    assert len(clients[0].models.calls) == 2
    assert len(clients[1].models.calls) == 1


@pytest.mark.asyncio
async def test_quota_error_rotates_to_next_key(monkeypatch):
    clients = install_fake_genai_multi(monkeypatch)
    monkeypatch.setattr("officeqa_rl.backends.gemini.time.sleep", lambda _: None)
    backend = GeminiGenerationBackend(
        "gemini-test-model",
        api_keys=["key-a", "key-b"],
    )
    clients[0].models.quota_failures_remaining = 1

    turn = await backend.generate(
        [{"role": "user", "content": "question"}],
        {"temperature": 0.0},
    )

    assert turn.text == "plain Gemini response without action parsing"
    assert len(clients[0].models.calls) == 1  # 429 on the first key
    assert len(clients[1].models.calls) == 1  # retried on the second key


@pytest.mark.asyncio
async def test_503_overload_uses_capacity_budget_and_rotates(monkeypatch):
    clients = install_fake_genai_multi(monkeypatch)
    monkeypatch.setattr("officeqa_rl.backends.gemini.time.sleep", lambda _: None)
    # max_retries=0 kills genuinely-transient errors immediately; a 503 must
    # survive on the capacity budget and hop to the next key instead.
    backend = GeminiGenerationBackend(
        "gemini-test-model",
        max_retries=0,
        api_keys=["key-a", "key-b"],
    )
    clients[0].models.overload_failures_remaining = 1

    turn = await backend.generate(
        [{"role": "user", "content": "question"}],
        {"temperature": 0.0},
    )

    assert turn.text == "plain Gemini response without action parsing"
    assert len(clients[0].models.calls) == 1  # 503 on the first key
    assert len(clients[1].models.calls) == 1  # retried on the second key


def test_quota_retry_delay_honors_server_hint():
    hinted = Exception("quota exceeded {'retryDelay': '7s'}")
    delay = GeminiGenerationBackend._quota_retry_delay(hinted, attempt=1)
    assert 7.0 <= delay <= 7.0 * 1.25

    unhinted = Exception("quota exceeded")
    delay = GeminiGenerationBackend._quota_retry_delay(unhinted, attempt=3)
    assert 20.0 <= delay <= 20.0 * 1.25  # min(60, 5 * 2**2) plus jitter


@pytest.mark.api
@pytest.mark.asyncio
async def test_real_gemini_generation_smoke():
    model_name = os.environ.get("OFFICEQA_TEST_GEMINI_MODEL")
    if not model_name:
        pytest.skip("Set OFFICEQA_TEST_GEMINI_MODEL and GOOGLE_API_KEY")

    backend = GeminiGenerationBackend(model_name, max_tokens=16)
    turn = await backend.generate(
        [{"role": "user", "content": "Reply with the single word OK."}],
        {"temperature": 0.0},
    )

    assert turn.text.strip()
