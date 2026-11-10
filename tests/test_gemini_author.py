"""Unit tests for Gemini authoring and API-key rotation (no network)."""

import sys
from types import SimpleNamespace

import pytest

from officeqa_rl.gemini_author import GeminiAuthor, gemini_api_keys_from_env


class FakeCapacityError(Exception):
    def __init__(self):
        super().__init__("429 RESOURCE_EXHAUSTED")
        self.code = 429
        self.status = "RESOURCE_EXHAUSTED"


class FakeModels:
    def __init__(self, client):
        self.client = client
        self.failures = []

    def generate_content(self, **kwargs):
        self.client.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        return SimpleNamespace(text="authored JSON", candidates=[])


class FakeClient:
    def __init__(self, api_key=None, **kwargs):
        self.api_key = api_key
        self.kwargs = kwargs
        self.calls = []
        self.models = FakeModels(self)


def install_fake_genai(monkeypatch):
    clients = []

    def client_factory(**kwargs):
        client = FakeClient(**kwargs)
        clients.append(client)
        return client

    fake_genai = SimpleNamespace(Client=client_factory)
    fake_types = SimpleNamespace(
        HttpOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    return clients


def test_round_robins_across_api_keys(monkeypatch):
    clients = install_fake_genai(monkeypatch)
    author = GeminiAuthor(api_keys=["key-a", "key-b"])

    assert author.generate("first") == "authored JSON"
    assert author.generate("second") == "authored JSON"

    assert [client.api_key for client in clients] == ["key-a", "key-b"]
    assert [len(client.calls) for client in clients] == [1, 1]


def test_rotates_on_capacity_error(monkeypatch):
    clients = install_fake_genai(monkeypatch)
    author = GeminiAuthor(api_keys=["key-a", "key-b"])
    clients[0].models.failures.append(FakeCapacityError())
    monkeypatch.setattr("officeqa_rl.gemini_author.time.sleep", lambda _: None)

    assert author.generate("question") == "authored JSON"

    assert len(clients[0].calls) == 1
    assert len(clients[1].calls) == 1


def test_key_pool_from_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEYS_FILE", raising=False)
    monkeypatch.setenv("GEMINI_API_KEYS", "key-a, key-b\nkey-c")
    assert gemini_api_keys_from_env() == ["key-a", "key-b", "key-c"]

    keys_file = tmp_path / "gemini-keys.txt"
    keys_file.write_text("# projects\nkey-1\n\n key-2 \n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEYS_FILE", str(keys_file))
    assert gemini_api_keys_from_env() == ["key-1", "key-2"]

    keys_file.write_text("# no keys\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="contains no keys"):
        gemini_api_keys_from_env()
