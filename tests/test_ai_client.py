"""Tests for the KnowledgeForge AI client."""

import json
from urllib import error

import pytest

from knowledgeforge.infrastructure.ai.client import AIClientError, OpenAICompatibleClient


def test_client_rejects_empty_prompt() -> None:
    client = OpenAICompatibleClient("http://localhost:1/v1", "key", "model")

    with pytest.raises(AIClientError, match="Prompt cannot be empty"):
        client.chat("  ")


def test_client_rejects_empty_embedding_text() -> None:
    client = OpenAICompatibleClient("http://localhost:1/v1", "key", "model")

    with pytest.raises(AIClientError, match="Text to embed cannot be empty"):
        client.embed("  ")


def test_client_translates_provider_errors(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    client = OpenAICompatibleClient("http://localhost:1/v1", "key", "model")

    with pytest.raises(AIClientError, match="AI provider request failed"):
        client.chat("hello")


def test_client_parses_embedding_response(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"data": [{"embedding": [1, 2.5, 3]}]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResponse())
    client = OpenAICompatibleClient(
        "http://localhost:1/v1",
        "key",
        "model",
        embedding_model="embed-model",
    )

    assert client.embed("hello") == (1.0, 2.5, 3.0)
