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


def test_client_rejects_empty_chat_completion_messages() -> None:
    client = OpenAICompatibleClient("http://localhost:1/v1", "key", "model")

    with pytest.raises(AIClientError, match="at least one message"):
        client.chat_completion([])


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

    assert client.embedding_model == "embed-model"
    assert client.embed("hello") == (1.0, 2.5, 3.0)


def test_client_chat_completion_sends_tools(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient("http://localhost:1/v1", "key", "model")

    result = client.chat_completion(
        [{"role": "user", "content": "Search."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge",
                    "description": "Search notes.",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert captured["url"] == "http://localhost:1/v1/chat/completions"
    assert captured["timeout"] == 60.0
    assert captured["payload"] == {
        "model": "model",
        "messages": [{"role": "user", "content": "Search."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge",
                    "description": "Search notes.",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "tool_choice": "auto",
    }
