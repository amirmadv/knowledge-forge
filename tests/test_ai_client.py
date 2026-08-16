"""Tests for the KnowledgeForge AI client."""

from urllib import error

import pytest

from knowledgeforge.infrastructure.ai.client import AIClientError, OpenAICompatibleClient


def test_client_rejects_empty_prompt() -> None:
    client = OpenAICompatibleClient("http://localhost:1/v1", "key", "model")

    with pytest.raises(AIClientError, match="Prompt cannot be empty"):
        client.chat("  ")


def test_client_translates_provider_errors(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    client = OpenAICompatibleClient("http://localhost:1/v1", "key", "model")

    with pytest.raises(AIClientError, match="AI provider request failed"):
        client.chat("hello")
