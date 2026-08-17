"""Tests for the KnowledgeForge AI application layer."""

import json
from pathlib import Path

from knowledgeforge.application.ai import KnowledgeAgent
from knowledgeforge.application.commands import (
    add_note_relationship,
    create_note,
    update_note,
)
from knowledgeforge.infrastructure.config.settings import Settings


class FakeAIClient:
    """Capture chat requests without calling a real provider."""

    def __init__(self) -> None:
        self.prompt = ""
        self.system = ""
        self.completion_calls = 0

    def chat(self, prompt: str, system: str | None = None) -> str:
        self.prompt = prompt
        self.system = system or ""
        return "mock answer"

    def chat_completion(self, messages, *, tools=None):
        self.completion_calls += 1
        if self.completion_calls == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_note",
                                        "arguments": json.dumps({"title": "Linear Regression"}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Linear Regression is in the vault.",
                    }
                }
            ]
        }


def _settings(vault_path: Path) -> Settings:
    return Settings(
        vault_path=vault_path,
        ai_enabled=True,
        ai_api_key="test-key",
    )


def test_agent_uses_note_content_and_graph_context(tmp_path: Path) -> None:
    """Agent prompts should include local note content and graph edges."""
    vault_path = tmp_path / "vault"
    create_note(
        title="Linear Regression",
        vault_path=vault_path,
    )
    create_note(
        title="Gradient Descent",
        vault_path=vault_path,
    )
    add_note_relationship(
        source_title="Linear Regression",
        target_title="Gradient Descent",
        relation_type="prerequisite",
        vault_path=vault_path,
    )

    client = FakeAIClient()
    agent = KnowledgeAgent(
        settings=_settings(vault_path),
        client=client,
    )

    result = agent.ask("Linear Regression")

    assert result.answer == "mock answer"
    assert result.sources == ("Linear Regression", "Gradient Descent")
    assert "[S1] # Note: Linear Regression" in client.prompt
    assert "[S2] # Note: Gradient Descent" in client.prompt
    assert "[S1] Linear Regression" in client.prompt
    assert "[S2] Gradient Descent" in client.prompt
    assert "prerequisite" in client.prompt
    assert "Cite factual claims with source markers such as [S1]." in client.prompt
    assert "KnowledgeForge personal knowledge agent" in client.system


def test_agent_includes_graph_neighbor_note_content(tmp_path: Path) -> None:
    """RAG context should include content from directly connected notes."""
    vault_path = tmp_path / "vault"
    create_note(title="Linear Regression", vault_path=vault_path)
    create_note(title="Gradient Descent", vault_path=vault_path)
    update_note(
        title="Gradient Descent",
        content=(
            "Gradient descent updates model parameters by following the loss gradient."
        ),
        vault_path=vault_path,
    )
    add_note_relationship(
        source_title="Linear Regression",
        target_title="Gradient Descent",
        relation_type="prerequisite",
        vault_path=vault_path,
    )

    client = FakeAIClient()
    agent = KnowledgeAgent(
        settings=_settings(vault_path),
        client=client,
    )

    agent.ask("Explain Linear Regression")

    assert "# Note: Linear Regression" in client.prompt
    assert "# Note: Gradient Descent" in client.prompt
    assert "Gradient descent updates model parameters" in client.prompt


def test_agent_falls_back_to_query_terms(tmp_path: Path) -> None:
    """Natural-language questions should retrieve notes when exact search misses."""
    vault_path = tmp_path / "vault"
    create_note(title="Linear Regression", vault_path=vault_path)
    update_note(
        title="Linear Regression",
        content=(
            "Linear regression predicts a target using a weighted linear "
            "combination of features."
        ),
        vault_path=vault_path,
    )

    client = FakeAIClient()
    agent = KnowledgeAgent(
        settings=_settings(vault_path),
        client=client,
    )

    agent.ask("How does linear regression work?")

    assert "# Note: Linear Regression" in client.prompt
    assert "weighted linear combination" in client.prompt


def test_agent_preserves_conversation_history(tmp_path: Path) -> None:
    """Follow-up questions should include bounded prior conversation turns."""
    vault_path = tmp_path / "vault"
    create_note(title="Linear Regression", vault_path=vault_path)
    client = FakeAIClient()
    agent = KnowledgeAgent(
        settings=_settings(vault_path),
        client=client,
    )

    agent.ask("What is linear regression?")
    agent.ask(
        "What is its loss function?",
        history=(("What is linear regression?", "It predicts a target."),),
    )

    assert "User: What is linear regression?" in client.prompt
    assert "Assistant: It predicts a target." in client.prompt
    assert "What is its loss function?" in client.prompt


def test_agent_inspects_note_graph(tmp_path: Path) -> None:
    """Agent graph inspection should delegate to the domain graph service."""
    vault_path = tmp_path / "vault"
    create_note(title="A", vault_path=vault_path)
    create_note(title="B", vault_path=vault_path)
    add_note_relationship(
        source_title="A",
        target_title="B",
        relation_type="related",
        vault_path=vault_path,
    )

    agent = KnowledgeAgent(
        settings=_settings(vault_path),
        client=FakeAIClient(),
    )

    graph = agent.inspect_graph("A", depth=1)

    assert {node.slug for node in graph.nodes} == {"a", "b"}
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "a"
    assert graph.edges[0].target == "b"


def test_agent_search_exposes_retrieval_evidence(tmp_path: Path) -> None:
    """Application search should expose explainable lexical evidence."""
    vault_path = tmp_path / "vault"
    create_note(title="Linear Regression", vault_path=vault_path)
    update_note(
        title="Linear Regression",
        content="Linear regression predicts a target from input features.",
        vault_path=vault_path,
    )
    create_note(title="Python", vault_path=vault_path)

    agent = KnowledgeAgent(
        settings=_settings(vault_path),
        client=FakeAIClient(),
    )

    evidence = agent.search_with_evidence("linear regression", limit=2)

    assert evidence
    assert evidence[0].note.title == "Linear Regression"
    assert evidence[0].lexical_score > 0
    assert evidence[0].score > 0
    assert evidence[0].reasons


def test_agent_rejects_empty_question(tmp_path: Path) -> None:
    """Agent should reject an empty user question before provider calls."""
    agent = KnowledgeAgent(
        settings=_settings(tmp_path / "vault"),
        client=FakeAIClient(),
    )

    try:
        agent.ask("   ")
    except ValueError as exc:
        assert str(exc) == "Question cannot be empty."
    else:
        raise AssertionError("Expected ValueError")


def test_agent_runtime_uses_provider_adapter_and_core_tool(tmp_path: Path) -> None:
    """The application agent should execute a real core tool through the adapter."""
    vault_path = tmp_path / "vault"
    create_note(
        title="Linear Regression",
        content="A note about linear regression.",
        vault_path=vault_path,
    )

    client = FakeAIClient()
    agent = KnowledgeAgent(
        settings=_settings(vault_path),
        client=client,
    )

    result = agent.run_agent("Read the Linear Regression note.")

    assert result.answer == "Linear Regression is in the vault."
    assert result.trace.tool_call_count == 1
    assert result.trace.termination_reason == "final_answer"
    assert result.trace.steps[0].observations[0].success is True
    assert result.trace.steps[0].observations[0].tool_name == "get_note"
    assert client.completion_calls == 2
