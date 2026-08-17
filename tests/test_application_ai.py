"""Tests for the KnowledgeForge AI application layer."""

from pathlib import Path

from knowledgeforge.application.ai import KnowledgeAgent
from knowledgeforge.application.commands import add_note_relationship, create_note
from knowledgeforge.infrastructure.config.settings import Settings


class FakeAIClient:
    """Capture chat requests without calling a real provider."""

    def __init__(self) -> None:
        self.prompt = ""
        self.system = ""

    def chat(self, prompt: str, system: str | None = None) -> str:
        self.prompt = prompt
        self.system = system or ""
        return "mock answer"


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
    assert "# Note: Linear Regression" in client.prompt
    assert "Linear Regression" in client.prompt
    assert "Gradient Descent" in client.prompt
    assert "prerequisite" in client.prompt
    assert "KnowledgeForge personal knowledge agent" in client.system


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
        raise AssertionError("Expected ValueError for an empty question.")
