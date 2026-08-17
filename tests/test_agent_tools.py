"""Tests for the KnowledgeForge agent tool architecture."""

from pathlib import Path

import pytest

from knowledgeforge.application.ai import KnowledgeAgent
from knowledgeforge.application.commands import (
    add_note_relationship,
    create_note,
    update_note,
)
from knowledgeforge.application.tools import (
    KnowledgeToolRegistry,
    ToolArgumentError,
    ToolNotFoundError,
)
from knowledgeforge.infrastructure.config.settings import Settings


class FakeAIClient:
    """Provide deterministic AI behavior for tool registry tests."""

    def chat(self, prompt: str, system: str | None = None) -> str:
        return "mock answer"

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def _agent(vault_path: Path) -> KnowledgeAgent:
    return KnowledgeAgent(
        settings=Settings(
            vault_path=vault_path,
            ai_enabled=True,
            ai_api_key="test-key",
        ),
        client=FakeAIClient(),
    )


def test_agent_exposes_stable_core_tool_specs(tmp_path: Path) -> None:
    """The agent should expose the four core tools with provider schemas."""
    agent = _agent(tmp_path / "vault")

    specs = agent.tools.list_specs()

    assert [spec.name for spec in specs] == [
        "search_knowledge",
        "inspect_note_graph",
        "get_note",
        "list_related_notes",
    ]
    assert all(spec.input_schema["type"] == "object" for spec in specs)
    assert [item["function"]["name"] for item in agent.tools.provider_tools()] == [
        "search_knowledge",
        "inspect_note_graph",
        "get_note",
        "list_related_notes",
    ]


def test_search_tool_returns_explainable_results(tmp_path: Path) -> None:
    """Search tool output should preserve retrieval evidence."""
    vault_path = tmp_path / "vault"
    create_note("Linear Regression", vault_path=vault_path)
    update_note(
        "Linear Regression",
        "Linear regression predicts a target from input features.",
        vault_path=vault_path,
    )

    result = _agent(vault_path).tools.execute(
        "search_knowledge",
        {"query": "linear regression", "limit": 1},
    )

    assert result.tool_name == "search_knowledge"
    assert result.data["results"][0]["title"] == "Linear Regression"
    assert result.data["results"][0]["score"] > 0
    assert result.data["results"][0]["reasons"]


def test_graph_and_related_tools_return_connected_notes(tmp_path: Path) -> None:
    """Graph tools should expose both structure and note metadata."""
    vault_path = tmp_path / "vault"
    create_note("Linear Regression", vault_path=vault_path)
    create_note("Gradient Descent", vault_path=vault_path)
    add_note_relationship(
        "Linear Regression",
        "Gradient Descent",
        "prerequisite",
        vault_path=vault_path,
    )

    registry = _agent(vault_path).tools
    graph = registry.execute(
        "inspect_note_graph",
        {"title": "Linear Regression", "depth": 1},
    )
    related = registry.execute(
        "list_related_notes",
        {"title": "Linear Regression"},
    )

    assert graph.data["nodes"] == ["gradient-descent", "linear-regression"]
    assert graph.data["edges"] == [
        {
            "source": "linear-regression",
            "target": "gradient-descent",
            "relation_type": "prerequisite",
        }
    ]
    assert related.data["notes"][0]["title"] == "Gradient Descent"


def test_get_note_tool_returns_metadata_and_content(tmp_path: Path) -> None:
    """The note tool should give an agent enough context to reason locally."""
    vault_path = tmp_path / "vault"
    create_note(
        "Gradient Descent",
        vault_path=vault_path,
        note_type="concept",
        status="active",
        tags=("machine-learning", "optimization"),
    )
    update_note(
        "Gradient Descent",
        "Gradient descent minimizes a loss function iteratively.",
        vault_path=vault_path,
    )

    result = _agent(vault_path).tools.execute(
        "get_note",
        {"title": "Gradient Descent"},
    )
    note = result.data["note"]

    assert note["title"] == "Gradient Descent"
    assert note["status"] == "active"
    assert note["tags"] == ["machine-learning", "optimization"]
    assert "Gradient descent minimizes" in note["content"]


def test_registry_rejects_unknown_tools_and_invalid_arguments(tmp_path: Path) -> None:
    """Tool execution errors should be deterministic and provider-independent."""
    registry = _agent(tmp_path / "vault").tools

    with pytest.raises(ToolNotFoundError, match="Unknown KnowledgeForge tool"):
        registry.execute("does_not_exist", {})

    with pytest.raises(ToolArgumentError, match="non-empty string"):
        registry.execute("get_note", {"title": "   "})

    with pytest.raises(ToolArgumentError, match="between 0 and 5"):
        registry.execute(
            "inspect_note_graph",
            {"title": "A", "depth": 6},
        )


def test_registry_rejects_duplicate_tool_names() -> None:
    """A registry must not silently replace an existing tool."""
    registry = KnowledgeToolRegistry()

    class FakeTool:
        @property
        def spec(self):
            from knowledgeforge.application.tools import ToolSpec

            return ToolSpec("duplicate", "first", {"type": "object"})

        def execute(self, arguments):
            raise AssertionError

    registry.register(FakeTool())
    with pytest.raises(ValueError, match="Tool already registered"):
        registry.register(FakeTool())
