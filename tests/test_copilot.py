"""Tests for the KnowledgeForge AI Copilot."""

from pathlib import Path

from knowledgeforge.application.ai import KnowledgeAgent
from knowledgeforge.application.copilot import KnowledgeCopilot
from knowledgeforge.domain.note import NoteService
from knowledgeforge.infrastructure.config.settings import Settings


class FakeAIClient:
    """Minimal deterministic AI client for Copilot tests."""

    def __init__(self, response: str = "mock response") -> None:
        self.response = response
        self.prompts: list[str] = []

    def chat(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return self.response

    def embed(self, text: str) -> tuple[float, ...]:
        return (1.0,)


def _copilot(tmp_path: Path, response: str = "mock response") -> tuple[KnowledgeCopilot, FakeAIClient, NoteService]:
    vault_path = tmp_path / "vault"
    note_service = NoteService(vault_path)
    client = FakeAIClient(response)
    agent = KnowledgeAgent(
        settings=Settings(vault_path=vault_path, ai_enabled=True),
        client=client,
        note_service=note_service,
    )
    return KnowledgeCopilot(agent), client, note_service


def test_summarize_uses_note_content(tmp_path: Path) -> None:
    copilot, client, note_service = _copilot(tmp_path, "summary")
    note_service.create("Linear Regression")
    note_service.update_content("Linear Regression", "Regression predicts a target.")

    result = copilot.summarize("Linear Regression")

    assert result.answer == "summary"
    assert result.sources == ("Linear Regression",)
    assert "Regression predicts a target." in client.prompts[0]


def test_suggest_tags_does_not_modify_note(tmp_path: Path) -> None:
    copilot, client, note_service = _copilot(tmp_path, "machine-learning, regression")
    note_service.create("Linear Regression")
    note_service.update_content("Linear Regression", "A regression model.")

    result = copilot.suggest_tags("Linear Regression")

    assert result.answer == "machine-learning, regression"
    assert note_service.read_content("Linear Regression").endswith("A regression model.\n")
    assert "comma-separated" in client.prompts[0]


def test_improve_updates_note_body(tmp_path: Path) -> None:
    copilot, _, note_service = _copilot(tmp_path, "Improved Markdown")
    note_service.create("Linear Regression")

    result = copilot.improve("Linear Regression")

    assert result.note is not None
    assert result.note.title == "Linear Regression"
    assert "Improved Markdown" in note_service.read_content("Linear Regression")


def test_create_note_generates_and_persists_body(tmp_path: Path) -> None:
    copilot, _, note_service = _copilot(tmp_path, "# Gradient Descent\n\nA method.")

    result = copilot.create_note(
        "Gradient Descent",
        "Create a short note explaining gradient descent.",
    )

    assert result.note is not None
    assert result.note.title == "Gradient Descent"
    assert "A method." in note_service.read_content("Gradient Descent")


def test_knowledge_gaps_includes_graph_context(tmp_path: Path) -> None:
    copilot, client, note_service = _copilot(tmp_path, "gaps")
    note_service.create("Linear Regression")
    note_service.create("Gradient Descent")

    from knowledgeforge.domain.graph import GraphService

    GraphService(tmp_path / "vault").add(
        "Linear Regression",
        "Gradient Descent",
        "prerequisite",
    )

    result = copilot.knowledge_gaps("Linear Regression")

    assert result.answer == "gaps"
    assert result.sources == ("Linear Regression",)
    assert "gradient-descent" in client.prompts[0]
