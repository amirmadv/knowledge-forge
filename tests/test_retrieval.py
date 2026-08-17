"""Tests for hybrid retrieval and bounded context construction."""

from pathlib import Path

from knowledgeforge.application.retrieval import ContextBuilder, HybridRetriever
from knowledgeforge.application.semantic import SemanticMatch
from knowledgeforge.domain.graph import GraphService
from knowledgeforge.domain.note import NoteService
from knowledgeforge.domain.relationship import RelationshipService
from knowledgeforge.application.commands import add_note_relationship, create_note, update_note


class FakeSemanticRetriever:
    """Return deterministic semantic scores for hybrid ranking tests."""

    def __init__(self, matches: list[SemanticMatch]) -> None:
        self._matches = matches

    def search(self, query: str, limit: int = 5) -> list[SemanticMatch]:
        return self._matches[:limit]


def _services(vault_path: Path) -> tuple[NoteService, GraphService]:
    return NoteService(vault_path), GraphService(vault_path)


def test_hybrid_retriever_combines_semantic_and_lexical_scores(
    tmp_path: Path,
) -> None:
    """Strong semantic and lexical signals should produce a ranked match."""
    vault_path = tmp_path / "vault"
    create_note(title="Gradient Descent", vault_path=vault_path)
    update_note(
        title="Gradient Descent",
        content="Gradient descent minimizes loss by updating model parameters.",
        vault_path=vault_path,
    )
    note_service, _ = _services(vault_path)
    note = note_service.list_notes()[0]

    retriever = HybridRetriever(
        note_service=note_service,
        semantic_retriever=FakeSemanticRetriever(
            [SemanticMatch(note=note, score=0.9)]
        ),
    )

    matches = retriever.search("loss gradient", limit=1)

    assert len(matches) == 1
    assert matches[0].note.title == "Gradient Descent"
    assert matches[0].semantic_score == 0.9
    assert matches[0].lexical_score > 0
    assert matches[0].score > 0.5


def test_hybrid_retriever_uses_metadata_as_a_signal(tmp_path: Path) -> None:
    """Tags and lifecycle metadata should contribute to ranking."""
    vault_path = tmp_path / "vault"
    create_note(
        title="Machine Learning",
        vault_path=vault_path,
        tags=("machine-learning", "ai"),
        note_type="research",
    )
    note_service, _ = _services(vault_path)

    retriever = HybridRetriever(
        note_service=note_service,
        semantic_retriever=FakeSemanticRetriever([]),
    )

    matches = retriever.search("machine learning research", limit=1)

    assert len(matches) == 1
    assert matches[0].note.title == "Machine Learning"
    assert matches[0].metadata_score > 0


def test_hybrid_retriever_returns_empty_for_invalid_query(tmp_path: Path) -> None:
    """Blank queries should not scan or rank the vault."""
    vault_path = tmp_path / "vault"
    create_note(title="Any Note", vault_path=vault_path)
    note_service, _ = _services(vault_path)

    retriever = HybridRetriever(
        note_service=note_service,
        semantic_retriever=FakeSemanticRetriever([]),
    )

    assert retriever.search("   ") == []
    assert retriever.search("anything", limit=0) == []


def test_context_builder_respects_total_character_limit(tmp_path: Path) -> None:
    """Context construction must remain bounded for LLM prompt safety."""
    vault_path = tmp_path / "vault"
    create_note(title="Linear Regression", vault_path=vault_path)
    update_note(
        title="Linear Regression",
        content="A" * 5000,
        vault_path=vault_path,
    )
    note_service, graph_service = _services(vault_path)
    builder = ContextBuilder(
        note_service=note_service,
        graph_service=graph_service,
        max_total_chars=400,
        max_chars_per_note=5000,
    )

    context = builder.build(note_service.list_notes())

    assert len(context) <= 400
    assert context.startswith("# Note: Linear Regression")


def test_context_builder_includes_graph_relationships(tmp_path: Path) -> None:
    """Context should preserve grounded graph relationships between notes."""
    vault_path = tmp_path / "vault"
    create_note(title="Linear Regression", vault_path=vault_path)
    create_note(title="Gradient Descent", vault_path=vault_path)
    update_note(
        title="Gradient Descent",
        content="Gradient descent updates model parameters.",
        vault_path=vault_path,
    )
    add_note_relationship(
        source_title="Linear Regression",
        target_title="Gradient Descent",
        relation_type="prerequisite",
        vault_path=vault_path,
    )
    note_service, graph_service = _services(vault_path)
    builder = ContextBuilder(note_service, graph_service)

    context = builder.build([note_service.list_notes()[0]])

    assert "Linear Regression" in context
    assert "Gradient Descent" in context
    assert "prerequisite" in context
