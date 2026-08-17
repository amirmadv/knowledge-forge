"""Tests for hybrid retrieval and bounded context construction."""

from pathlib import Path

from knowledgeforge.application.commands import (
    add_note_relationship,
    create_note,
    update_note,
)
from knowledgeforge.application.retrieval import ContextBuilder, HybridRetriever
from knowledgeforge.application.semantic import SemanticMatch
from knowledgeforge.domain.graph import GraphService
from knowledgeforge.domain.note import NoteService


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


def test_hybrid_retriever_exposes_score_breakdown_and_reasons(
    tmp_path: Path,
) -> None:
    """Explainable retrieval should expose raw and weighted signals."""
    vault_path = tmp_path / "vault"
    create_note(
        title="Gradient Descent",
        vault_path=vault_path,
        tags=("optimization",),
    )
    update_note(
        title="Gradient Descent",
        content="Gradient descent minimizes loss.",
        vault_path=vault_path,
    )
    note_service, _ = _services(vault_path)
    note = note_service.list_notes()[0]
    retriever = HybridRetriever(
        note_service=note_service,
        semantic_retriever=FakeSemanticRetriever(
            [SemanticMatch(note=note, score=0.8)]
        ),
    )

    evidence = retriever.search_with_evidence("gradient loss optimization", limit=1)[0]

    assert evidence.note.title == "Gradient Descent"
    assert evidence.semantic_score == 0.8
    assert evidence.semantic_contribution == 0.65 * 0.8
    assert evidence.lexical_contribution == 0.25 * evidence.lexical_score
    assert evidence.metadata_contribution == 0.10 * evidence.metadata_score
    assert evidence.score == (
        evidence.semantic_contribution
        + evidence.lexical_contribution
        + evidence.metadata_contribution
    )
    assert "semantic similarity" in evidence.reasons
    assert "title/body lexical match" in evidence.reasons
    assert "metadata/tag match" in evidence.reasons


def test_hybrid_retriever_evidence_order_is_deterministic(
    tmp_path: Path,
) -> None:
    """Equal scores should use a stable, human-readable title tie-breaker."""
    vault_path = tmp_path / "vault"
    create_note(title="Alpha Note", vault_path=vault_path)
    create_note(title="Beta Note", vault_path=vault_path)
    note_service, _ = _services(vault_path)
    notes = note_service.list_notes()
    retriever = HybridRetriever(
        note_service=note_service,
        semantic_retriever=FakeSemanticRetriever(
            [
                SemanticMatch(note=notes[0], score=0.5),
                SemanticMatch(note=notes[1], score=0.5),
            ]
        ),
    )

    first = retriever.search_with_evidence("anything", limit=2)
    second = retriever.search_with_evidence("anything", limit=2)

    assert [item.note.title for item in first] == ["Alpha Note", "Beta Note"]
    assert [item.note.title for item in first] == [item.note.title for item in second]


def test_hybrid_retriever_evidence_excludes_zero_signal_notes(tmp_path: Path) -> None:
    """Explainable retrieval must preserve the existing zero-signal filter."""
    vault_path = tmp_path / "vault"
    create_note(title="Relevant Note", vault_path=vault_path)
    create_note(title="Unrelated Note", vault_path=vault_path)
    note_service, _ = _services(vault_path)
    notes = note_service.list_notes()
    retriever = HybridRetriever(
        note_service=note_service,
        semantic_retriever=FakeSemanticRetriever(
            [SemanticMatch(note=notes[0], score=0.9)]
        ),
    )

    evidence = retriever.search_with_evidence("topic", limit=8)

    assert [item.note.title for item in evidence] == ["Relevant Note"]


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
    assert context.startswith("[S1] # Note: Linear Regression")


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


def test_context_builder_returns_stable_source_references(tmp_path: Path) -> None:
    """Each rendered note should receive a stable prompt citation marker."""
    vault_path = tmp_path / "vault"
    create_note(title="Linear Regression", vault_path=vault_path)
    create_note(title="Gradient Descent", vault_path=vault_path)
    note_service, graph_service = _services(vault_path)
    builder = ContextBuilder(note_service, graph_service)

    context, sources = builder.build_with_sources(note_service.list_notes())

    assert "[S1] # Note: Gradient Descent" in context
    assert "[S2] # Note: Linear Regression" in context
    assert [(source.marker, source.title) for source in sources] == [
        ("S1", "Gradient Descent"),
        ("S2", "Linear Regression"),
    ]
