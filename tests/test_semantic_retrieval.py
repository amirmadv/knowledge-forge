"""Tests for semantic retrieval."""

from pathlib import Path

from knowledgeforge.application.commands import create_note, update_note
from knowledgeforge.application.semantic import SemanticRetriever, cosine_similarity
from knowledgeforge.domain.note import NoteService


class FakeEmbeddingClient:
    """Return deterministic vectors for semantic retrieval tests."""

    def embed(self, text: str) -> tuple[float, ...]:
        lowered = text.casefold()
        return (
            float("linear" in lowered),
            float("gradient" in lowered),
            float("database" in lowered),
        )


def test_cosine_similarity_ranks_matching_vectors() -> None:
    assert cosine_similarity((1.0, 0.0), (1.0, 0.0)) == 1.0
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == 0.0


def test_semantic_retriever_finds_related_note_without_keyword_match(
    tmp_path: Path,
) -> None:
    vault_path = tmp_path / "vault"
    create_note(title="Linear Regression", vault_path=vault_path)
    update_note(
        title="Linear Regression",
        content="A model learns coefficients for prediction.",
        vault_path=vault_path,
    )
    create_note(title="Gradient Descent", vault_path=vault_path)
    update_note(
        title="Gradient Descent",
        content="An optimizer follows the gradient to reduce loss.",
        vault_path=vault_path,
    )
    create_note(title="Database Indexing", vault_path=vault_path)

    retriever = SemanticRetriever(
        note_service=NoteService(vault_path),
        client=FakeEmbeddingClient(),
    )

    matches = retriever.search("optimization with gradient information")

    assert matches
    assert matches[0].note.title == "Gradient Descent"
    assert matches[0].score > 0.5
