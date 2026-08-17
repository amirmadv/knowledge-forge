"""Semantic retrieval primitives for KnowledgeForge."""

from __future__ import annotations

import math
from dataclasses import dataclass

from knowledgeforge.domain.note import Note, NoteService


@dataclass(frozen=True, slots=True)
class SemanticMatch:
    """A note ranked by embedding similarity."""

    note: Note
    score: float


class SemanticRetriever:
    """Retrieve notes with an OpenAI-compatible embedding client."""

    def __init__(self, note_service: NoteService, client: object) -> None:
        self._note_service = note_service
        self._client = client
        self._cache: dict[str, tuple[float, ...]] = {}

    def search(self, query: str, limit: int = 5) -> list[SemanticMatch]:
        """Return the most semantically similar notes."""
        if not query.strip() or limit <= 0:
            return []

        embed = getattr(self._client, "embed", None)
        if not callable(embed):
            return []

        query_vector = tuple(embed(query))
        if not query_vector:
            return []

        matches: list[SemanticMatch] = []
        for note in self._note_service.list_notes():
            content = self._note_service.read_content(note.title)
            key = f"{note.path}:{note.updated_at.isoformat()}"
            vector = self._cache.get(key)
            if vector is None:
                vector = tuple(embed(f"{note.title}\n{content}"))
                if vector:
                    self._cache[key] = vector

            score = cosine_similarity(query_vector, vector)
            if score > 0:
                matches.append(SemanticMatch(note=note, score=score))

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:limit]


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Calculate cosine similarity for two equal-length vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
