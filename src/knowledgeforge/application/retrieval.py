"""Hybrid retrieval and context construction for KnowledgeForge."""

from __future__ import annotations

import re
from dataclasses import dataclass

from knowledgeforge.application.semantic import SemanticMatch, SemanticRetriever
from knowledgeforge.domain.graph import GraphService
from knowledgeforge.domain.note import Note, NoteService
from knowledgeforge.infrastructure.ai.client import AIClientError


@dataclass(frozen=True, slots=True)
class RetrievalMatch:
    """A note ranked by the hybrid retrieval pipeline."""

    note: Note
    score: float
    lexical_score: float
    semantic_score: float
    metadata_score: float


class HybridRetriever:
    """Combine lexical, semantic, and metadata signals into one ranking."""

    SEMANTIC_WEIGHT = 0.65
    LEXICAL_WEIGHT = 0.25
    METADATA_WEIGHT = 0.10
    MIN_SEMANTIC_SCORE = 0.20
    MIN_LEXICAL_SCORE = 0.01

    def __init__(
        self,
        note_service: NoteService,
        semantic_retriever: SemanticRetriever,
    ) -> None:
        self._note_service = note_service
        self._semantic_retriever = semantic_retriever

    def search(self, query: str, limit: int = 8) -> list[RetrievalMatch]:
        """Return notes ranked by combined retrieval signals."""
        normalized_query = query.strip()
        if not normalized_query or limit <= 0:
            return []

        notes = self._note_service.list_notes()
        if not notes:
            return []

        terms = self._terms(normalized_query)
        semantic_by_slug = self._semantic_matches(normalized_query)
        matches: list[RetrievalMatch] = []

        for note in notes:
            content = self._note_service.read_content(note.title)
            lexical_score = self._lexical_score(
                normalized_query,
                terms,
                note,
                content,
            )
            semantic_score = semantic_by_slug.get(note.slug, 0.0)
            metadata_score = self._metadata_score(terms, note)

            if (
                lexical_score < self.MIN_LEXICAL_SCORE
                and semantic_score < self.MIN_SEMANTIC_SCORE
                and metadata_score == 0.0
            ):
                continue

            score = (
                self.SEMANTIC_WEIGHT * semantic_score
                + self.LEXICAL_WEIGHT * lexical_score
                + self.METADATA_WEIGHT * metadata_score
            )
            matches.append(
                RetrievalMatch(
                    note=note,
                    score=score,
                    lexical_score=lexical_score,
                    semantic_score=semantic_score,
                    metadata_score=metadata_score,
                )
            )

        matches.sort(
            key=lambda match: (
                match.score,
                match.lexical_score,
                match.semantic_score,
                match.note.title.casefold(),
            ),
            reverse=True,
        )
        return matches[:limit]

    def _semantic_matches(self, query: str) -> dict[str, float]:
        try:
            matches: list[SemanticMatch] = self._semantic_retriever.search(
                query,
                limit=max(8, len(self._note_service.list_notes())),
            )
        except AIClientError:
            return {}

        return {
            match.note.slug: max(0.0, min(1.0, match.score))
            for match in matches
        }

    @staticmethod
    def _terms(query: str) -> tuple[str, ...]:
        """Extract unique terms useful for lexical and metadata matching."""
        terms: list[str] = []
        seen: set[str] = set()
        for raw in re.findall(r"\w+", query.casefold(), flags=re.UNICODE):
            if len(raw) < 3 or raw in seen:
                continue
            seen.add(raw)
            terms.append(raw)
        return tuple(terms)

    @staticmethod
    def _lexical_score(
        query: str,
        terms: tuple[str, ...],
        note: Note,
        content: str,
    ) -> float:
        """Calculate a bounded lexical relevance score."""
        title = note.title.casefold()
        body = content.casefold()
        normalized_query = query.casefold()

        if normalized_query == title:
            return 1.0
        if normalized_query in title:
            return 0.90
        if normalized_query in body:
            return 0.70
        if not terms:
            return 0.0

        title_hits = sum(term in title for term in terms)
        body_hits = sum(term in body for term in terms)
        coverage = (title_hits * 1.5 + body_hits) / (len(terms) * 2.5)
        return min(0.65, coverage)

    @staticmethod
    def _metadata_score(terms: tuple[str, ...], note: Note) -> float:
        """Score metadata fields that directly match query terms."""
        if not terms:
            return 0.0

        metadata_values = {
            note.metadata.note_type.casefold(),
            note.metadata.status.casefold(),
            *(tag.casefold() for tag in note.metadata.tags),
        }
        hits = sum(
            any(term == value or term in value for value in metadata_values)
            for term in terms
        )
        return min(1.0, hits / len(terms))


class ContextBuilder:
    """Build bounded, grounded LLM context from retrieved notes and graph data."""

    def __init__(
        self,
        note_service: NoteService,
        graph_service: GraphService,
        max_notes: int = 8,
        max_chars_per_note: int = 6000,
        max_total_chars: int = 24000,
        graph_depth: int = 1,
    ) -> None:
        self._note_service = note_service
        self._graph_service = graph_service
        self._max_notes = max_notes
        self._max_chars_per_note = max_chars_per_note
        self._max_total_chars = max_total_chars
        self._graph_depth = graph_depth

    def build(self, notes: list[Note]) -> str:
        """Render note content and graph relationships within size limits."""
        sections: list[str] = []
        title_by_slug = {
            note.slug: note.title for note in self._note_service.list_notes()
        }
        total_chars = 0

        for note in notes[: self._max_notes]:
            content = self._note_service.read_content(note.title)
            content = content[: self._max_chars_per_note]
            graph = self._graph_service.graph(note.title, depth=self._graph_depth)

            edges = "\n".join(
                "- "
                f"{title_by_slug.get(edge.source, edge.source)} "
                f"--[{edge.relation_type.value}]--> "
                f"{title_by_slug.get(edge.target, edge.target)}"
                for edge in graph.edges
            )
            nodes = ", ".join(
                title_by_slug.get(node.slug, node.slug)
                for node in graph.nodes
            )

            section = (
                f"# Note: {note.title}\n"
                f"Content:\n{content}\n"
                f"Graph nodes: {nodes or '(none)'}\n"
                f"Graph edges:\n{edges or '(none)'}"
            )

            remaining = self._max_total_chars - total_chars
            if remaining <= 0:
                break
            if len(section) > remaining:
                section = section[:remaining]

            sections.append(section)
            total_chars += len(section)

        return "\n\n---\n\n".join(sections)
