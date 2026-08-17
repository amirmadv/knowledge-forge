"""Application-level AI use cases for KnowledgeForge."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from knowledgeforge.domain.graph import GraphService, NoteGraph
from knowledgeforge.domain.note import Note, NoteService
from knowledgeforge.infrastructure.ai.client import (
    AIClientError,
    OpenAICompatibleClient,
)
from knowledgeforge.infrastructure.config.settings import Settings


@dataclass(frozen=True, slots=True)
class AIAnswer:
    """Result returned by an AI application use case."""

    answer: str
    sources: tuple[str, ...] = ()


class KnowledgeAgent:
    """Graph-aware AI agent for the local KnowledgeForge vault.

    Retrieval is intentionally local-first: exact note matches are preferred,
    multi-word questions fall back to token search, and graph neighbors are
    added to the context before the provider is called. This gives the agent
    a lightweight RAG layer without requiring a vector database.
    """

    MAX_CONTEXT_NOTES = 8
    GRAPH_DEPTH = 1
    MAX_QUERY_TERMS = 8
    MIN_QUERY_TERM_LENGTH = 3

    def __init__(
        self,
        settings: Settings,
        vault_path: Path | None = None,
        client: OpenAICompatibleClient | None = None,
        note_service: NoteService | None = None,
        graph_service: GraphService | None = None,
    ) -> None:
        if not settings.ai_enabled:
            raise AIClientError(
                "AI is disabled. Set KNOWLEDGEFORGE_AI_ENABLED=true."
            )
        if not settings.ai_api_key and client is None:
            raise AIClientError("AI API key is not configured.")

        resolved_vault_path = vault_path or settings.vault_path

        self._client = client or OpenAICompatibleClient(
            base_url=settings.ai_base_url,
            api_key=settings.ai_api_key,
            model=settings.ai_model,
            timeout=settings.ai_timeout,
        )
        self._note_service = note_service or NoteService(resolved_vault_path)
        self._graph_service = graph_service or GraphService(resolved_vault_path)

    def ask(self, question: str) -> AIAnswer:
        """Answer a question using notes and their graph neighborhood."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question cannot be empty.")

        notes = self._retrieve_context_notes(normalized_question)
        context = self._build_context(notes)

        prompt = (
            "Answer the user's question using the supplied KnowledgeForge "
            "knowledge first. The knowledge context contains note content "
            "and graph relationships from the user's vault. Do not invent "
            "facts that are not supported by the context. If the context is "
            "insufficient, state what is missing clearly.\n\n"
            f"Question:\n{normalized_question}\n\n"
            f"Knowledge context:\n{context or '(no matching notes)'}"
        )

        return AIAnswer(
            answer=self._client.chat(
                prompt,
                system=(
                    "You are the KnowledgeForge personal knowledge agent. "
                    "Prefer the user's local knowledge over generic assumptions."
                ),
            ),
            sources=tuple(note.title for note in notes),
        )

    def inspect_graph(self, title: str, depth: int = 1) -> NoteGraph:
        """Inspect the graph neighborhood of a note."""
        return self._graph_service.graph(title, depth=depth)

    def _retrieve_context_notes(self, question: str) -> list[Note]:
        """Retrieve direct matches and expand them through the note graph."""
        seeds = self._search_question(question)
        if not seeds:
            return []

        all_notes = self._note_service.list_notes()
        notes_by_slug = {note.path.stem: note for note in all_notes}
        selected: dict[str, Note] = {}

        for note in seeds:
            selected[note.path.stem] = note

            graph = self._graph_service.graph(
                note.title,
                depth=self.GRAPH_DEPTH,
            )

            for node in graph.nodes:
                neighbor = notes_by_slug.get(node.slug)
                if neighbor is not None:
                    selected.setdefault(node.slug, neighbor)

                if len(selected) >= self.MAX_CONTEXT_NOTES:
                    break

            if len(selected) >= self.MAX_CONTEXT_NOTES:
                break

        return list(selected.values())[: self.MAX_CONTEXT_NOTES]

    def _search_question(self, question: str) -> list[Note]:
        """Search the question as a whole, then fall back to useful terms."""
        exact_matches = self._note_service.search(question)
        if exact_matches:
            return exact_matches[: self.MAX_CONTEXT_NOTES]

        terms = self._query_terms(question)
        matches: dict[str, Note] = {}

        for term in terms:
            for note in self._note_service.search(term):
                matches.setdefault(note.path.stem, note)

                if len(matches) >= self.MAX_CONTEXT_NOTES:
                    return list(matches.values())

        return list(matches.values())

    @classmethod
    def _query_terms(cls, question: str) -> list[str]:
        """Extract bounded, meaningful search terms from a question."""
        terms: list[str] = []
        seen: set[str] = set()

        for raw_term in re.findall(r"\w+", question.casefold(), flags=re.UNICODE):
            if len(raw_term) < cls.MIN_QUERY_TERM_LENGTH or raw_term in seen:
                continue

            seen.add(raw_term)
            terms.append(raw_term)

            if len(terms) >= cls.MAX_QUERY_TERMS:
                break

        return terms

    def _build_context(self, notes: list[Note]) -> str:
        """Build bounded textual context from notes and graph data."""
        sections: list[str] = []
        title_by_slug = {
            note.path.stem: note.title
            for note in self._note_service.list_notes()
        }

        for note in notes:
            content = self._note_service.read_content(note.title)
            graph = self._graph_service.graph(
                note.title,
                depth=self.GRAPH_DEPTH,
            )
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

            sections.append(
                f"# Note: {note.title}\n"
                f"Content:\n{content}\n"
                f"Graph nodes: {nodes or '(none)'}\n"
                f"Graph edges:\n{edges or '(none)'}"
            )

        return "\n\n---\n\n".join(sections)
