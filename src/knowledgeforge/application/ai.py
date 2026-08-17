"""Application-level AI use cases for KnowledgeForge."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from knowledgeforge.application.semantic import SemanticRetriever
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

    Retrieval is local-first: exact and keyword matches are combined with
    optional semantic embeddings, then graph neighbors are added before the
    provider is called. If embeddings are unavailable, lexical retrieval still
    keeps the agent fully usable.
    """

    MAX_CONTEXT_NOTES = 8
    GRAPH_DEPTH = 1
    MAX_QUERY_TERMS = 8
    MIN_QUERY_TERM_LENGTH = 3
    MAX_HISTORY_TURNS = 8
    SEMANTIC_LIMIT = 5
    SEMANTIC_MIN_SCORE = 0.20

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
            embedding_model=settings.ai_embedding_model,
        )
        self._note_service = note_service or NoteService(resolved_vault_path)
        self._graph_service = graph_service or GraphService(resolved_vault_path)
        self._semantic_retriever = SemanticRetriever(
            note_service=self._note_service,
            client=self._client,
        )

    @property
    def note_service(self) -> NoteService:
        """Expose the configured note service to cooperating use cases."""
        return self._note_service

    @property
    def client(self) -> OpenAICompatibleClient:
        """Expose the configured AI client to cooperating use cases."""
        return self._client

    def ask(
        self,
        question: str,
        history: tuple[tuple[str, str], ...] = (),
    ) -> AIAnswer:
        """Answer a question using notes, graph context, and optional history."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question cannot be empty.")

        notes = self._retrieve_context_notes(normalized_question)
        context = self._build_context(notes)
        conversation = self._build_history(history)

        prompt = (
            "Answer the user's question using the supplied KnowledgeForge "
            "knowledge first. The knowledge context contains note content "
            "and graph relationships from the user's vault. Do not invent "
            "facts that are not supported by the context. If the context is "
            "insufficient, state what is missing clearly.\n\n"
            f"Conversation history:\n{conversation}\n\n"
            f"Question:\n{normalized_question}\n\n"
            f"Knowledge context:\n{context or '(no matching notes)'}"
        )

        return AIAnswer(
            answer=self._client.chat(
                prompt,
                system=(
                    "You are the KnowledgeForge personal knowledge agent. "
                    "Prefer the user's local knowledge over generic assumptions. "
                    "Use conversation history only to resolve references and "
                    "maintain continuity; the vault remains the source of truth."
                ),
            ),
            sources=tuple(note.title for note in notes),
        )

    def inspect_graph(self, title: str, depth: int = 1) -> NoteGraph:
        """Inspect the graph neighborhood of a note."""
        return self._graph_service.graph(title, depth=depth)

    def _retrieve_context_notes(self, question: str) -> list[Note]:
        """Retrieve ranked matches and expand them through the note graph."""
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
        """Use exact, semantic, and lexical retrieval in that order."""
        exact_matches = self._note_service.search(question)
        if exact_matches:
            return exact_matches[: self.MAX_CONTEXT_NOTES]

        matches: dict[str, Note] = {}

        try:
            semantic_matches = self._semantic_retriever.search(
                question,
                limit=self.SEMANTIC_LIMIT,
            )
        except AIClientError:
            semantic_matches = []

        for match in semantic_matches:
            if match.score >= self.SEMANTIC_MIN_SCORE:
                matches.setdefault(match.note.path.stem, match.note)

        for term in self._query_terms(question):
            for note in self._note_service.search(term):
                matches.setdefault(note.path.stem, note)

                if len(matches) >= self.MAX_CONTEXT_NOTES:
                    return list(matches.values())

        return list(matches.values())[: self.MAX_CONTEXT_NOTES]

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

    @classmethod
    def _build_history(cls, history: tuple[tuple[str, str], ...]) -> str:
        """Format bounded conversation history for the provider prompt."""
        if not history:
            return "(none)"

        bounded_history = history[-cls.MAX_HISTORY_TURNS :]
        return "\n\n".join(
            f"User: {question.strip()}\nAssistant: {answer.strip()}"
            for question, answer in bounded_history
        )

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
