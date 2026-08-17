"""Application-level AI use cases for KnowledgeForge."""

from __future__ import annotations

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


class KnowledgeAgent:
    """Graph-aware AI agent for the local KnowledgeForge vault."""

    MAX_CONTEXT_NOTES = 8
    GRAPH_DEPTH = 1

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

        notes = self._note_service.search(normalized_question)
        context = self._build_context(notes[: self.MAX_CONTEXT_NOTES])

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
            )
        )

    def inspect_graph(self, title: str, depth: int = 1) -> NoteGraph:
        """Inspect the graph neighborhood of a note."""
        return self._graph_service.graph(title, depth=depth)

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
