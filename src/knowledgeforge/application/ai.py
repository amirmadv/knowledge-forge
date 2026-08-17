"""Application-level AI use cases for KnowledgeForge."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from knowledgeforge.application.agent_runtime import AgentRunResult, AgentRuntime
from knowledgeforge.application.retrieval import (
    ContextBuilder,
    HybridRetriever,
    RetrievalEvidence,
    SourceRef,
)
from knowledgeforge.application.semantic import SemanticRetriever
from knowledgeforge.application.tools import ToolAccess
from knowledgeforge.domain.graph import GraphService, NoteGraph
from knowledgeforge.domain.note import Note, NoteService
from knowledgeforge.infrastructure.ai.agent_planner import OpenAICompatibleAgentPlanner
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
    """Graph-aware AI agent for the local KnowledgeForge vault."""

    MAX_CONTEXT_NOTES = 8
    GRAPH_DEPTH = 1
    MAX_QUERY_TERMS = 8
    MIN_QUERY_TERM_LENGTH = 3
    MAX_HISTORY_TURNS = 8
    AGENT_SYSTEM_PROMPT = (
        "You are the KnowledgeForge personal knowledge agent. "
        "Use the user's local knowledge tools when you need facts from the vault. "
        "Do not invent facts that are not supported by tool results. "
        "When you have enough evidence, answer the user directly and concisely."
    )

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
        self._retriever = HybridRetriever(
            note_service=self._note_service,
            semantic_retriever=self._semantic_retriever,
        )
        self._context_builder = ContextBuilder(
            note_service=self._note_service,
            graph_service=self._graph_service,
            max_notes=self.MAX_CONTEXT_NOTES,
            graph_depth=self.GRAPH_DEPTH,
        )

    @property
    def note_service(self) -> NoteService:
        """Expose the configured note service to cooperating use cases."""
        return self._note_service

    @property
    def graph_service(self) -> GraphService:
        """Expose the configured graph service to cooperating use cases."""
        return self._graph_service

    @property
    def client(self) -> OpenAICompatibleClient:
        """Expose the configured AI client to cooperating use cases."""
        return self._client

    @property
    def tools(self):
        """Expose the default read-only provider-neutral core tool registry."""
        return self.tools_for_access(ToolAccess.READ_ONLY)

    def tools_for_access(self, access: ToolAccess):
        """Build the registry appropriate for an explicit agent access policy."""
        from knowledgeforge.application.tools import build_knowledge_tool_registry

        return build_knowledge_tool_registry(
            self,
            include_write_tools=access is ToolAccess.WRITE,
        )

    def runtime(self, access: ToolAccess = ToolAccess.READ_ONLY) -> AgentRuntime:
        """Build a bounded runtime with an explicit tool access policy."""
        from knowledgeforge.application.agent_runtime import AgentRuntimeConfig

        return AgentRuntime(
            self.tools_for_access(access),
            AgentRuntimeConfig(tool_access=access),
        )

    @property
    def planner(self) -> OpenAICompatibleAgentPlanner:
        """Expose the concrete OpenAI-compatible provider adapter."""
        return OpenAICompatibleAgentPlanner(
            self._client,
            system_prompt=self.AGENT_SYSTEM_PROMPT,
        )

    def run_agent(
        self,
        prompt: str,
        access: ToolAccess = ToolAccess.READ_ONLY,
    ) -> AgentRunResult:
        """Run the bounded tool-using agent under an explicit access policy."""
        return self.runtime(access).run(prompt, self.planner)

    def ask(
        self,
        question: str,
        history: tuple[tuple[str, str], ...] = (),
    ) -> AIAnswer:
        """Answer a question using hybrid retrieval and grounded context."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question cannot be empty.")

        notes = self._retrieve_context_notes(normalized_question)
        context, source_refs = self._context_builder.build_with_sources(notes)
        conversation = self._build_history(history)
        source_instructions = self._build_source_instructions(source_refs)

        prompt = (
            "Answer the user's question using the supplied KnowledgeForge "
            "knowledge first. The knowledge context contains note content "
            "and graph relationships from the user's vault. Do not invent "
            "facts that are not supported by the context. If the context is "
            "insufficient, state what is missing clearly.\n\n"
            "Grounding rules:\n"
            "- Treat the supplied vault context as the source of truth.\n"
            "- Cite factual claims with source markers such as [S1].\n"
            "- Use only source markers that exist in the context.\n"
            "- Do not fabricate citations or source titles.\n"
            "- When sources disagree, explicitly identify the conflict.\n\n"
            f"Available sources:\n{source_instructions}\n\n"
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
                    "maintain continuity; the vault remains the source of truth. "
                    "Ground factual claims in the supplied source markers."
                ),
            ),
            sources=tuple(note.title for note in notes),
        )

    def search_with_evidence(
        self,
        query: str,
        limit: int = 8,
    ) -> list[RetrievalEvidence]:
        """Search the vault and expose explainable retrieval evidence."""
        return self._retriever.search_with_evidence(query, limit=limit)

    def inspect_graph(self, title: str, depth: int = 1) -> NoteGraph:
        """Inspect the graph neighborhood of a note."""
        return self._graph_service.graph(title, depth=depth)

    def rebuild_semantic_index(self) -> int:
        """Rebuild the persisted semantic index for the local vault."""
        return self._semantic_retriever.rebuild()

    def _retrieve_context_notes(self, question: str) -> list[Note]:
        """Retrieve hybrid matches and expand them through the note graph."""
        matches = self._retriever.search(
            question,
            limit=self.MAX_CONTEXT_NOTES,
        )
        if not matches:
            return []

        all_notes = self._note_service.list_notes()
        notes_by_slug = {note.slug: note for note in all_notes}
        selected: dict[str, Note] = {
            match.note.slug: match.note for match in matches
        }

        for match in matches:
            graph = self._graph_service.graph(
                match.note.title,
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

    @staticmethod
    def _build_source_instructions(sources: tuple[SourceRef, ...]) -> str:
        """Describe source markers without exposing implementation details."""
        if not sources:
            return "(none)"
        return "\n".join(
            f"[{source.marker}] {source.title}" for source in sources
        )

    @classmethod
    def _query_terms(cls, question: str) -> list[str]:
        """Extract bounded, meaningful search terms for compatibility."""
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
