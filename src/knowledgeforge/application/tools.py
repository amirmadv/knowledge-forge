"""Tool contracts and built-in tools for the KnowledgeForge agent."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from knowledgeforge.application.retrieval import RetrievalEvidence
from knowledgeforge.domain.graph import GraphService, NoteGraph
from knowledgeforge.domain.note import Note, NoteService


class ToolAccess(StrEnum):
    """Authorization level required to expose or execute a tool."""

    READ_ONLY = "read_only"
    WRITE = "write"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Provider-neutral description of an agent tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    access: ToolAccess = ToolAccess.READ_ONLY


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Deterministic result returned by a tool execution."""

    tool_name: str
    data: dict[str, Any]


class KnowledgeTool(Protocol):
    """Protocol implemented by every agent tool."""

    @property
    def spec(self) -> ToolSpec: ...

    def execute(self, arguments: dict[str, Any]) -> ToolResult: ...


class ToolNotFoundError(KeyError):
    """Raised when an unknown tool is requested."""


class ToolArgumentError(ValueError):
    """Raised when a tool receives invalid arguments."""


class ToolAuthorizationError(PermissionError):
    """Raised when a tool requires more access than the current policy allows."""


class KnowledgeToolRegistry:
    """Registry that owns the tools exposed to an agent planner/provider."""

    def __init__(self, tools: tuple[KnowledgeTool, ...] = ()) -> None:
        self._tools: dict[str, KnowledgeTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: KnowledgeTool) -> None:
        """Register a tool by its stable public name."""
        name = tool.spec.name.strip()
        if not name:
            raise ValueError("Tool name cannot be empty.")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> KnowledgeTool:
        """Return a registered tool or raise a deterministic error."""
        normalized_name = name.strip()
        try:
            return self._tools[normalized_name]
        except KeyError as exc:
            raise ToolNotFoundError(
                f"Unknown KnowledgeForge tool: {normalized_name}"
            ) from exc

    def list_specs(self) -> tuple[ToolSpec, ...]:
        """Return tool specifications in stable registration order."""
        return tuple(tool.spec for tool in self._tools.values())

    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        access: ToolAccess = ToolAccess.READ_ONLY,
    ) -> ToolResult:
        """Execute one registered tool if its access level is authorized."""
        if arguments is not None and not isinstance(arguments, dict):
            raise ToolArgumentError("Tool arguments must be an object.")

        tool = self.get(name)
        _authorize(tool.spec, access)
        return tool.execute(arguments or {})

    def provider_tools(
        self,
        access: ToolAccess = ToolAccess.READ_ONLY,
    ) -> list[dict[str, Any]]:
        """Return provider declarations for tools allowed by the access policy."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_schema,
                },
            }
            for spec in self.list_specs()
            if _is_authorized(spec.access, access)
        ]


class SearchKnowledgeTool:
    """Search the vault and expose ranking evidence to an agent."""

    def __init__(self, search: Any) -> None:
        self._search = search

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_knowledge",
            description=(
                "Search the local vault using hybrid retrieval and return ranked evidence."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        query = _required_string(arguments, "query")
        limit = _optional_int(arguments, "limit", 8, minimum=1, maximum=20)
        evidence: list[RetrievalEvidence] = self._search(query, limit=limit)
        return ToolResult(
            tool_name=self.spec.name,
            data={"results": [_evidence_to_dict(item) for item in evidence]},
        )


class InspectNoteGraphTool:
    """Inspect a note's graph neighborhood."""

    def __init__(self, graph_service: GraphService) -> None:
        self._graph_service = graph_service

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="inspect_note_graph",
            description=(
                "Inspect incoming and outgoing relationships around a note."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Note title."},
                    "depth": {"type": "integer", "minimum": 0, "maximum": 5},
                },
                "required": ["title"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        title = _required_string(arguments, "title")
        depth = _optional_int(arguments, "depth", 1, minimum=0, maximum=5)
        graph: NoteGraph = self._graph_service.graph(title, depth=depth)
        return ToolResult(
            tool_name=self.spec.name,
            data={
                "nodes": [node.slug for node in graph.nodes],
                "edges": [
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "relation_type": edge.relation_type.value,
                    }
                    for edge in graph.edges
                ],
            },
        )


class GetNoteTool:
    """Read one note's metadata and complete Markdown content."""

    def __init__(self, note_service: NoteService) -> None:
        self._note_service = note_service

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="get_note",
            description="Read a note's metadata and complete Markdown content.",
            input_schema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Note title.",
                    }
                },
                "required": ["title"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        title = _required_string(arguments, "title")
        note = self._note_service.get(title)
        return ToolResult(
            tool_name=self.spec.name,
            data={"note": _note_to_dict(note)},
        )


class ListRelatedNotesTool:
    """List direct graph neighbors with their note metadata."""

    def __init__(
        self,
        note_service: NoteService,
        graph_service: GraphService,
    ) -> None:
        self._note_service = note_service
        self._graph_service = graph_service

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="list_related_notes",
            description=(
                "List notes directly connected to a selected note in either direction."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Note title.",
                    }
                },
                "required": ["title"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        title = _required_string(arguments, "title")
        slugs = self._graph_service.neighbors(title)
        notes = {note.slug: note for note in self._note_service.list_notes()}
        return ToolResult(
            tool_name=self.spec.name,
            data={
                "notes": [
                    _note_to_dict(notes[slug])
                    for slug in slugs
                    if slug in notes
                ]
            },
        )


class CreateNoteTool:
    """Create a note through an explicitly write-authorized agent capability."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="create_note",
            description=(
                "Create a new Markdown note in the local vault. This mutates the vault "
                "and is only available under an explicit write authorization policy."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "New note title."},
                    "content": {
                        "type": "string",
                        "description": "Complete Markdown content for the new note.",
                    },
                    "note_type": {"type": "string", "default": "concept"},
                    "status": {"type": "string", "default": "draft"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                },
                "required": ["title", "content"],
                "additionalProperties": False,
            },
            access=ToolAccess.WRITE,
        )

    def __init__(self, note_service: NoteService) -> None:
        self._note_service = note_service

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        title = _required_string(arguments, "title")
        content = _required_string(arguments, "content")
        note_type = arguments.get("note_type", "concept")
        status = arguments.get("status", "draft")
        tags = arguments.get("tags", [])

        if not isinstance(note_type, str) or not note_type.strip():
            raise ToolArgumentError("Tool argument 'note_type' must be a non-empty string.")
        if not isinstance(status, str) or not status.strip():
            raise ToolArgumentError("Tool argument 'status' must be a non-empty string.")
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ToolArgumentError("Tool argument 'tags' must be a list of strings.")

        note = self._note_service.create(
            title=title,
            note_type=note_type.strip(),
            status=status.strip(),
            tags=tuple(tag.strip() for tag in tags if tag.strip()),
        )
        note = self._note_service.update(title=title, content=content)

        return ToolResult(
            tool_name=self.spec.name,
            data={"note": _note_to_dict(note)},
        )


def build_knowledge_tool_registry(
    agent: Any,
    *,
    include_write_tools: bool = False,
) -> KnowledgeToolRegistry:
    """Build the core registry, optionally including explicitly write-capable tools."""
    tools: list[KnowledgeTool] = [
        SearchKnowledgeTool(agent.search_with_evidence),
        InspectNoteGraphTool(agent.graph_service),
        GetNoteTool(agent.note_service),
        ListRelatedNotesTool(agent.note_service, agent.graph_service),
    ]
    if include_write_tools:
        tools.append(CreateNoteTool(agent.note_service))
    return KnowledgeToolRegistry(tuple(tools))


def _is_authorized(required: ToolAccess, granted: ToolAccess) -> bool:
    if required is ToolAccess.READ_ONLY:
        return True
    return granted is ToolAccess.WRITE


def _authorize(spec: ToolSpec, granted: ToolAccess) -> None:
    if _is_authorized(spec.access, granted):
        return
    raise ToolAuthorizationError(
        f"Tool '{spec.name}' requires write authorization. "
        "The current agent policy is read-only."
    )


def _required_string(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ToolArgumentError(
            f"Tool argument '{name}' must be a non-empty string."
        )
    return value.strip()


def _optional_int(
    arguments: dict[str, Any],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolArgumentError(f"Tool argument '{name}' must be an integer.")
    if not minimum <= value <= maximum:
        raise ToolArgumentError(
            f"Tool argument '{name}' must be between {minimum} and {maximum}."
        )
    return value


def _note_to_dict(note: Note) -> dict[str, Any]:
    return {
        "title": note.title,
        "slug": note.slug,
        "filename": note.filename,
        "note_type": note.metadata.note_type,
        "status": note.metadata.status,
        "tags": list(note.metadata.tags),
        "created_at": note.created_at.isoformat(),
        "updated_at": note.updated_at.isoformat(),
        "content": note.path.read_text(encoding="utf-8"),
    }


def _evidence_to_dict(evidence: RetrievalEvidence) -> dict[str, Any]:
    return {
        "title": evidence.note.title,
        "slug": evidence.note.slug,
        "score": evidence.score,
        "semantic_score": evidence.semantic_score,
        "lexical_score": evidence.lexical_score,
        "metadata_score": evidence.metadata_score,
        "semantic_contribution": evidence.semantic_contribution,
        "lexical_contribution": evidence.lexical_contribution,
        "metadata_contribution": evidence.metadata_contribution,
        "reasons": list(evidence.reasons),
    }
