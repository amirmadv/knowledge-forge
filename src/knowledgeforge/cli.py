"""Command-line interface for KnowledgeForge."""

from __future__ import annotations

from typing import Annotated

import typer

from knowledgeforge.application.commands import (
    create_note,
    create_template,
    delete_note,
    get_graph_statistics,
    get_note_ancestors,
    get_note_descendants,
    get_note_graph,
    initialize_knowledgeforge,
    list_notes,
    list_templates,
    search_notes,
    show_note,
    show_template,
    update_note,
)
from knowledgeforge.domain.note import (
    NoteAlreadyExistsError,
    NoteNotFoundError,
)
from knowledgeforge.domain.template import (
    InvalidTemplateNameError,
    TemplateNotFoundError,
)

app = typer.Typer(
    name="knowledgeforge",
    help="AI-powered personal knowledge management system.",
    no_args_is_help=True,
)

note_app = typer.Typer(
    name="note",
    help="Create and manage KnowledgeForge notes.",
    no_args_is_help=True,
)

template_app = typer.Typer(
    name="template",
    help="Create and manage KnowledgeForge templates.",
    no_args_is_help=True,
)

app.add_typer(note_app, name="note")
app.add_typer(template_app, name="template")


@app.callback()
def main() -> None:
    """Manage the KnowledgeForge application."""


@app.command("init")
def init_command() -> None:
    """Initialize a KnowledgeForge vault."""
    result = initialize_knowledgeforge()

    typer.echo(result.message)
    typer.echo(f"Vault: {result.vault_path}")


@note_app.command("create")
def create_note_command(
    title: str,
    template: Annotated[
        str,
        typer.Option(
            "--template",
            "-t",
            help="Template name used to create the note.",
        ),
    ] = "default",
    note_type: Annotated[
        str,
        typer.Option(
            "--type",
            help="Type of the note.",
        ),
    ] = "concept",
    status: Annotated[
        str,
        typer.Option(
            "--status",
            help="Lifecycle status of the note.",
        ),
    ] = "draft",
    tags: Annotated[
        list[str] | None,
        typer.Option(
            "--tag",
            help="Tag assigned to the note. Repeat --tag for multiple tags.",
        ),
    ] = None,
) -> None:
    """Create a new Markdown note."""
    try:
        result = create_note(
            title=title,
            template=template,
            note_type=note_type,
            status=status,
            tags=tuple(tags or ()),
        )
    except (
        NoteAlreadyExistsError,
        NoteNotFoundError,
        ValueError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(result.message)
    typer.echo(f"Path: {result.note.path}")


@note_app.command("list")
def list_notes_command(
    note_type: Annotated[
        str | None,
        typer.Option(
            "--type",
            help="Filter notes by note type.",
        ),
    ] = None,
    status: Annotated[
        str | None,
        typer.Option(
            "--status",
            help="Filter notes by lifecycle status.",
        ),
    ] = None,
    tag: Annotated[
        str | None,
        typer.Option(
            "--tag",
            help="Filter notes by tag.",
        ),
    ] = None,
) -> None:
    """List Markdown notes with optional metadata filters."""
    notes = list_notes(
        note_type=note_type,
        status=status,
        tag=tag,
    )

    typer.echo("KnowledgeForge Notes")

    if not notes:
        typer.echo("\nNo notes found.")
        return

    for index, note in enumerate(notes, start=1):
        typer.echo(f"\n{index}. {note.title}")
        typer.echo(f"   {note.path}")
        typer.echo(f"   Type: {note.metadata.note_type}")
        typer.echo(f"   Status: {note.metadata.status}")

        if note.metadata.tags:
            typer.echo(
                f"   Tags: {', '.join(note.metadata.tags)}"
            )


@note_app.command("search")
def search_notes_command(query: str) -> None:
    """Search notes by title, content, and metadata."""
    try:
        notes = search_notes(query)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Search results for: {query}")

    if not notes:
        typer.echo("\nNo matching notes found.")
        return

    for index, note in enumerate(notes, start=1):
        typer.echo(f"\n{index}. {note.title}")
        typer.echo(f"   {note.path}")


@note_app.command("show")
def show_note_command(title: str) -> None:
    """Show a note and its Markdown content."""
    try:
        note, content = show_note(title)
    except (
        NoteNotFoundError,
        ValueError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Title: {note.title}")
    typer.echo(f"Path: {note.path}")
    typer.echo(f"Slug: {note.slug}")
    typer.echo(f"Type: {note.metadata.note_type}")
    typer.echo(f"Status: {note.metadata.status}")

    if note.metadata.tags:
        typer.echo(
            f"Tags: {', '.join(note.metadata.tags)}"
        )

    typer.echo("\n" + content)


@note_app.command("edit")
def edit_note_command(
    title: str,
    content: Annotated[
        str,
        typer.Option(
            "--content",
            "-c",
            help="New Markdown content for the note.",
        ),
    ],
) -> None:
    """Replace the content of an existing note."""
    try:
        result = update_note(
            title,
            content,
        )
    except (
        NoteNotFoundError,
        ValueError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(result.message)
    typer.echo(f"Path: {result.note.path}")


@note_app.command("delete")
def delete_note_command(title: str) -> None:
    """Delete an existing Markdown note."""
    try:
        result = delete_note(title)
    except (
        NoteNotFoundError,
        ValueError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(result.message)
    typer.echo(f"Path: {result.note.path}")


@note_app.command("graph")
def note_graph_command(
    title: str,
    depth: Annotated[
        int,
        typer.Option(
            "--depth",
            "-d",
            help="Maximum graph traversal depth.",
        ),
    ] = 1,
) -> None:
    """Show the graph around a note."""
    try:
        graph = get_note_graph(
            title=title,
            depth=depth,
        )
    except (
        NoteNotFoundError,
        ValueError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Graph for: {title}")
    typer.echo(f"Depth: {depth}")

    typer.echo("\nNodes:")

    for node in graph.nodes:
        typer.echo(f"  - {node.slug}")

    typer.echo("\nEdges:")

    if not graph.edges:
        typer.echo("  None")
        return

    for edge in graph.edges:
        typer.echo(
            f"  - {edge.source} "
            f"--[{edge.relation_type.value}]--> "
            f"{edge.target}"
        )


@note_app.command("ancestors")
def note_ancestors_command(title: str) -> None:
    """Show all recursive ancestors of a note."""
    try:
        ancestors = get_note_ancestors(title)
    except (
        NoteNotFoundError,
        ValueError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Ancestors of: {title}")

    if not ancestors:
        typer.echo("\nNone")
        return

    for ancestor in ancestors:
        typer.echo(f"  - {ancestor}")


@note_app.command("descendants")
def note_descendants_command(title: str) -> None:
    """Show all recursive descendants of a note."""
    try:
        descendants = get_note_descendants(title)
    except (
        NoteNotFoundError,
        ValueError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Descendants of: {title}")

    if not descendants:
        typer.echo("\nNone")
        return

    for descendant in descendants:
        typer.echo(f"  - {descendant}")


@note_app.command("stats")
def note_stats_command() -> None:
    """Show aggregate graph statistics."""
    try:
        statistics = get_graph_statistics()
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("KnowledgeForge Graph Statistics")
    typer.echo(f"Nodes: {statistics.total_nodes}")
    typer.echo(f"Edges: {statistics.total_edges}")
    typer.echo(f"Orphan nodes: {statistics.orphan_nodes}")
    typer.echo(f"Root nodes: {statistics.root_nodes}")
    typer.echo(f"Leaf nodes: {statistics.leaf_nodes}")
    typer.echo(f"Average degree: {statistics.average_degree:.2f}")
    typer.echo(f"Max degree: {statistics.max_degree}")
    typer.echo(f"Density: {statistics.density:.4f}")


@template_app.command("create")
def create_template_command(
    name: str,
    content: Annotated[
        str,
        typer.Option(
            "--content",
            "-c",
            help="Markdown content of the template.",
        ),
    ],
) -> None:
    """Create a new Markdown template."""
    try:
        result = create_template(
            name=name,
            content=content,
        )
    except (
        InvalidTemplateNameError,
        FileExistsError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(result.message)
    typer.echo(f"Path: {result.template.path}")


@template_app.command("list")
def list_templates_command() -> None:
    """List all available templates."""
    templates = list_templates()

    typer.echo("KnowledgeForge Templates")

    if not templates:
        typer.echo("\nNo templates found.")
        return

    for index, template_name in enumerate(templates, start=1):
        typer.echo(f"\n{index}. {template_name}")


@template_app.command("show")
def show_template_command(name: str) -> None:
    """Show a template and its Markdown content."""
    try:
        template = show_template(name)
    except (
        TemplateNotFoundError,
        InvalidTemplateNameError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Name: {template.name}")
    typer.echo(f"Path: {template.path}")
    typer.echo("\n" + template.content)