"""CLI for the KnowledgeForge AI agent and Copilot."""

from __future__ import annotations

import typer

from knowledgeforge.application.ai import KnowledgeAgent
from knowledgeforge.application.chat import KnowledgeChatSession
from knowledgeforge.application.copilot import KnowledgeCopilot
from knowledgeforge.infrastructure.config.settings import Settings

app = typer.Typer(
    name="knowledgeforge-ai",
    help="Ask the KnowledgeForge AI agent about your local knowledge vault.",
    no_args_is_help=True,
)


def _agent() -> KnowledgeAgent:
    """Build the configured KnowledgeForge AI agent."""
    return KnowledgeAgent(Settings())


def _copilot() -> KnowledgeCopilot:
    """Build the configured KnowledgeForge AI Copilot."""
    agent = _agent()
    return KnowledgeCopilot(agent)


def _print_answer(answer: str, sources: tuple[str, ...]) -> None:
    """Print an AI answer and its grounded sources."""
    typer.echo(answer)

    if sources:
        typer.echo("\nSources:")
        for source in sources:
            typer.echo(f"- {source}")


@app.command("ask")
def ask(question: str) -> None:
    """Ask the AI agent a question using local vault context."""
    try:
        result = _agent().ask(question)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    _print_answer(result.answer, result.sources)


@app.command("chat")
def chat() -> None:
    """Start an interactive, vault-grounded AI conversation."""
    try:
        session = KnowledgeChatSession(_agent())
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("KnowledgeForge AI chat")
    typer.echo("Commands: /clear, /exit")

    while True:
        try:
            question = typer.prompt("You")
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            break

        command = question.strip().casefold()
        if command in {"/exit", "/quit"}:
            break
        if command == "/clear":
            session.clear()
            typer.echo("Conversation cleared.")
            continue
        if not question.strip():
            continue

        try:
            result = session.ask(question)
        except Exception as exc:
            typer.echo(str(exc), err=True)
            continue

        typer.echo("\nKnowledgeForge")
        _print_answer(result.answer, result.sources)
        typer.echo()


@app.command("copilot-summary")
def copilot_summary(title: str) -> None:
    """Summarize a note with the AI Copilot."""
    try:
        result = _copilot().summarize(title)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    _print_answer(result.answer, result.sources)


@app.command("copilot-tags")
def copilot_tags(title: str) -> None:
    """Suggest tags for a note without modifying it."""
    try:
        result = _copilot().suggest_tags(title)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    _print_answer(result.answer, result.sources)


@app.command("copilot-improve")
def copilot_improve(title: str) -> None:
    """Improve a note's Markdown while preserving its facts."""
    try:
        result = _copilot().improve(title)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    _print_answer(result.answer, result.sources)


@app.command("copilot-related")
def copilot_related(title: str) -> None:
    """Find notes related to a selected note."""
    try:
        result = _copilot().related(title)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    _print_answer(result.answer, result.sources)


@app.command("copilot-gaps")
def copilot_gaps(title: str) -> None:
    """Identify knowledge gaps around a note."""
    try:
        result = _copilot().knowledge_gaps(title)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    _print_answer(result.answer, result.sources)


@app.command("copilot-create")
def copilot_create(title: str, instruction: str) -> None:
    """Create a new note from an AI-generated Markdown body."""
    try:
        result = _copilot().create_note(title, instruction)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    _print_answer(result.answer, result.sources)
