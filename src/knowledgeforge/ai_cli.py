"""CLI for the KnowledgeForge AI agent."""

from __future__ import annotations

import typer

from knowledgeforge.application.ai import KnowledgeAgent
from knowledgeforge.application.chat import KnowledgeChatSession
from knowledgeforge.infrastructure.config.settings import Settings

app = typer.Typer(
    name="knowledgeforge-ai",
    help="Ask the KnowledgeForge AI agent about your local knowledge vault.",
    no_args_is_help=True,
)


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
        result = KnowledgeAgent(Settings()).ask(question)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    _print_answer(result.answer, result.sources)


@app.command("chat")
def chat() -> None:
    """Start an interactive, vault-grounded AI conversation."""
    try:
        session = KnowledgeChatSession(KnowledgeAgent(Settings()))
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
