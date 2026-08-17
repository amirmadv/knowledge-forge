"""CLI for the KnowledgeForge AI agent."""

from __future__ import annotations

import typer

from knowledgeforge.application.ai import KnowledgeAgent
from knowledgeforge.infrastructure.config.settings import Settings

app = typer.Typer(
    name="knowledgeforge-ai",
    help="Ask the KnowledgeForge AI agent about your local knowledge vault.",
    no_args_is_help=True,
)


@app.command("ask")
def ask(question: str) -> None:
    """Ask the AI agent a question using local vault context."""
    try:
        result = KnowledgeAgent(Settings()).ask(question)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(result.answer)

    if result.sources:
        typer.echo("\nSources:")
        for source in result.sources:
            typer.echo(f"- {source}")
