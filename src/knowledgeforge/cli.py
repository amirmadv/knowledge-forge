"""Command-line interface for KnowledgeForge."""

import typer

from knowledgeforge.application.commands import initialize_knowledgeforge

app = typer.Typer(
    name="knowledgeforge",
    help="AI-powered personal knowledge management system.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Manage the KnowledgeForge application."""


@app.command("init")
def init_command() -> None:
    """Initialize a KnowledgeForge vault."""
    result = initialize_knowledgeforge()

    typer.echo(result.message)
    typer.echo(f"Vault: {result.vault_path}")