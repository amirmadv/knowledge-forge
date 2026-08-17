"""CLI for the KnowledgeForge AI agent and Copilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from knowledgeforge.application.ai import KnowledgeAgent
from knowledgeforge.application.chat import KnowledgeChatSession
from knowledgeforge.application.copilot import KnowledgeCopilot
from knowledgeforge.application.evaluation import (
    evaluate_retriever,
    load_retrieval_evaluation_cases,
)
from knowledgeforge.application.retrieval import RetrievalEvidence
from knowledgeforge.infrastructure.config.settings import Settings

app = typer.Typer(
    name="knowledgeforge-ai",
    help="Ask the KnowledgeForge AI agent about your local knowledge vault.",
    no_args_is_help=True,
)

DATASET_ARGUMENT = typer.Argument(
    ...,
    exists=True,
    dir_okay=False,
    readable=True,
    help="JSON retrieval evaluation dataset.",
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


def _print_retrieval_evidence(evidence: list[RetrievalEvidence]) -> None:
    """Print deterministic, human-readable retrieval evidence."""
    if not evidence:
        typer.echo("No matching notes found.")
        return

    for rank, item in enumerate(evidence, start=1):
        typer.echo(f"{rank}. {item.note.title}")
        typer.echo(f"   score: {item.score:.3f}")
        typer.echo(f"   semantic: {item.semantic_score:.3f}")
        typer.echo(f"   lexical: {item.lexical_score:.3f}")
        typer.echo(f"   metadata: {item.metadata_score:.3f}")
        if item.reasons:
            typer.echo(f"   reasons: {', '.join(item.reasons)}")
        typer.echo()


def _evaluation_payload(dataset: Path, report, k: int) -> dict[str, object]:
    """Build a stable JSON-serializable retrieval evaluation payload."""
    result = report.result
    gate = report.quality_gate
    return {
        "dataset": str(dataset),
        "k": k,
        "queries_evaluated": result.queries_evaluated,
        "metrics": {
            "precision_at_k": result.precision_at_k,
            "recall_at_k": result.recall_at_k,
            "mrr": result.mrr,
        },
        "thresholds": {
            "min_precision_at_k": report.min_precision_at_k,
            "min_recall_at_k": report.min_recall_at_k,
            "min_mrr": report.min_mrr,
        },
        "quality_gate": {
            "passed": gate.passed,
            "failures": list(gate.failures),
        },
        "cases": [
            {
                "query": item.query,
                "relevant": sorted(item.relevant),
                "retrieved": list(item.retrieved),
                "precision_at_k": item.precision_at_k,
                "recall_at_k": item.recall_at_k,
                "reciprocal_rank": item.reciprocal_rank,
            }
            for item in report.items
        ],
    }


@app.command("ask")
def ask(question: str) -> None:
    """Ask the AI agent a question using local vault context."""
    try:
        result = _agent().ask(question)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    _print_answer(result.answer, result.sources)


@app.command("search")
def search(
    query: str,
    limit: Annotated[
        int,
        typer.Option(min=1, max=50, help="Maximum number of results."),
    ] = 8,
    explain: Annotated[
        bool,
        typer.Option(
            "--explain",
            help="Show score contributions and retrieval reasons.",
        ),
    ] = False,
) -> None:
    """Search the local vault and optionally explain the ranking."""
    try:
        evidence = _agent().search_with_evidence(query, limit=limit)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Search results for: {query}")
    if explain:
        _print_retrieval_evidence(evidence)
        return

    if not evidence:
        typer.echo("No matching notes found.")
        return

    for rank, item in enumerate(evidence, start=1):
        typer.echo(f"{rank}. {item.note.title} ({item.score:.3f})")


@app.command("evaluate")
def evaluate(
    dataset: Annotated[Path, DATASET_ARGUMENT],
    k: Annotated[
        int,
        typer.Option(
            min=1,
            max=50,
            help="Number of top-ranked results used by the metrics.",
        ),
    ] = 5,
    min_precision: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Minimum required precision@k.",
        ),
    ] = 0.0,
    min_recall: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Minimum required recall@k.",
        ),
    ] = 0.0,
    min_mrr: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Minimum required mean reciprocal rank.",
        ),
    ] = 0.0,
    details: Annotated[
        bool,
        typer.Option(
            "--details",
            help="Print per-query retrieval metrics and ranked note slugs.",
        ),
    ] = False,
    fail_on_gate: Annotated[
        bool,
        typer.Option(
            "--fail-on-gate/--no-fail-on-gate",
            help="Return a non-zero exit code when the quality gate fails.",
        ),
    ] = True,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            help="Output format: text or json.",
        ),
    ] = "text",
) -> None:
    """Evaluate hybrid retrieval against an offline gold dataset."""
    output = output.casefold()
    if output not in {"text", "json"}:
        typer.echo("Output format must be 'text' or 'json'.", err=True)
        raise typer.Exit(code=1)

    try:
        cases = load_retrieval_evaluation_cases(dataset)
        agent = _agent()

        report = evaluate_retriever(
            cases,
            lambda query, limit: [
                evidence.note.slug
                for evidence in agent.search_with_evidence(query, limit=limit)
            ],
            k=k,
            min_precision_at_k=min_precision,
            min_recall_at_k=min_recall,
            min_mrr=min_mrr,
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    result = report.result
    gate = report.quality_gate

    if output == "json":
        typer.echo(
            json.dumps(
                _evaluation_payload(dataset, report, k),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        typer.echo("KnowledgeForge Retrieval Evaluation")
        typer.echo(f"Dataset: {dataset}")
        typer.echo(f"Queries: {result.queries_evaluated}")
        typer.echo(f"Precision@{k}: {result.precision_at_k:.4f}")
        typer.echo(f"Recall@{k}: {result.recall_at_k:.4f}")
        typer.echo(f"MRR: {result.mrr:.4f}")
        typer.echo(
            "Quality gate: "
            f"{'PASSED' if gate.passed else 'FAILED'}"
        )

        if gate.failures:
            typer.echo("Failures:")
            for failure in gate.failures:
                typer.echo(f"- {failure}")

        if details:
            typer.echo("\nCases:")
            for index, item in enumerate(report.items, start=1):
                retrieved = ", ".join(item.retrieved) or "(none)"
                typer.echo(f"\n{index}. {item.query}")
                typer.echo(f"   precision@{k}: {item.precision_at_k:.4f}")
                typer.echo(f"   recall@{k}: {item.recall_at_k:.4f}")
                typer.echo(f"   reciprocal rank: {item.reciprocal_rank:.4f}")
                typer.echo(f"   retrieved: {retrieved}")

    if fail_on_gate and not gate.passed:
        raise typer.Exit(code=2)


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


@app.command("index")
def index() -> None:
    """Build or refresh the local semantic embedding index."""
    try:
        count = _agent().rebuild_semantic_index()
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Semantic index ready: {count} notes indexed.")


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
