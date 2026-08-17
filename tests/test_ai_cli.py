"""Tests for the KnowledgeForge AI CLI."""

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from knowledgeforge.ai_cli import app

runner = CliRunner()


def _fake_agent() -> SimpleNamespace:
    """Build a deterministic fake agent for CLI output tests."""
    evidence = [
        SimpleNamespace(
            note=SimpleNamespace(title="Gradient Descent"),
            score=0.8125,
            semantic_score=0.9,
            lexical_score=0.5,
            metadata_score=0.25,
            reasons=("semantic similarity", "body lexical match", "metadata/tag match"),
        ),
        SimpleNamespace(
            note=SimpleNamespace(title="Linear Regression"),
            score=0.421,
            semantic_score=0.4,
            lexical_score=0.5,
            metadata_score=0.0,
            reasons=("title/body lexical match",),
        ),
    ]
    return SimpleNamespace(search_with_evidence=lambda query, limit=8: evidence[:limit])


def test_search_command_prints_ranked_titles(monkeypatch) -> None:
    """The default search output should remain compact and deterministic."""
    monkeypatch.setattr("knowledgeforge.ai_cli._agent", _fake_agent)

    result = runner.invoke(app, ["search", "gradient descent"])

    assert result.exit_code == 0
    assert "Search results for: gradient descent" in result.stdout
    assert "1. Gradient Descent (0.812)" in result.stdout
    assert "2. Linear Regression (0.421)" in result.stdout
    assert "reasons:" not in result.stdout


def test_search_explain_prints_score_breakdown_and_reasons(monkeypatch) -> None:
    """The explain flag should expose the retrieval signals and reasons."""
    monkeypatch.setattr("knowledgeforge.ai_cli._agent", _fake_agent)

    result = runner.invoke(app, ["search", "gradient descent", "--explain"])

    assert result.exit_code == 0
    assert "score: 0.812" in result.stdout
    assert "semantic: 0.900" in result.stdout
    assert "lexical: 0.500" in result.stdout
    assert "metadata: 0.250" in result.stdout
    assert "semantic similarity" in result.stdout
    assert "body lexical match" in result.stdout
    assert "metadata/tag match" in result.stdout


def test_search_limit_is_respected(monkeypatch) -> None:
    """The CLI should pass the requested result limit to the application layer."""
    calls: list[int] = []
    agent = SimpleNamespace(
        search_with_evidence=lambda query, limit=8: (
            calls.append(limit) or _fake_agent().search_with_evidence(query, limit)
        )
    )
    monkeypatch.setattr("knowledgeforge.ai_cli._agent", lambda: agent)

    result = runner.invoke(app, ["search", "gradient", "--limit", "1"])

    assert result.exit_code == 0
    assert calls == [1]
    assert "Gradient Descent" in result.stdout
    assert "Linear Regression" not in result.stdout


def test_search_explain_handles_empty_results(monkeypatch) -> None:
    """An empty evidence list should produce a clear result instead of an error."""
    monkeypatch.setattr(
        "knowledgeforge.ai_cli._agent",
        lambda: SimpleNamespace(search_with_evidence=lambda query, limit=8: []),
    )

    result = runner.invoke(app, ["search", "unknown", "--explain"])

    assert result.exit_code == 0
    assert "No matching notes found." in result.stdout
