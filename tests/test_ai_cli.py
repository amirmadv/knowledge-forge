"""Tests for the KnowledgeForge AI CLI."""

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from knowledgeforge.ai_cli import app
from knowledgeforge.application.agent_runtime import (
    AgentModelResponse,
    AgentRunResult,
    AgentStep,
    AgentToolCall,
    AgentTrace,
    ToolObservation,
)

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
            reasons=(
                "semantic similarity",
                "body lexical match",
                "metadata/tag match",
            ),
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
    return SimpleNamespace(
        search_with_evidence=lambda query, limit=8: evidence[:limit]
    )


def _fake_agent_run_result() -> AgentRunResult:
    """Build a deterministic bounded agent result for CLI tests."""
    call = AgentToolCall(
        call_id="call_1",
        name="search_knowledge",
        arguments={"query": "linear regression", "limit": 3},
    )
    observation = ToolObservation(
        call_id="call_1",
        tool_name="search_knowledge",
        success=True,
        data={"results": [{"title": "Linear Regression"}]},
    )
    step = AgentStep(
        response=AgentModelResponse(content="", tool_calls=(call,)),
        observations=(observation,),
    )
    final_step = AgentStep(response=AgentModelResponse(content="Found it."))
    return AgentRunResult(
        answer="Found it.",
        trace=AgentTrace(
            steps=(step, final_step),
            tool_call_count=1,
            termination_reason="final_answer",
        ),
    )


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
            calls.append(limit)
            or _fake_agent().search_with_evidence(query, limit)
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


class _FakeNote:
    def __init__(self, slug: str) -> None:
        self.slug = slug


class _FakeEvidence:
    def __init__(self, slug: str) -> None:
        self.note = _FakeNote(slug)


class _FakeEvaluationAgent:
    def search_with_evidence(self, query: str, limit: int = 8) -> list[_FakeEvidence]:
        results = {
            "linear regression": ["linear-regression", "machine-learning"],
            "gradient descent": ["machine-learning", "gradient-descent"],
        }
        return [_FakeEvidence(slug) for slug in results.get(query, [])[:limit]]


def _write_evaluation_dataset(path: Path, relevant: str = "linear-regression") -> None:
    """Write a small deterministic dataset shared by CLI evaluation tests."""
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "query": "linear regression",
                        "relevant": [relevant],
                    },
                    {
                        "query": "gradient descent",
                        "relevant": ["gradient-descent"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_evaluate_command_reports_metrics_and_passes_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The evaluate command should report aggregate metrics and a passing gate."""
    dataset = tmp_path / "retrieval.json"
    _write_evaluation_dataset(dataset)
    monkeypatch.setattr(
        "knowledgeforge.ai_cli._agent",
        lambda: _FakeEvaluationAgent(),
    )

    result = runner.invoke(
        app,
        [
            "evaluate",
            str(dataset),
            "--k",
            "2",
            "--min-precision",
            "0.5",
            "--min-recall",
            "1.0",
            "--min-mrr",
            "0.75",
        ],
    )

    assert result.exit_code == 0
    assert "KnowledgeForge Retrieval Evaluation" in result.stdout
    assert "Queries: 2" in result.stdout
    assert "Precision@2: 0.5000" in result.stdout
    assert "Recall@2: 1.0000" in result.stdout
    assert "MRR: 0.7500" in result.stdout
    assert "Quality gate: PASSED" in result.stdout


def test_evaluate_command_fails_when_quality_gate_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The evaluate command should fail CI when the configured gate fails."""
    dataset = tmp_path / "retrieval.json"
    dataset.write_text(
        '[{"query": "linear regression", "relevant": ["missing-note"]}]',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "knowledgeforge.ai_cli._agent",
        lambda: _FakeEvaluationAgent(),
    )

    result = runner.invoke(
        app,
        [
            "evaluate",
            str(dataset),
            "--k",
            "1",
            "--min-precision",
            "1.0",
        ],
    )

    assert result.exit_code == 2
    assert "Quality gate: FAILED" in result.stdout
    assert "precision@1" in result.stdout


def test_evaluate_command_can_report_failed_gate_without_failing_process(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The gate can be inspected without failing the calling process."""
    dataset = tmp_path / "retrieval.json"
    dataset.write_text(
        '[{"query": "linear regression", "relevant": ["missing-note"]}]',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "knowledgeforge.ai_cli._agent",
        lambda: _FakeEvaluationAgent(),
    )

    result = runner.invoke(
        app,
        [
            "evaluate",
            str(dataset),
            "--min-precision",
            "1.0",
            "--no-fail-on-gate",
            "--details",
        ],
    )

    assert result.exit_code == 0
    assert "Quality gate: FAILED" in result.stdout
    assert "Cases:" in result.stdout
    assert "retrieved: linear-regression, machine-learning" in result.stdout


def test_evaluate_command_supports_json_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The evaluate command should emit stable machine-readable JSON."""
    dataset = tmp_path / "retrieval.json"
    _write_evaluation_dataset(dataset)
    monkeypatch.setattr(
        "knowledgeforge.ai_cli._agent",
        lambda: _FakeEvaluationAgent(),
    )

    result = runner.invoke(
        app,
        [
            "evaluate",
            str(dataset),
            "--k",
            "2",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["k"] == 2
    assert payload["queries_evaluated"] == 2
    assert payload["metrics"] == {
        "precision_at_k": 0.5,
        "recall_at_k": 1.0,
        "mrr": 0.75,
    }
    assert payload["quality_gate"] == {"passed": True, "failures": []}
    assert payload["cases"][0]["retrieved"] == [
        "linear-regression",
        "machine-learning",
    ]


def test_evaluate_command_rejects_unknown_output_format(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The evaluate command should reject unsupported output formats."""
    dataset = tmp_path / "retrieval.json"
    _write_evaluation_dataset(dataset)
    monkeypatch.setattr(
        "knowledgeforge.ai_cli._agent",
        lambda: _FakeEvaluationAgent(),
    )

    result = runner.invoke(
        app,
        ["evaluate", str(dataset), "--output", "yaml"],
    )

    assert result.exit_code == 1
    assert "Output format must be 'text' or 'json'." in result.stderr


def test_agent_command_prints_answer_and_trace(monkeypatch) -> None:
    """The agent command should expose a compact human-readable trace."""
    fake = SimpleNamespace(run_agent=lambda prompt: _fake_agent_run_result())
    monkeypatch.setattr("knowledgeforge.ai_cli._agent", lambda: fake)

    result = runner.invoke(app, ["agent", "Find linear regression"])

    assert result.exit_code == 0
    assert "Found it." in result.stdout
    assert "Agent trace:" in result.stdout
    assert "termination: final_answer" in result.stdout
    assert "tool calls: 1" in result.stdout
    assert "search_knowledge: ok" in result.stdout


def test_agent_command_supports_json_output(monkeypatch) -> None:
    """The agent command should expose a stable machine-readable trace."""
    fake = SimpleNamespace(run_agent=lambda prompt: _fake_agent_run_result())
    monkeypatch.setattr("knowledgeforge.ai_cli._agent", lambda: fake)

    result = runner.invoke(
        app,
        ["agent", "Find linear regression", "--output", "json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["answer"] == "Found it."
    assert payload["trace"]["tool_call_count"] == 1
    assert payload["trace"]["termination_reason"] == "final_answer"
    assert payload["trace"]["steps"][0]["response"]["tool_calls"][0]["name"] == (
        "search_knowledge"
    )
    assert payload["trace"]["steps"][0]["observations"][0]["success"] is True


def test_agent_command_rejects_unknown_output_format(monkeypatch) -> None:
    """The agent command should reject unsupported output formats."""
    fake = SimpleNamespace(run_agent=lambda prompt: _fake_agent_run_result())
    monkeypatch.setattr("knowledgeforge.ai_cli._agent", lambda: fake)

    result = runner.invoke(app, ["agent", "hello", "--output", "yaml"])

    assert result.exit_code == 1
    assert "Output format must be 'text' or 'json'." in result.stderr
