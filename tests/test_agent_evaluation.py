from pathlib import Path

import pytest

from knowledgeforge.application.agent_evaluation import (
    AgentEvaluationCase,
    evaluate_agent_cases,
    load_agent_evaluation_cases,
)
from knowledgeforge.application.agent_runtime import (
    AgentModelResponse,
    AgentRuntime,
    AgentToolCall,
)
from knowledgeforge.application.tools import (
    KnowledgeToolRegistry,
    ToolResult,
    ToolSpec,
)


class _FakeGetNoteTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="get_note",
            description="Read a note.",
            input_schema={
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        return ToolResult(
            tool_name="get_note",
            data={
                "title": arguments["title"],
                "content": "A note about regression.",
            },
        )


def _registry() -> KnowledgeToolRegistry:
    return KnowledgeToolRegistry((_FakeGetNoteTool(),))


def test_evaluation_replays_real_runtime_and_reports_success() -> None:
    case = AgentEvaluationCase(
        case_id="read-note",
        prompt="Read Linear Regression.",
        responses=(
            AgentModelResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-1",
                        name="get_note",
                        arguments={"title": "Linear Regression"},
                    ),
                )
            ),
            AgentModelResponse(content="Linear Regression is in the vault."),
        ),
        expected_answer_contains=("Linear Regression",),
        required_tools=("get_note",),
        max_tool_calls=1,
    )

    report = evaluate_agent_cases([case], AgentRuntime(_registry()))

    assert report.cases_evaluated == 1
    assert report.passed_cases == 1
    assert report.completion_rate == 1.0
    assert report.average_tool_calls == 1.0
    assert report.tool_success_rate == 1.0
    assert report.repeated_tool_call_cases == 0
    assert report.items[0].observed_tools == ("get_note",)


def test_evaluation_reports_missing_required_tool_and_answer_text() -> None:
    case = AgentEvaluationCase(
        case_id="bad-plan",
        prompt="Read Linear Regression.",
        responses=(AgentModelResponse(content="I don't know."),),
        expected_answer_contains=("Linear Regression",),
        required_tools=("get_note",),
    )

    report = evaluate_agent_cases([case], AgentRuntime(_registry()))

    assert report.passed_cases == 0
    assert report.completion_rate == 0.0
    assert "Required tool was not called: get_note." in report.items[0].failures
    assert "Answer does not contain expected text: 'Linear Regression'." in report.items[0].failures


def test_evaluation_tracks_repeated_tool_signatures() -> None:
    case = AgentEvaluationCase(
        case_id="repeat-tool",
        prompt="Read Linear Regression.",
        responses=(
            AgentModelResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-1",
                        name="get_note",
                        arguments={"title": "Linear Regression"},
                    ),
                )
            ),
            AgentModelResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-2",
                        name="get_note",
                        arguments={"title": "Linear Regression"},
                    ),
                )
            ),
            AgentModelResponse(content="Linear Regression is in the vault."),
        ),
    )

    report = evaluate_agent_cases([case], AgentRuntime(_registry()))

    assert report.items[0].repeated_tool_call_count == 1
    assert report.repeated_tool_call_cases == 1


def test_load_agent_evaluation_cases_supports_object_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "agent-eval.json"
    dataset.write_text(
        """
        {
          "cases": [
            {
              "id": "read-note",
              "prompt": "Read Linear Regression.",
              "expected_answer_contains": ["Linear Regression"],
              "required_tools": ["get_note"],
              "max_tool_calls": 1,
              "responses": [
                {
                  "content": "",
                  "tool_calls": [
                    {
                      "call_id": "call-1",
                      "name": "get_note",
                      "arguments": {"title": "Linear Regression"}
                    }
                  ]
                },
                {
                  "content": "Linear Regression is in the vault.",
                  "tool_calls": []
                }
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    cases = load_agent_evaluation_cases(dataset)

    assert len(cases) == 1
    assert cases[0].case_id == "read-note"
    assert cases[0].responses[0].tool_calls[0].name == "get_note"


def test_load_agent_evaluation_cases_rejects_invalid_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "invalid.json"
    dataset.write_text("{\"cases\": []}", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty cases list"):
        load_agent_evaluation_cases(dataset)


def test_evaluation_report_has_stable_json_shape() -> None:
    case = AgentEvaluationCase(
        case_id="final-answer",
        prompt="Say hello.",
        responses=(AgentModelResponse(content="Hello."),),
    )

    report = evaluate_agent_cases([case], AgentRuntime(_registry()))
    payload = report.as_dict()

    assert payload["cases_evaluated"] == 1
    assert payload["passed_cases"] == 1
    assert payload["cases"][0]["case_id"] == "final-answer"
    assert "completion_rate" in payload
