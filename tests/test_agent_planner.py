"""Tests for the OpenAI-compatible agent planner adapter."""

import json

import pytest

from knowledgeforge.application.agent_runtime import AgentMessage, AgentToolCall
from knowledgeforge.infrastructure.ai.agent_planner import OpenAICompatibleAgentPlanner
from knowledgeforge.infrastructure.ai.client import AIClientError


class FakeClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.messages = None
        self.tools = None

    def chat_completion(self, messages, *, tools=None):
        self.messages = messages
        self.tools = tools
        return self.response


def test_planner_normalizes_final_text_response() -> None:
    client = FakeClient(
        {
            "choices": [
                {"message": {"content": "Final answer", "role": "assistant"}}
            ]
        }
    )
    planner = OpenAICompatibleAgentPlanner(client, system_prompt="Be grounded.")

    response = planner.respond(
        (AgentMessage(role="user", content="What is in my vault?"),),
        tools=[],
    )

    assert response.content == "Final answer"
    assert response.tool_calls == ()
    assert client.messages == [
        {"role": "system", "content": "Be grounded."},
        {"role": "user", "content": "What is in my vault?"},
    ]
    assert client.tools == []


def test_planner_normalizes_tool_call_response() -> None:
    client = FakeClient(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search_knowledge",
                                    "arguments": '{"query":"machine learning","limit":3}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )
    planner = OpenAICompatibleAgentPlanner(client)

    response = planner.respond(
        (AgentMessage(role="user", content="Find machine learning notes."),),
        tools=[{"type": "function", "function": {"name": "search_knowledge"}}],
    )

    assert response.content == ""
    assert response.tool_calls == (
        AgentToolCall(
            call_id="call_1",
            name="search_knowledge",
            arguments={"query": "machine learning", "limit": 3},
        ),
    )


def test_planner_translates_assistant_tool_and_tool_messages() -> None:
    client = FakeClient(
        {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
    )
    planner = OpenAICompatibleAgentPlanner(client)
    call = AgentToolCall(
        call_id="call_1",
        name="get_note",
        arguments={"title": "ML"},
    )

    planner.respond(
        (
            AgentMessage(role="user", content="Read ML."),
            AgentMessage(role="assistant", tool_calls=(call,)),
            AgentMessage(
                role="tool",
                content=json.dumps({"note": {"title": "ML"}}),
                tool_call_id="call_1",
                tool_name="get_note",
            ),
        ),
        tools=[],
    )

    assert client.messages == [
        {"role": "user", "content": "Read ML."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_note",
                        "arguments": '{"title":"ML"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "get_note",
            "content": '{"note": {"title": "ML"}}',
        },
    ]


def test_planner_rejects_invalid_tool_arguments() -> None:
    client = FakeClient(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search_knowledge",
                                    "arguments": "[]",
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )

    with pytest.raises(AIClientError, match="must be an object"):
        OpenAICompatibleAgentPlanner(client).respond(
            (AgentMessage(role="user", content="Search."),),
            tools=[],
        )


def test_planner_rejects_invalid_provider_response() -> None:
    client = FakeClient({"choices": []})

    with pytest.raises(AIClientError, match="invalid agent response"):
        OpenAICompatibleAgentPlanner(client).respond(
            (AgentMessage(role="user", content="Hello"),),
            tools=[],
        )
