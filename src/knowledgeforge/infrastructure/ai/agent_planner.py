"""OpenAI-compatible provider adapter for the KnowledgeForge agent runtime."""

from __future__ import annotations

import json
from typing import Any

from knowledgeforge.application.agent_runtime import (
    AgentMessage,
    AgentModelResponse,
    AgentPlanner,
    AgentToolCall,
)
from knowledgeforge.infrastructure.ai.client import AIClientError, OpenAICompatibleClient


class OpenAICompatibleAgentPlanner(AgentPlanner):
    """Translate provider-neutral runtime messages to Chat Completions messages."""

    def __init__(
        self,
        client: OpenAICompatibleClient,
        system_prompt: str | None = None,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt.strip() if system_prompt else None

    def respond(
        self,
        messages: tuple[AgentMessage, ...],
        tools: list[dict[str, Any]],
    ) -> AgentModelResponse:
        """Request one provider response and normalize it for the runtime."""
        provider_messages = self._to_provider_messages(messages)
        if self._system_prompt and not any(
            message.get("role") == "system" for message in provider_messages
        ):
            provider_messages.insert(
                0,
                {"role": "system", "content": self._system_prompt},
            )

        data = self._client.chat_completion(provider_messages, tools=tools)
        return self._from_provider_response(data)

    @staticmethod
    def _to_provider_messages(
        messages: tuple[AgentMessage, ...],
    ) -> list[dict[str, Any]]:
        provider_messages: list[dict[str, Any]] = []
        for message in messages:
            if message.role in {"system", "user"}:
                provider_messages.append(
                    {"role": message.role, "content": message.content}
                )
                continue

            if message.role == "assistant":
                provider_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content or None,
                }
                if message.tool_calls:
                    provider_message["tool_calls"] = [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(
                                    call.arguments,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            },
                        }
                        for call in message.tool_calls
                    ]
                provider_messages.append(provider_message)
                continue

            if message.role == "tool":
                if not message.tool_call_id or not message.tool_name:
                    raise AIClientError(
                        "Tool messages require tool_call_id and tool_name."
                    )
                provider_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "name": message.tool_name,
                        "content": message.content,
                    }
                )
                continue

            raise AIClientError(f"Unsupported runtime message role: {message.role}.")

        if not provider_messages:
            raise AIClientError("Agent planner requires at least one message.")
        return provider_messages

    @staticmethod
    def _from_provider_response(data: dict[str, Any]) -> AgentModelResponse:
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AIClientError("AI provider returned an invalid agent response.") from exc

        if not isinstance(message, dict):
            raise AIClientError("AI provider returned an invalid agent message.")

        content = message.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise AIClientError("AI provider returned non-text agent content.")

        raw_tool_calls = message.get("tool_calls", []) or []
        if not isinstance(raw_tool_calls, list):
            raise AIClientError("AI provider returned invalid tool calls.")

        tool_calls: list[AgentToolCall] = []
        seen_ids: set[str] = set()
        for raw_call in raw_tool_calls:
            tool_call = _parse_tool_call(raw_call)
            if tool_call.call_id in seen_ids:
                raise AIClientError(
                    f"AI provider returned duplicate tool call id: {tool_call.call_id}."
                )
            seen_ids.add(tool_call.call_id)
            tool_calls.append(tool_call)

        return AgentModelResponse(
            content=content,
            tool_calls=tuple(tool_calls),
        )


def _parse_tool_call(raw_call: Any) -> AgentToolCall:
    if not isinstance(raw_call, dict):
        raise AIClientError("AI provider returned an invalid tool call.")

    call_id = raw_call.get("id")
    call_type = raw_call.get("type")
    function = raw_call.get("function")
    if not isinstance(call_id, str) or not call_id.strip():
        raise AIClientError("AI provider returned a tool call without an id.")
    if call_type != "function" or not isinstance(function, dict):
        raise AIClientError("AI provider returned an unsupported tool call type.")

    name = function.get("name")
    raw_arguments = function.get("arguments", "{}")
    if not isinstance(name, str) or not name.strip():
        raise AIClientError("AI provider returned a tool call without a function name.")
    if not isinstance(raw_arguments, str):
        raise AIClientError("AI provider returned non-text tool arguments.")

    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise AIClientError(
            f"AI provider returned invalid JSON tool arguments for {name}."
        ) from exc

    if not isinstance(arguments, dict):
        raise AIClientError(
            f"AI provider tool arguments must be an object for {name}."
        )

    return AgentToolCall(
        call_id=call_id.strip(),
        name=name.strip(),
        arguments=arguments,
    )
