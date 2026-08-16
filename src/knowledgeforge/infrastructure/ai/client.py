"""Provider-agnostic OpenAI-compatible AI client."""

from __future__ import annotations

import json
from urllib import error, request


class AIClientError(RuntimeError):
    """Raised when an AI provider request cannot be completed."""


class OpenAICompatibleClient:
    """Small standard-library client for OpenAI-compatible chat APIs."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def chat(self, prompt: str, system: str | None = None) -> str:
        """Send one chat request and return the assistant text."""
        if not prompt.strip():
            raise AIClientError("Prompt cannot be empty.")

        messages: list[dict[str, str]] = []
        if system and system.strip():
            messages.append({"role": "system", "content": system.strip()})
        messages.append({"role": "user", "content": prompt.strip()})

        payload = json.dumps(
            {"model": self._model, "messages": messages},
        ).encode("utf-8")

        req = request.Request(
            f"{self._base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self._timeout) as response:
                raw = response.read().decode("utf-8")
        except (error.HTTPError, error.URLError, TimeoutError) as exc:
            raise AIClientError(f"AI provider request failed: {exc}") from exc

        try:
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise AIClientError("AI provider returned an invalid response.") from exc

        if not isinstance(content, str) or not content.strip():
            raise AIClientError("AI provider returned empty content.")

        return content.strip()
