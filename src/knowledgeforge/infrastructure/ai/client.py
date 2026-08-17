"""Provider-agnostic OpenAI-compatible AI client."""

from __future__ import annotations

import json
from urllib import error, request


class AIClientError(RuntimeError):
    """Raised when an AI provider request cannot be completed."""


class OpenAICompatibleClient:
    """Small standard-library client for chat and embedding APIs."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._embedding_model = embedding_model

    def chat(self, prompt: str, system: str | None = None) -> str:
        """Send one chat request and return the assistant text."""
        if not prompt.strip():
            raise AIClientError("Prompt cannot be empty.")

        messages: list[dict[str, str]] = []
        if system and system.strip():
            messages.append({"role": "system", "content": system.strip()})
        messages.append({"role": "user", "content": prompt.strip()})

        data = self._post(
            "/chat/completions",
            {"model": self._model, "messages": messages},
        )

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AIClientError("AI provider returned an invalid response.") from exc

        if not isinstance(content, str) or not content.strip():
            raise AIClientError("AI provider returned empty content.")

        return content.strip()

    def embed(self, text: str) -> tuple[float, ...]:
        """Create an embedding vector for semantic retrieval."""
        if not text.strip():
            raise AIClientError("Text to embed cannot be empty.")

        data = self._post(
            "/embeddings",
            {"model": self._embedding_model, "input": text.strip()},
        )

        try:
            vector = data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AIClientError("AI provider returned an invalid embedding response.") from exc

        if not isinstance(vector, list) or not all(
            isinstance(value, (int, float)) for value in vector
        ):
            raise AIClientError("AI provider returned an invalid embedding vector.")

        return tuple(float(value) for value in vector)

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        """POST JSON to an OpenAI-compatible endpoint."""
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self._base_url}{path}",
            data=body,
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
        except json.JSONDecodeError as exc:
            raise AIClientError("AI provider returned invalid JSON.") from exc

        if not isinstance(data, dict):
            raise AIClientError("AI provider returned an invalid response.")

        return data
