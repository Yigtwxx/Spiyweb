"""LLM provider abstraction - local-first (Ollama), free APIs via config.

One OpenAI-compatible chat-completions client over stdlib `urllib` covers
Ollama, Groq and OpenRouter alike; `dependencies = []` survives because this
module needs nothing installed. `core/` never imports this module - LLM calls
happen at index time only.

The API key is read from the environment variable NAMED in `LLMConfig`; the
value itself never appears in source, in an exception message or in a log.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Protocol

from spiyweb.config import LLMConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    Transport = Callable[[str, bytes, Mapping[str, str], float], bytes]
    """(url, json body, headers, timeout seconds) -> raw response bytes."""


class LLMError(RuntimeError):
    """A completion could not be obtained: transport, auth or malformed reply."""


class LLMClient(Protocol):
    """Minimal completion interface the extraction pipeline depends on."""

    def complete(self, prompt: str) -> str:
        """Return the model's completion for `prompt`."""
        ...


def _urllib_transport(
    url: str, body: bytes, headers: Mapping[str, str], timeout: float
) -> bytes:
    """Default transport: one POST, raw response bytes back."""
    request = urllib.request.Request(
        url, data=body, headers=dict(headers), method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return bytes(response.read())


class OpenAICompatClient:
    """Chat-completions client for any OpenAI-compatible endpoint.

    The transport and the retry sleep are injectable so tests exercise every
    path without a network or a real clock.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        transport: Callable[[str, bytes, Mapping[str, str], float], bytes]
        | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config if config is not None else LLMConfig()
        self._transport = transport if transport is not None else _urllib_transport
        self._sleep = sleep

    def complete(self, prompt: str) -> str:
        """POST `prompt` as a single user message; return the reply text.

        Retries `max_retries` times with exponential backoff on transport
        errors; a malformed response body raises `LLMError` immediately (the
        provider answered - retrying will not fix its shape).
        """
        config = self._config
        url = config.base_url.rstrip("/") + "/chat/completions"
        body = json.dumps(
            {
                "model": config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if config.api_key_env is not None:
            api_key = os.environ.get(config.api_key_env)
            if api_key is None:
                raise LLMError(
                    f"environment variable {config.api_key_env} is not set; "
                    "it must hold the API key for "
                    f"{config.base_url}"
                )
            headers["Authorization"] = f"Bearer {api_key}"

        last_error: Exception | None = None
        for attempt in range(config.max_retries + 1):
            if attempt > 0:
                self._sleep(config.retry_backoff_seconds * 2 ** (attempt - 1))
            try:
                raw = self._transport(url, body, headers, config.timeout_seconds)
            except (urllib.error.URLError, OSError) as error:
                last_error = error
                continue
            return _parse_completion(raw)
        raise LLMError(
            f"request to {url} failed after {config.max_retries + 1} attempts"
        ) from last_error


def _parse_completion(raw: bytes) -> str:
    """Extract `choices[0].message.content`; anything else is an `LLMError`."""
    try:
        payload = json.loads(raw)
        content = payload["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise LLMError(
            "provider returned a response without choices[0].message.content"
        ) from error
    if not isinstance(content, str):
        raise LLMError("provider returned a non-string message content")
    return content
