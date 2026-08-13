"""LLM client: request shape, auth via env name only, parse errors, retries."""

from __future__ import annotations

import json

import pytest

from spiyweb import LLMConfig
from spiyweb.llm import LLMError, OpenAICompatClient


def make_reply(content: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode()


class RecordingTransport:
    """Captures every request; replies from a scripted list of outcomes."""

    def __init__(self, outcomes: list[bytes | Exception]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, bytes, dict[str, str], float]] = []

    def __call__(
        self, url: str, body: bytes, headers: dict[str, str], timeout: float
    ) -> bytes:
        self.calls.append((url, body, dict(headers), timeout))
        outcome = self.outcomes[min(len(self.calls), len(self.outcomes)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_request_carries_model_prompt_temperature_and_max_tokens() -> None:
    transport = RecordingTransport([make_reply("ok")])
    config = LLMConfig(model="test-model", temperature=0.3, max_tokens=7)
    client = OpenAICompatClient(config, transport=transport)

    assert client.complete("the prompt") == "ok"
    url, body, _headers, timeout = transport.calls[0]
    assert url == "http://localhost:11434/v1/chat/completions"
    payload = json.loads(body)
    assert payload["model"] == "test-model"
    assert payload["messages"] == [{"role": "user", "content": "the prompt"}]
    assert payload["temperature"] == pytest.approx(0.3)
    assert payload["max_tokens"] == 7
    assert timeout == pytest.approx(config.timeout_seconds)


def test_no_api_key_env_sends_no_authorization_header() -> None:
    transport = RecordingTransport([make_reply("ok")])
    OpenAICompatClient(LLMConfig(), transport=transport).complete("p")
    _, _, headers, _ = transport.calls[0]
    assert "Authorization" not in headers, (
        "Ollama's local endpoint is keyless; no header may be invented"
    )


def test_api_key_env_set_sends_bearer_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPIYWEB_TEST_KEY", "sekrit-value")
    transport = RecordingTransport([make_reply("ok")])
    config = LLMConfig(api_key_env="SPIYWEB_TEST_KEY")
    OpenAICompatClient(config, transport=transport).complete("p")
    _, _, headers, _ = transport.calls[0]
    assert headers["Authorization"] == "Bearer sekrit-value"


def test_missing_env_var_raises_naming_the_variable_never_a_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPIYWEB_TEST_KEY", raising=False)
    transport = RecordingTransport([make_reply("ok")])
    config = LLMConfig(api_key_env="SPIYWEB_TEST_KEY")
    client = OpenAICompatClient(config, transport=transport)
    with pytest.raises(LLMError, match="SPIYWEB_TEST_KEY"):
        client.complete("p")
    assert transport.calls == [], "no request may leave without the key"


def test_malformed_json_and_missing_choices_raise_llm_error() -> None:
    for raw in [b"not json", b"{}", b'{"choices": []}']:
        client = OpenAICompatClient(LLMConfig(), transport=RecordingTransport([raw]))
        with pytest.raises(LLMError, match="choices"):
            client.complete("p")


def test_non_string_content_raises_llm_error() -> None:
    raw = json.dumps({"choices": [{"message": {"content": 42}}]}).encode()
    client = OpenAICompatClient(LLMConfig(), transport=RecordingTransport([raw]))
    with pytest.raises(LLMError, match="non-string"):
        client.complete("p")


def test_transport_failures_are_retried_with_exponential_backoff() -> None:
    transport = RecordingTransport(
        [OSError("down"), OSError("down"), make_reply("recovered")]
    )
    sleeps: list[float] = []
    config = LLMConfig(max_retries=2, retry_backoff_seconds=1.0)
    client = OpenAICompatClient(config, transport=transport, sleep=sleeps.append)

    assert client.complete("p") == "recovered"
    assert len(transport.calls) == 3
    assert sleeps == [pytest.approx(1.0), pytest.approx(2.0)], (
        "backoff doubles per retry: base * 2**attempt"
    )


def test_exhausted_retries_raise_llm_error_with_attempt_count() -> None:
    transport = RecordingTransport([OSError("down")])
    config = LLMConfig(max_retries=1, retry_backoff_seconds=0.0)
    client = OpenAICompatClient(config, transport=transport, sleep=lambda _: None)
    with pytest.raises(LLMError, match="2 attempts"):
        client.complete("p")
    assert len(transport.calls) == 2


def test_base_url_trailing_slash_does_not_double_the_separator() -> None:
    transport = RecordingTransport([make_reply("ok")])
    config = LLMConfig(base_url="https://api.example.test/v1/")
    OpenAICompatClient(config, transport=transport).complete("p")
    url, _, _, _ = transport.calls[0]
    assert url == "https://api.example.test/v1/chat/completions"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_url", ""),
        ("model", ""),
        ("timeout_seconds", 0.0),
        ("temperature", -0.1),
        ("max_tokens", 0),
        ("max_retries", -1),
        ("retry_backoff_seconds", -1.0),
    ],
)
def test_config_rejects_out_of_range_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field.split("_")[0]):
        LLMConfig(**{field: value})  # type: ignore[arg-type]
