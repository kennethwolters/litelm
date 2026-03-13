"""Tests for completion/acompletion — mocked at the OpenAI client level."""

import asyncio
from unittest import mock

import openai
import pytest

from litelm._types import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
)
from litelm._completion import completion, acompletion, _prepare_call, _map_openai_error
from litelm._types import ModelResponse
from litelm._exceptions import (
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    InternalServerError,
    LitelmError,
    RateLimitError,
    Timeout,
)


def _mock_completion():
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="Hi!"), finish_reason="stop")],
        created=0,
        model="gpt-4o",
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


def test_prepare_call_strips_litellm_kwargs():
    kwargs = {
        "cache": True,
        "num_retries": 3,
        "retry_strategy": "exponential",
        "caching": True,
        "temperature": 0.5,
        "api_key": "sk-test",
    }
    provider, model, base_url, api_key, api_version, num_retries, cleaned = _prepare_call("openai/gpt-4o", kwargs)
    assert provider == "openai"
    assert model == "gpt-4o"
    assert num_retries == 3
    assert "cache" not in cleaned
    assert "num_retries" not in cleaned
    assert "retry_strategy" not in cleaned
    assert "caching" not in cleaned
    assert cleaned["temperature"] == 0.5


def test_prepare_call_headers():
    kwargs = {"headers": {"X-Custom": "val"}, "api_key": "k"}
    _, _, _, _, _, _, cleaned = _prepare_call("openai/gpt-4o", kwargs)
    assert cleaned["extra_headers"] == {"X-Custom": "val"}
    assert "headers" not in cleaned


@mock.patch("litelm._completion.get_sync_client")
def test_completion_sync(mock_get_client):
    mock_client = mock.MagicMock()
    mock_client.chat.completions.create.return_value = _mock_completion()
    mock_get_client.return_value = mock_client

    result = completion("openai/gpt-4o", messages=[{"role": "user", "content": "hi"}], api_key="sk-test")
    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "Hi!"
    mock_client.chat.completions.create.assert_called_once()


@mock.patch("litelm._completion.get_async_client")
def test_acompletion(mock_get_client):
    mock_client = mock.MagicMock()

    async def mock_create(**kwargs):
        return _mock_completion()

    mock_client.chat.completions.create = mock.AsyncMock(return_value=_mock_completion())
    mock_get_client.return_value = mock_client

    result = asyncio.run(acompletion("openai/gpt-4o", messages=[{"role": "user", "content": "hi"}], api_key="sk-test"))
    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "Hi!"


# --- Error wrapping tests ---

def _make_openai_error(cls, msg="test error"):
    """Create an openai SDK error with minimal required args."""
    if cls in (openai.APITimeoutError, openai.APIConnectionError):
        return cls(request=mock.MagicMock())
    resp = mock.MagicMock()
    resp.status_code = 400
    return cls(message=msg, response=resp, body=None)


def test_map_openai_error_rate_limit():
    with pytest.raises(RateLimitError):
        _map_openai_error(_make_openai_error(openai.RateLimitError, "rate limited"))


def test_map_openai_error_auth():
    with pytest.raises(AuthenticationError):
        _map_openai_error(_make_openai_error(openai.AuthenticationError, "bad key"))


def test_map_openai_error_timeout():
    with pytest.raises(Timeout):
        _map_openai_error(_make_openai_error(openai.APITimeoutError))


def test_map_openai_error_bad_request_context_window():
    with pytest.raises(ContextWindowExceededError):
        _map_openai_error(_make_openai_error(openai.BadRequestError, "context length exceeded, too many tokens"))


def test_map_openai_error_bad_request_plain():
    with pytest.raises(BadRequestError):
        _map_openai_error(_make_openai_error(openai.BadRequestError, "invalid model"))


def test_map_openai_error_internal():
    with pytest.raises(InternalServerError):
        _map_openai_error(_make_openai_error(openai.InternalServerError, "server error"))


@mock.patch("litelm._completion.get_sync_client")
def test_completion_wraps_rate_limit(mock_get_client):
    """completion() wraps openai.RateLimitError → litelm.RateLimitError."""
    mock_client = mock.MagicMock()
    mock_client.chat.completions.create.side_effect = _make_openai_error(openai.RateLimitError, "rate limited")
    mock_get_client.return_value = mock_client

    with pytest.raises(RateLimitError):
        completion("openai/gpt-4o", messages=[{"role": "user", "content": "hi"}], api_key="sk-test")
