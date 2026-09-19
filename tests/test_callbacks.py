"""Tests for LiteLLM-compatible completion callbacks."""

from __future__ import annotations

import asyncio
import datetime
import logging
from unittest import mock

import openai
import pytest

import litelm
from litelm import _callbacks
from litelm._exceptions import AuthenticationError, RateLimitError


@pytest.fixture(autouse=True)
def _clear_callbacks():
    """Every test starts with empty canonical and legacy registries."""
    original_success = litelm.success_callback
    original_failure = litelm.failure_callback
    litelm.success_callback = []
    litelm.failure_callback = []
    litelm.success_callbacks.clear()
    yield
    litelm.success_callback = original_success
    litelm.failure_callback = original_failure
    original_success.clear()
    original_failure.clear()
    litelm.success_callbacks.clear()


def _messages():
    return [{"role": "user", "content": "hi"}]


def _openai_error(cls, message="provider failed"):
    response = mock.MagicMock()
    response.status_code = 429
    return cls(message=message, response=response, body=None)


def test_success_callback_uses_litellm_signature_and_supports_assignment():
    received = []
    litelm.success_callback = [lambda *args: received.append(args)]

    response = litelm.completion("openai/gpt-4o-mini", messages=_messages(), mock_response="hello")

    assert len(received) == 1
    kwargs, response_obj, start_time, end_time = received[0]
    assert kwargs["model"] == "openai/gpt-4o-mini"
    assert kwargs["messages"] == _messages()
    assert kwargs["stream"] is False
    assert kwargs["litellm_params"]["custom_llm_provider"] == "openai"
    assert response_obj is response
    assert isinstance(start_time, datetime.datetime)
    assert isinstance(end_time, datetime.datetime)
    assert end_time >= start_time


def test_legacy_success_callbacks_event_dict_still_works():
    received = []
    litelm.success_callbacks.append(received.append)

    response = litelm.completion("openai/gpt-4o-mini", messages=_messages(), mock_response="hello")

    assert len(received) == 1
    assert received[0]["model"] == "openai/gpt-4o-mini"
    assert received[0]["provider"] == "openai"
    assert received[0]["response"] is response
    assert received[0]["latency_ms"] >= 0.0
    assert received[0]["stream"] is False


def test_failure_callback_receives_same_mapped_sdk_exception(monkeypatch):
    client = mock.MagicMock()
    client.chat.completions.create.side_effect = _openai_error(openai.RateLimitError)
    monkeypatch.setattr("litelm._completion.get_sync_client", lambda *args, **kwargs: client)
    received = []
    litelm.failure_callback = [lambda *args: received.append(args)]

    with pytest.raises(RateLimitError) as raised:
        litelm.completion("openai/gpt-4o-mini", messages=_messages(), api_key="sk-test")

    assert len(received) == 1
    kwargs, response_obj, start_time, end_time = received[0]
    assert kwargs["exception"] is raised.value
    assert kwargs["model"] == "openai/gpt-4o-mini"
    assert kwargs["litellm_params"]["custom_llm_provider"] == "openai"
    assert response_obj is None
    assert end_time >= start_time


def test_async_failure_callback_covers_custom_handler(monkeypatch):
    expected = AuthenticationError("denied")

    class Handler:
        async def acompletion(self, *args, **kwargs):
            raise expected

    monkeypatch.setattr("litelm._completion.get_handler", lambda provider: Handler())
    received = []
    litelm.failure_callback.append(lambda *args: received.append(args))

    with pytest.raises(AuthenticationError) as raised:
        asyncio.run(litelm.acompletion("anthropic/claude-test", messages=_messages()))

    assert raised.value is expected
    assert len(received) == 1
    assert received[0][0]["exception"] is expected
    assert received[0][0]["litellm_params"]["custom_llm_provider"] == "anthropic"


def test_failure_callback_covers_provider_preparation(monkeypatch):
    expected = ValueError("bad model")
    monkeypatch.setattr("litelm._completion._prepare_call", mock.Mock(side_effect=expected))
    received = []
    litelm.failure_callback.append(lambda *args: received.append(args))

    with pytest.raises(ValueError) as raised:
        litelm.completion("broken", messages=_messages())

    assert raised.value is expected
    assert len(received) == 1
    assert received[0][0]["exception"] is expected
    assert received[0][0]["litellm_params"]["custom_llm_provider"] == "openai"


def test_failure_callbacks_run_in_order_and_cannot_mask_error(caplog):
    expected = ValueError("provider failed")
    order = []

    def broken(*args):
        order.append("broken")
        raise RuntimeError("observer failed")

    def healthy(*args):
        order.append("healthy")
        assert args[0]["exception"] is expected

    litelm.failure_callback.extend([broken, healthy])
    caplog.set_level(logging.WARNING, logger="litelm._callbacks")

    with mock.patch("litelm._completion._prepare_call", side_effect=expected):
        with pytest.raises(ValueError) as raised:
            litelm.completion("broken", messages=_messages())

    assert raised.value is expected
    assert order == ["broken", "healthy"]
    assert any("failure_callback raised" in record.getMessage() for record in caplog.records)


def test_streaming_callbacks_remain_out_of_scope():
    successes = []
    failures = []
    litelm.success_callback.append(lambda *args: successes.append(args))
    litelm.failure_callback.append(lambda *args: failures.append(args))

    stream = litelm.completion("openai/gpt-4o-mini", messages=_messages(), mock_response="hello", stream=True)
    list(stream)

    assert successes == []
    assert failures == []


def test_callback_dispatch_short_circuits_empty_registries(monkeypatch):
    callback_kwargs = {"model": "x"}
    monkeypatch.setattr(_callbacks, "_invoke", mock.Mock())

    _callbacks.fire_success(callback_kwargs, None, datetime.datetime.now(), datetime.datetime.now())
    _callbacks.fire_failure(callback_kwargs, None, datetime.datetime.now(), datetime.datetime.now())

    _callbacks._invoke.assert_not_called()


def test_completion_does_not_build_callback_payload_without_observers(monkeypatch):
    build_details = mock.Mock()
    monkeypatch.setattr("litelm._completion._callback_call_details", build_details)

    response = litelm.completion("openai/gpt-4o-mini", messages=_messages(), mock_response="hello")

    assert response is not None
    build_details.assert_not_called()
