"""Tests for LiteLLM-compatible completion callbacks."""

from __future__ import annotations

import asyncio
import datetime
import logging
from unittest import mock

import httpx2
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


@pytest.mark.parametrize("failure", [False, True])
def test_suppression_prevents_all_observers_and_callback_payload_construction(monkeypatch, failure):
    observer, build = mock.Mock(), mock.Mock()
    litelm.success_callback = [observer]
    litelm.failure_callback = [observer]
    litelm.success_callbacks.append(observer)
    monkeypatch.setattr("litelm._completion._callback_call_details", build)
    if failure:
        monkeypatch.setattr("litelm._completion._prepare_call", mock.Mock(side_effect=ValueError("toy-error")))
    with litelm.suppress_callbacks():
        if failure:
            with pytest.raises(ValueError, match="toy-error"):
                litelm.completion("toy/model", messages=_messages(), api_key="toy-key")
        else:
            result = litelm.completion("toy/model", messages=_messages(), api_key="toy-key", mock_response="toy-answer")
            assert result.choices[0].message.content == "toy-answer"
    observer.assert_not_called()
    build.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])
def test_async_suppression_is_task_local_inherited_and_restored(monkeypatch, failure):
    observed = []
    litelm.success_callback.append(lambda details, *args: observed.append(details["messages"][0]["content"]))
    litelm.failure_callback.append(lambda details, *args: observed.append(details["messages"][0]["content"]))
    if failure:

        async def underlying(*args, **kwargs):
            await asyncio.sleep(0)
            raise ValueError("toy-error")

        monkeypatch.setattr("litelm._completion._acompletion", underlying)

    async def call(label):
        try:
            result = await litelm.acompletion(
                "toy/model",
                messages=[{"role": "user", "content": label}],
                api_key="toy-key",
                mock_response="toy-answer",
            )
            assert not failure and result.choices[0].message.content == "toy-answer"
        except ValueError as error:
            assert failure and str(error) == "toy-error"

    async def main():
        active, other_finished = asyncio.Event(), asyncio.Event()

        async def private():
            try:
                with litelm.suppress_callbacks():
                    with litelm.suppress_callbacks():
                        await call("private-nested")
                    await asyncio.create_task(call("private-child"))
                    active.set()
                    await other_finished.wait()
                    await call("private-after-other")
                    raise RuntimeError("scope-exit")
            except RuntimeError:
                pass
            await call("restored")

        async def other():
            await active.wait()
            await call("public")
            other_finished.set()

        await asyncio.gather(private(), other())

    asyncio.run(main())
    assert observed == ["public", "restored"]


@pytest.mark.parametrize("failure", [False, True])
async def test_suppression_through_real_async_sdk_and_borrowed_transport(failure):
    observer = mock.Mock()
    litelm.success_callback.append(observer)
    litelm.failure_callback.append(observer)
    litelm.success_callbacks.append(observer)
    sent = []

    async def handle(request):
        sent.append(request)
        if failure:
            return httpx2.Response(503, json={"error": {"message": "toy-error"}})
        return httpx2.Response(
            200,
            json={
                "id": "toy",
                "model": "toy",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "toy-answer"}}
                ],
            },
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle), trust_env=False) as client:
        with litelm.suppress_callbacks():
            if failure:
                with pytest.raises(litelm.InternalServerError):
                    await litelm.acompletion(
                        "openai/toy",
                        messages=_messages(),
                        api_key="toy-key",
                        api_base="https://example.invalid/v1",
                        http_client=client,
                    )
            else:
                result = await litelm.acompletion(
                    "openai/toy",
                    messages=_messages(),
                    api_key="toy-key",
                    api_base="https://example.invalid/v1",
                    http_client=client,
                )
                assert result.choices[0].message.content == "toy-answer"
        assert not client.is_closed
    assert len(sent) == 1
    observer.assert_not_called()


async def test_cancellation_restores_suppression_policy_in_the_same_task(monkeypatch):
    observer = mock.Mock()
    litelm.success_callback.append(observer)
    entered = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("litelm._completion._acompletion", blocked)

    async def private():
        try:
            with litelm.suppress_callbacks():
                await litelm.acompletion("toy/model", messages=_messages(), api_key="toy-key")
        except asyncio.CancelledError:
            assert _callbacks.has_success_callbacks()
            raise

    task = asyncio.create_task(private())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _callbacks.has_success_callbacks()
    observer.assert_not_called()


def test_suppressed_dispatch_covers_observers_registered_inside_scope():
    observer = mock.Mock()
    with litelm.suppress_callbacks():
        litelm.success_callback.append(observer)
        litelm.failure_callback.append(observer)
        litelm.success_callbacks.append(observer)
        now = datetime.datetime.now()
        _callbacks.fire_success({}, None, now, now)
        _callbacks.fire_failure({}, None, now, now)
    observer.assert_not_called()
    _callbacks.fire_success({}, None, now, now)
    assert observer.call_count == 2


def test_completion_does_not_build_callback_payload_without_observers(monkeypatch):
    build_details = mock.Mock()
    monkeypatch.setattr("litelm._completion._callback_call_details", build_details)

    response = litelm.completion("openai/gpt-4o-mini", messages=_messages(), mock_response="hello")

    assert response is not None
    build_details.assert_not_called()
