"""Tests for success_callbacks."""

from __future__ import annotations

import asyncio
import logging

import pytest

import litelm
from litelm._callbacks import fire_success, success_callbacks


@pytest.fixture(autouse=True)
def _clear_callbacks():
    """Every test starts with an empty registry."""
    success_callbacks.clear()
    yield
    success_callbacks.clear()


def test_fire_success_invokes_each_callback_in_order():
    order: list[str] = []
    success_callbacks.append(lambda ev: order.append("a"))
    success_callbacks.append(lambda ev: order.append("b"))

    fire_success({"model": "x", "provider": "y", "response": None, "latency_ms": 1.0, "stream": False})

    assert order == ["a", "b"]


def test_fire_success_isolates_callback_exceptions(caplog):
    calls: list[str] = []

    def bad(ev):
        raise RuntimeError("boom")

    def good(ev):
        calls.append("reached")

    success_callbacks.append(bad)
    success_callbacks.append(good)

    caplog.set_level(logging.WARNING, logger="litelm._callbacks")
    fire_success({"model": "x", "provider": "y", "response": None, "latency_ms": 1.0, "stream": False})

    assert calls == ["reached"]
    assert any("success_callback raised" in r.getMessage() for r in caplog.records)


def test_completion_fires_success_for_mock_response():
    received: list[dict] = []
    success_callbacks.append(lambda ev: received.append(ev))

    response = litelm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello",
    )

    assert len(received) == 1
    event = received[0]
    assert event["model"] == "openai/gpt-4o-mini"
    assert event["provider"] == "openai"
    assert event["response"] is response
    assert event["latency_ms"] >= 0.0
    assert event["stream"] is False


def test_acompletion_fires_success_for_mock_response():
    received: list[dict] = []
    success_callbacks.append(lambda ev: received.append(ev))

    response = asyncio.run(
        litelm.acompletion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="hello",
        )
    )

    assert len(received) == 1
    event = received[0]
    assert event["model"] == "openai/gpt-4o-mini"
    assert event["response"] is response


def test_completion_with_no_callbacks_does_not_build_event():
    """Fast path: when registry is empty, no event dict is constructed."""
    # Not directly observable as a side-effect, but exercising the path
    # ensures no regression in the short-circuit.
    assert success_callbacks == []
    response = litelm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello",
    )
    assert response is not None


def test_stream_does_not_fire_success():
    """Streaming callbacks are deliberately out of scope in 0.5.0 — usage
    data only lands in the final chunk, and per-call firing semantics are
    ambiguous. Document the behavior via test."""
    received: list[dict] = []
    success_callbacks.append(lambda ev: received.append(ev))

    stream = litelm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello",
        stream=True,
    )
    # Exhaust the generator so any hypothetical on-stream-end callback
    # would fire before we assert.
    for _ in stream:
        pass

    assert received == []
