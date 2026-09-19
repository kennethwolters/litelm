"""LiteLLM-compatible completion callback registries.

LiteLLM's canonical public registries are the singular ``success_callback``
and ``failure_callback`` lists. Callable entries receive four positional
arguments: call kwargs, response object, start time, and end time.

``success_callbacks`` remains as a legacy litelm registry for the one-event-
dict API released in 0.5.0. It is separate because that callback signature is
not compatible with LiteLLM's.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)

Callback = Callable[[dict[str, Any], Any, datetime, datetime], None]

success_callback: list[Callback] = []
failure_callback: list[Callback] = []
success_callbacks: list[Callable[[dict[str, Any]], None]] = []


def _public_registry(name: str, fallback: list[Callback]) -> list[Callback]:
    """Read the public registry dynamically so module-level assignment works."""
    module = sys.modules.get("litelm")
    registry = getattr(module, name, fallback) if module is not None else fallback
    return registry if isinstance(registry, list) else fallback


def has_success_callbacks() -> bool:
    """Return whether canonical or legacy success observers are registered."""
    return bool(_public_registry("success_callback", success_callback) or success_callbacks)


def has_failure_callbacks() -> bool:
    """Return whether failure observers are registered."""
    return bool(_public_registry("failure_callback", failure_callback))


def _invoke(callbacks: list[Callback], args: tuple[Any, ...], name: str) -> None:
    for callback in callbacks:
        try:
            callback(*args)
        except Exception as exc:
            log.warning("litelm %s raised: %s", name, exc, exc_info=True)


def fire_success(call_kwargs: dict[str, Any], response: Any, start_time: datetime, end_time: datetime) -> None:
    """Invoke canonical and legacy success callbacks without propagating errors."""
    callbacks = _public_registry("success_callback", success_callback)
    if callbacks:
        _invoke(callbacks, (call_kwargs, response, start_time, end_time), "success_callback")

    if success_callbacks:
        provider = call_kwargs.get("litellm_params", {}).get("custom_llm_provider", "unknown")
        event = {
            "model": call_kwargs.get("model", ""),
            "provider": provider,
            "response": response,
            "latency_ms": (end_time - start_time).total_seconds() * 1000,
            "stream": bool(call_kwargs.get("stream", False)),
        }
        for callback in success_callbacks:
            try:
                callback(event)
            except Exception as exc:
                log.warning("litelm success_callback raised: %s", exc, exc_info=True)


def fire_failure(call_kwargs: dict[str, Any], response: Any, start_time: datetime, end_time: datetime) -> None:
    """Invoke LiteLLM-compatible failure callbacks without propagating errors."""
    callbacks = _public_registry("failure_callback", failure_callback)
    if callbacks:
        _invoke(callbacks, (call_kwargs, response, start_time, end_time), "failure_callback")
