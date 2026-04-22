"""Observability callback registry.

Callers append functions to `success_callbacks` (module-level list) to
receive per-completion telemetry. Callbacks fire synchronously in
registration order after every successful non-streaming completion or
acompletion; streaming responses are not yet covered (usage data only
lands in the final chunk, so per-call firing is ambiguous).

Event schema:

    {
        "model": str,              # model string as invoked
        "provider": str,           # resolved provider ("openai", "azure", ...)
        "response": ModelResponse, # the completion; read usage/id off this
        "latency_ms": float,       # wall-clock between call start and return
        "stream": bool,            # always False in this release
    }

Exceptions raised inside a callback are logged and swallowed — a broken
observer must not cascade into a completion failure for unrelated
callers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)


success_callbacks: list[Callable[[dict[str, Any]], None]] = []


def fire_success(event: dict[str, Any]) -> None:
    """Invoke every registered success callback with `event`.

    Never raises; callback exceptions are logged at WARNING.
    """
    for cb in success_callbacks:
        try:
            cb(event)
        except Exception as exc:
            log.warning("litelm success_callback raised: %s", exc, exc_info=True)
