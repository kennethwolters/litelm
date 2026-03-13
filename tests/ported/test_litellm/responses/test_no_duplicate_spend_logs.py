"""
Test that responses() API does not create duplicate spend logs.

This test verifies the fix for issue #15740 where kwargs.pop() was removing
the logging object before passing kwargs to internal acompletion() calls,
causing duplicate spend log entries for non-OpenAI providers.
"""
import asyncio
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litelm
from litelm.integrations.custom_logger import CustomLogger


def test_logging_object_not_popped():
    """
    Test that litelm_logging_obj is not popped from kwargs.

    This is a regression test for issue #15740. The bug was using
    kwargs.pop() which removed the logging object, causing duplicate
    spend logs for non-OpenAI providers.
    """
    import inspect

    from litelm.responses import main as responses_module

    # Get the source code of the responses function
    source = inspect.getsource(responses_module.responses)

    # Check that .pop("litelm_logging_obj") is NOT used
    # The bug was using kwargs.pop("litelm_logging_obj") which removes it
    assert 'kwargs.pop("litelm_logging_obj")' not in source, (
        "FAIL: Found kwargs.pop('litelm_logging_obj') in responses() function. "
        "This causes duplicate spend logs. Use kwargs.get('litelm_logging_obj') instead."
    )

    # Check that .get("litelm_logging_obj") IS used
    assert 'kwargs.get("litelm_logging_obj")' in source, (
        "FAIL: Expected kwargs.get('litelm_logging_obj') but not found. "
        "The logging object must be accessed with .get() not .pop() to prevent duplication."
    )


@pytest.mark.asyncio
async def test_async_no_duplicate_spend_logs():
    """
    Test that spend logs are only created once, not duplicated.

    This integration test verifies the fix by using a custom logger
    that counts log_success_event calls for a specific request ID.
    Before the fix, it would be called twice for non-OpenAI providers.
    """
    import uuid

    # Generate a unique ID to track only this test's request
    test_request_id = f"test-no-dup-{uuid.uuid4()}"

    # Create a custom logger to count log_success_event calls for our specific request
    class SpendLogCounter(CustomLogger):
        def __init__(self, tracking_id: str):
            super().__init__()
            self.tracking_id = tracking_id
            self.log_count = 0

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            # Only count logs for our specific test request
            litelm_call_id = kwargs.get("litelm_call_id", "")
            if litelm_call_id == self.tracking_id:
                self.log_count += 1

    spend_logger = SpendLogCounter(tracking_id=test_request_id)

    # Save original callbacks and append our logger (don't replace to avoid affecting other tests)
    original_callbacks = litelm.callbacks.copy() if litelm.callbacks else []
    litelm.callbacks = original_callbacks + [spend_logger]

    try:
        # Call responses API with Anthropic model using mock_response
        # Pass our unique ID as litelm_call_id to track this specific request
        response = await litelm.aresponses(
            model="anthropic/claude-3-7-sonnet-latest",
            input=[{
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
                "type": "message"
            }],
            instructions="You are a helpful assistant.",
            mock_response="Hello! I'm doing well.",
            litelm_call_id=test_request_id,
        )

        # Yield to the event loop so the _client_async_logging_helper task
        # (scheduled via asyncio.create_task in the @client decorator) runs first
        # and initializes GLOBAL_LOGGING_WORKER on the current event loop.
        # Without this, flush() may block on a stale queue from a previous test's loop.
        await asyncio.sleep(0)

        # Wait for async logging to complete. Use a timeout so that if the
        # worker is on a stale event loop (common in CI), flush() doesn't hang
        # indefinitely — the queue.join() inside flush() would never resolve.
        from litelm.litelm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
        try:
            await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)
        except asyncio.TimeoutError:
            pass
        await asyncio.sleep(0.5)

        # Verify that log_success_event was called exactly once for our request
        assert spend_logger.log_count == 1, (
            f"FAIL: log_success_event called {spend_logger.log_count} times instead of 1 "
            f"for request {test_request_id}. This indicates duplicate spend logs."
        )

    finally:
        # Restore original callbacks
        litelm.callbacks = original_callbacks
