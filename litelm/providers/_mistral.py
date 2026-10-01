"""Mistral handler — OpenAI-compatible with message/response transforms.

Transforms:
  - Request: strip 'name' from non-tool messages (Mistral rejects it)
  - Response: convert empty string content "" to None
"""

from litelm._client_cache import get_async_client, get_sync_client
from litelm._completion import _map_openai_error, _openai_errors, _wrap_stream_async, _wrap_stream_sync
from litelm._types import ModelResponse


def _transform_messages(messages):
    """Strip fields Mistral rejects from replayed messages."""
    if not messages:
        return messages
    result = []
    for msg in messages:
        msg = dict(msg)
        if msg.get("role") != "tool" and "name" in msg:
            del msg["name"]
        if msg.get("role") == "assistant":
            msg.pop("reasoning_content", None)
            msg.pop("thinking_blocks", None)
        result.append(msg)
    return result


def _fix_response(response):
    """Fix Mistral response quirks: empty content → None, tool_call type → 'function'."""
    if hasattr(response, "choices") and response.choices:
        for choice in response.choices:
            msg = getattr(choice, "message", None)
            if msg is None:
                continue
            if getattr(msg, "content", None) == "":
                msg.content = None
            for tc in getattr(msg, "tool_calls", None) or []:
                if getattr(tc, "type", None) is None:
                    tc.type = "function"
    return response


def completion(
    model_name, messages, *, stream=False, api_key=None, base_url=None, timeout=None, max_retries=0, **kwargs
):
    """Synchronous Mistral completion with message transforms."""
    messages = _transform_messages(messages)
    client = get_sync_client("mistral", base_url, api_key, max_retries=max_retries)
    sdk_kwargs = dict(model=model_name, messages=messages, stream=stream, **kwargs)
    if timeout is not None:
        sdk_kwargs["timeout"] = timeout
    try:
        response = client.chat.completions.create(**sdk_kwargs)
    except _openai_errors as e:
        _map_openai_error(e)
    if stream:
        return _wrap_stream_sync(response)
    _fix_response(response)
    return ModelResponse(response)


async def acompletion(
    model_name, messages, *, stream=False, api_key=None, base_url=None, timeout=None, max_retries=0, **kwargs
):
    """Async Mistral completion with message transforms."""
    messages = _transform_messages(messages)
    client = get_async_client("mistral", base_url, api_key, max_retries=max_retries)
    sdk_kwargs = dict(model=model_name, messages=messages, stream=stream, **kwargs)
    if timeout is not None:
        sdk_kwargs["timeout"] = timeout
    try:
        response = await client.chat.completions.create(**sdk_kwargs)
    except _openai_errors as e:
        _map_openai_error(e)
    if stream:
        return _wrap_stream_async(response)
    _fix_response(response)
    return ModelResponse(response)
