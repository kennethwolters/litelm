"""Mistral handler — OpenAI-compatible with message/response transforms.

Transforms:
  - Request: strip 'name' from non-tool messages (Mistral rejects it)
  - Response: convert empty string content "" to None
"""

from litelm._client_cache import get_async_client, get_sync_client
from litelm._types import ModelResponse, ModelResponseStream


def _transform_messages(messages):
    """Strip 'name' field from non-tool messages."""
    if not messages:
        return messages
    result = []
    for msg in messages:
        msg = dict(msg)
        if msg.get("role") != "tool" and "name" in msg:
            del msg["name"]
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


def completion(model_name, messages, *, stream=False, api_key=None, base_url=None, timeout=None, **kwargs):
    """Synchronous Mistral completion with message transforms."""
    messages = _transform_messages(messages)
    client = get_sync_client("mistral", base_url, api_key)
    sdk_kwargs = dict(model=model_name, messages=messages, stream=stream, **kwargs)
    if timeout is not None:
        sdk_kwargs["timeout"] = timeout
    response = client.chat.completions.create(**sdk_kwargs)
    if stream:
        return _wrap_stream_sync(response)
    _fix_response(response)
    return ModelResponse(response)


async def acompletion(model_name, messages, *, stream=False, api_key=None, base_url=None, timeout=None, **kwargs):
    """Async Mistral completion with message transforms."""
    messages = _transform_messages(messages)
    client = get_async_client("mistral", base_url, api_key)
    sdk_kwargs = dict(model=model_name, messages=messages, stream=stream, **kwargs)
    if timeout is not None:
        sdk_kwargs["timeout"] = timeout
    response = await client.chat.completions.create(**sdk_kwargs)
    if stream:
        return _wrap_stream_async(response)
    _fix_response(response)
    return ModelResponse(response)


def _wrap_stream_sync(stream):
    for chunk in stream:
        yield ModelResponseStream(chunk)


async def _wrap_stream_async(stream):
    async for chunk in stream:
        yield ModelResponseStream(chunk)
