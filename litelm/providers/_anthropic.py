"""Anthropic native API handler using the anthropic SDK.

Translates between OpenAI-style messages/params and the Anthropic Messages API,
then wraps responses back into OpenAI-compatible ModelResponse objects.
"""

import json
import os
import time

from litelm._exceptions import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    Timeout,
    UnprocessableEntityError,
    is_context_window_error,
)
from litelm._types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChunkChoice,
    CompletionUsage,
    Function,
    ModelResponse,
    ModelResponseStream,
)

_SDK = None


def _get_sdk():
    global _SDK
    if _SDK is None:
        try:
            import anthropic

            _SDK = anthropic
        except ImportError:
            raise ImportError(
                "The anthropic SDK is required for anthropic/ models. Install it with: pip install litelm[anthropic]"
            )
    return _SDK


# ---------------------------------------------------------------------------
# Request translation: OpenAI → Anthropic
# ---------------------------------------------------------------------------


def _extract_system(messages):
    """Separate system messages from the conversation."""
    system_parts = []
    conversation = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                if content:
                    system_parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, str):
                        if block:
                            system_parts.append({"type": "text", "text": block})
                    elif isinstance(block, dict):
                        if block.get("type") == "text" and not block.get("text"):
                            continue
                        system_parts.append(block)
        else:
            conversation.append(msg)
    return system_parts or None, conversation


def _translate_content(content):
    """Translate OpenAI content (string or list of parts) to Anthropic content blocks."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    blocks = []
    for part in content:
        if isinstance(part, str):
            blocks.append({"type": "text", "text": part})
        elif isinstance(part, dict):
            ptype = part.get("type", "text")
            if ptype == "text":
                text = part.get("text", "")
                if not text and "cache_control" not in part:
                    continue
                block = {"type": "text", "text": text}
                if "cache_control" in part:
                    block["cache_control"] = part["cache_control"]
                blocks.append(block)
            elif ptype == "image_url":
                url_data = part.get("image_url", {})
                url = url_data.get("url", "") if isinstance(url_data, dict) else url_data
                if url.startswith("data:"):
                    media_type, _, b64_data = url.partition(";base64,")
                    media_type = media_type.replace("data:", "")
                    block = {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_data,
                        },
                    }
                else:
                    block = {
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    }
                if "cache_control" in part:
                    block["cache_control"] = part["cache_control"]
                blocks.append(block)
            else:
                blocks.append(part)
    return blocks


def _translate_messages(messages):
    """Translate OpenAI messages list to Anthropic format."""
    result = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "assistant":
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            blocks = _translate_content(content) if content else []
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    args_str = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_str)
                    except (json.JSONDecodeError, TypeError):
                        args = {"raw": args_str}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": args,
                        }
                    )
            result.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")
            result.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": content if isinstance(content, str) else json.dumps(content),
                        }
                    ],
                }
            )
        elif role == "user":
            blocks = _translate_content(msg.get("content"))
            result.append({"role": "user", "content": blocks})
        else:
            blocks = _translate_content(msg.get("content"))
            result.append({"role": role, "content": blocks})

    # Anthropic requires alternating user/assistant. Merge consecutive same-role.
    merged = []
    for msg in result:
        if merged and merged[-1]["role"] == msg["role"]:
            prev_content = merged[-1]["content"]
            cur_content = msg["content"]
            if isinstance(prev_content, list) and isinstance(cur_content, list):
                merged[-1]["content"] = prev_content + cur_content
            elif isinstance(prev_content, str) and isinstance(cur_content, str):
                merged[-1]["content"] = prev_content + "\n" + cur_content
            else:
                merged.append(msg)
        else:
            merged.append(msg)
    return merged


_UNSUPPORTED_SCHEMA_FIELDS = frozenset({
    "maxItems", "minItems", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength",
})


def _filter_schema(schema):
    """Recursively strip JSON schema fields unsupported by Anthropic, appending them to description."""
    if not isinstance(schema, dict):
        return schema
    result = {}
    constraint_labels = []
    for key, value in schema.items():
        if key in _UNSUPPORTED_SCHEMA_FIELDS:
            constraint_labels.append(f"{key}: {value}")
        elif key == "properties" and isinstance(value, dict):
            result[key] = {k: _filter_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            result[key] = _filter_schema(value)
        elif key in ("$defs", "definitions") and isinstance(value, dict):
            result[key] = {k: _filter_schema(v) for k, v in value.items()}
        elif key in ("anyOf", "allOf") and isinstance(value, list):
            result[key] = [_filter_schema(item) if isinstance(item, dict) else item for item in value]
        else:
            result[key] = value
    if constraint_labels:
        desc = result.get("description", "")
        note = "Note: " + ", ".join(constraint_labels) + "."
        result["description"] = f"{desc} {note}".strip() if desc else note
    return result


def _translate_tools(tools):
    """Translate OpenAI tools format to Anthropic tools format."""
    if not tools:
        return None
    anthropic_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool["function"]
            tool_def = {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": _filter_schema(fn.get("parameters", {"type": "object", "properties": {}})),
            }
            if "cache_control" in tool:
                tool_def["cache_control"] = tool["cache_control"]
            anthropic_tools.append(tool_def)
    return anthropic_tools or None


def _translate_tool_choice(tool_choice):
    """Translate OpenAI tool_choice to Anthropic tool_choice."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"type": "auto"}
        elif tool_choice == "none":
            return {"type": "auto"}
        elif tool_choice == "required":
            return {"type": "any"}
    elif isinstance(tool_choice, dict):
        fn = tool_choice.get("function", {})
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            return {"type": "tool", "name": name}
    return None


_MAX_OUTPUT_TOKENS = {
    "claude-opus-4": 32768,
    "claude-4-opus": 32768,
    "claude-sonnet-4": 16384,
    "claude-4-sonnet": 16384,
    "claude-3-7-sonnet": 16384,
    "claude-3-5-sonnet": 8192,
    "claude-3-5-haiku": 8192,
    "claude-3-opus": 4096,
    "claude-3-sonnet": 4096,
    "claude-3-haiku": 4096,
}
_DEFAULT_MAX_TOKENS = 4096


def _get_max_tokens(model_name):
    for prefix, tokens in _MAX_OUTPUT_TOKENS.items():
        if model_name.startswith(prefix):
            return tokens
    return _DEFAULT_MAX_TOKENS


def _is_opus_4_6(model):
    """Check if a model is Claude Opus 4.6."""
    m = model.lower().replace("_", "-").replace(".", "-")
    return "opus-4-6" in m


_REASONING_EFFORT_BUDGET = {
    "minimal": 128,
    "low": 1024,
    "medium": 2048,
    "high": 4096,
}


def _map_reasoning_effort(reasoning_effort, model_name):
    """Map OpenAI reasoning_effort to Anthropic thinking/output_config params.

    Returns (thinking_dict_or_None, output_config_or_None).
    """
    if not reasoning_effort or str(reasoning_effort).lower() == "none":
        return None, None

    effort = str(reasoning_effort).lower()

    if _is_opus_4_6(model_name):
        thinking = {"type": "adaptive"}
        output_config = {"effort": effort} if effort in ("low", "medium", "high") else None
        return thinking, output_config

    budget = _REASONING_EFFORT_BUDGET.get(effort)
    if budget is None:
        return None, None
    return {"type": "enabled", "budget_tokens": budget}, None


def _build_request_kwargs(model_name, messages, stream, api_key, base_url, **kwargs):
    """Build kwargs dict for the Anthropic SDK create call."""
    system, conversation = _extract_system(messages)
    translated = _translate_messages(conversation)

    req = {
        "model": model_name,
        "messages": translated,
        "stream": stream,
    }

    if system:
        req["system"] = system

    max_tokens = kwargs.pop("max_tokens", None) or kwargs.pop("max_completion_tokens", None) or _get_max_tokens(model_name)
    req["max_tokens"] = max_tokens

    for key in ("temperature", "top_p", "stop"):
        if key in kwargs:
            val = kwargs.pop(key)
            if val is not None:
                if key == "stop":
                    req["stop_sequences"] = val if isinstance(val, list) else [val]
                else:
                    req[key] = val

    tools = kwargs.pop("tools", None)
    anthropic_tools = _translate_tools(tools)
    if anthropic_tools:
        req["tools"] = anthropic_tools

    tool_choice = kwargs.pop("tool_choice", None)
    tc = _translate_tool_choice(tool_choice)
    if tc:
        req["tool_choice"] = tc

    response_format = kwargs.pop("response_format", None)
    if response_format and isinstance(response_format, dict):
        rf_type = response_format.get("type")
        if rf_type == "json_object":
            if translated and translated[-1]["role"] == "assistant":
                pass
            else:
                req["messages"].append({"role": "assistant", "content": [{"type": "text", "text": "{"}]})

    thinking = kwargs.pop("thinking", None)
    if thinking:
        req["thinking"] = thinking

    reasoning_effort = kwargs.pop("reasoning_effort", None)
    if reasoning_effort and not thinking:
        thinking_param, output_config = _map_reasoning_effort(reasoning_effort, model_name)
        if thinking_param:
            req["thinking"] = thinking_param
        if output_config:
            req["output_config"] = output_config

    for drop in (
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "logprobs",
        "top_logprobs",
        "n",
        "extra_headers",
        "response_format",
        "reasoning_effort",
    ):
        kwargs.pop(drop, None)

    extra_headers = kwargs.pop("extra_headers", None)
    if extra_headers:
        req["extra_headers"] = extra_headers

    req.update(kwargs)
    return req


def _get_client(api_key, base_url, async_client=False):
    """Create an Anthropic client."""
    sdk = _get_sdk()
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    kwargs = {"api_key": key}
    if base_url and base_url != "https://api.anthropic.com":
        kwargs["base_url"] = base_url
    cls = sdk.AsyncAnthropic if async_client else sdk.Anthropic
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Response translation: Anthropic → OpenAI
# ---------------------------------------------------------------------------


def _build_model_response(response):
    """Convert an Anthropic Message to a ModelResponse (OpenAI ChatCompletion format)."""
    content_text = ""
    tool_calls = []
    reasoning_content = ""
    thinking_blocks = []

    for block in response.content:
        if block.type == "text":
            content_text += block.text
        elif block.type == "tool_use":
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=block.id,
                    type="function",
                    function=Function(
                        name=block.name,
                        arguments=json.dumps(block.input) if isinstance(block.input, dict) else str(block.input),
                    ),
                )
            )
        elif block.type == "thinking":
            reasoning_content += block.thinking
            thinking_blocks.append({
                "type": "thinking",
                "thinking": block.thinking,
                "signature": getattr(block, "signature", ""),
            })

    message = ChatCompletionMessage(
        role="assistant",
        content=content_text or None,
        tool_calls=tool_calls or None,
        reasoning_content=reasoning_content or None,
        thinking_blocks=thinking_blocks or None,
    )

    stop_reason_map = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }
    finish_reason = stop_reason_map.get(response.stop_reason, "stop")

    usage = CompletionUsage(
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens,
        total_tokens=response.usage.input_tokens + response.usage.output_tokens,
    )

    # Populate prompt_tokens_details from Anthropic cache tokens
    cache_creation = getattr(response.usage, "cache_creation_input_tokens", None)
    cache_read = getattr(response.usage, "cache_read_input_tokens", None)
    if cache_creation is not None or cache_read is not None:
        details = {}
        if cache_read is not None:
            details["cached_tokens"] = cache_read
        if cache_creation is not None:
            details["cache_creation_tokens"] = cache_creation
        usage.prompt_tokens_details = details

    completion = ChatCompletion(
        id=response.id,
        choices=[Choice(index=0, message=message, finish_reason=finish_reason)],
        created=int(time.time()),
        model=response.model,
        object="chat.completion",
        usage=usage,
    )
    return ModelResponse(completion)


def _build_stream_chunk(event, model, chunk_id):
    """Convert an Anthropic streaming event to a ModelResponseStream chunk."""
    delta_kwargs = {}
    finish_reason = None
    usage = None

    event_type = event.type

    if event_type == "content_block_start":
        block = event.content_block
        if block.type == "text":
            delta_kwargs["content"] = ""
            delta_kwargs["role"] = "assistant"
        elif block.type == "thinking":
            delta_kwargs["role"] = "assistant"
        elif block.type == "tool_use":
            delta_kwargs["tool_calls"] = [
                ChoiceDeltaToolCall(
                    index=event.index,
                    id=block.id,
                    type="function",
                    function=ChoiceDeltaToolCallFunction(name=block.name, arguments=""),
                )
            ]
    elif event_type == "content_block_delta":
        delta = event.delta
        if delta.type == "text_delta":
            delta_kwargs["content"] = delta.text
        elif delta.type == "thinking_delta":
            delta_kwargs["reasoning_content"] = delta.thinking
            delta_kwargs["thinking_blocks"] = [{"type": "thinking", "thinking": delta.thinking}]
        elif delta.type == "signature_delta":
            delta_kwargs["thinking_blocks"] = [{"type": "thinking", "signature": delta.signature}]
        elif delta.type == "input_json_delta":
            delta_kwargs["tool_calls"] = [
                ChoiceDeltaToolCall(
                    index=event.index,
                    id=None,
                    type="function",
                    function=ChoiceDeltaToolCallFunction(name=None, arguments=delta.partial_json),
                )
            ]
    elif event_type == "message_delta":
        stop_reason_map = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
        }
        finish_reason = stop_reason_map.get(event.delta.stop_reason, "stop")
        if hasattr(event, "usage") and event.usage:
            usage = CompletionUsage(
                prompt_tokens=0,
                completion_tokens=event.usage.output_tokens,
                total_tokens=event.usage.output_tokens,
            )
    elif event_type == "message_start":
        delta_kwargs["role"] = "assistant"
        if hasattr(event.message, "usage") and event.message.usage:
            usage = CompletionUsage(
                prompt_tokens=event.message.usage.input_tokens,
                completion_tokens=0,
                total_tokens=event.message.usage.input_tokens,
            )
    else:
        return None

    delta = ChoiceDelta(**delta_kwargs)
    choice = ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)

    chunk_kwargs = {
        "id": chunk_id,
        "choices": [choice],
        "created": int(time.time()),
        "model": model,
        "object": "chat.completion.chunk",
    }
    if usage:
        chunk_kwargs["usage"] = usage

    return ModelResponseStream(ChatCompletionChunk(**chunk_kwargs))


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def _map_error(e):
    """Map anthropic SDK exceptions to litelm exception types."""
    sdk = _get_sdk()
    msg = str(e)

    if isinstance(e, sdk.AuthenticationError):
        raise AuthenticationError(
            message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)
        ) from e
    elif isinstance(e, sdk.RateLimitError):
        raise RateLimitError(message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)) from e
    elif isinstance(e, sdk.BadRequestError):
        if is_context_window_error(msg):
            raise ContextWindowExceededError(
                message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)
            ) from e
        raise BadRequestError(message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)) from e
    elif isinstance(e, sdk.APITimeoutError):
        raise Timeout(request=getattr(e, "request", None)) from e
    elif isinstance(e, sdk.APIConnectionError):
        raise APIConnectionError(request=getattr(e, "request", None)) from e
    elif isinstance(e, sdk.InternalServerError):
        raise InternalServerError(
            message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)
        ) from e
    elif isinstance(e, sdk.NotFoundError):
        raise NotFoundError(message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)) from e
    elif isinstance(e, sdk.PermissionDeniedError):
        raise PermissionDeniedError(
            message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)
        ) from e
    elif isinstance(e, sdk.UnprocessableEntityError):
        raise UnprocessableEntityError(
            message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)
        ) from e
    elif isinstance(e, sdk.APIStatusError):
        raise APIStatusError(message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)) from e
    raise e


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def completion(model_name, messages, *, stream=False, api_key=None, base_url=None, timeout=None, **kwargs):
    """Synchronous Anthropic chat completion."""
    sdk = _get_sdk()
    client = _get_client(api_key, base_url)
    req = _build_request_kwargs(model_name, messages, stream, api_key, base_url, **kwargs)
    if timeout is not None:
        req["timeout"] = timeout

    try:
        if stream:
            return _stream_sync(client, req)
        else:
            response = client.messages.create(**req)
            return _build_model_response(response)
    except sdk.APIError as e:
        _map_error(e)


def _stream_sync(client, req):
    """Synchronous streaming generator."""
    sdk = _get_sdk()
    req["stream"] = True
    try:
        with client.messages.stream(**{k: v for k, v in req.items() if k != "stream"}) as stream:
            model = req["model"]
            chunk_id = f"chatcmpl-anthropic-{int(time.time())}"
            for event in stream:
                chunk = _build_stream_chunk(event, model, chunk_id)
                if chunk is not None:
                    yield chunk
    except sdk.APIError as e:
        _map_error(e)


async def acompletion(model_name, messages, *, stream=False, api_key=None, base_url=None, timeout=None, **kwargs):
    """Async Anthropic chat completion."""
    sdk = _get_sdk()
    client = _get_client(api_key, base_url, async_client=True)
    req = _build_request_kwargs(model_name, messages, stream, api_key, base_url, **kwargs)
    if timeout is not None:
        req["timeout"] = timeout

    try:
        if stream:
            return _stream_async(client, req)
        else:
            response = await client.messages.create(**req)
            return _build_model_response(response)
    except sdk.APIError as e:
        _map_error(e)


async def _stream_async(client, req):
    """Async streaming generator."""
    sdk = _get_sdk()
    req["stream"] = True
    try:
        async with client.messages.stream(**{k: v for k, v in req.items() if k != "stream"}) as stream:
            model = req["model"]
            chunk_id = f"chatcmpl-anthropic-{int(time.time())}"
            async for event in stream:
                chunk = _build_stream_chunk(event, model, chunk_id)
                if chunk is not None:
                    yield chunk
    except sdk.APIError as e:
        _map_error(e)
