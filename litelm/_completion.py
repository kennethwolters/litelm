"""Chat completion functions — openai SDK imported lazily only when needed."""

import time

from litelm._callbacks import fire_success, success_callbacks
from litelm._client_cache import get_async_client, get_sync_client
from litelm._dispatch import get_handler
from litelm._exceptions import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    InternalServerError,
    LitelmError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    Timeout,
    UnprocessableEntityError,
    is_context_window_error,
)
from litelm._providers import parse_model
from litelm._types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
    Function,
    ModelResponse,
    ModelResponseStream,
)

# Build tuple of all openai SDK error types to catch
_openai_errors: tuple = ()
_bad_request_errors = [BadRequestError]
try:
    import openai as _openai

    _bad_request_errors.append(_openai.BadRequestError)
    _openai_errors = (_openai.APIError,)
except ImportError:
    pass
_bad_request_errors = tuple(_bad_request_errors)


def _map_openai_error(e):
    """Map openai SDK exceptions to litelm exception types. Re-raises as litelm error."""
    try:
        import openai
    except ImportError:
        raise LitelmError(message=str(e)) from e

    msg = str(e)

    if isinstance(e, openai.BadRequestError):
        if is_context_window_error(msg):
            raise ContextWindowExceededError(
                message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)
            ) from e
        raise BadRequestError(message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)) from e
    elif isinstance(e, openai.RateLimitError):
        raise RateLimitError(message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)) from e
    elif isinstance(e, openai.AuthenticationError):
        raise AuthenticationError(
            message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)
        ) from e
    elif isinstance(e, openai.APITimeoutError):
        raise Timeout(request=getattr(e, "request", None)) from e
    elif isinstance(e, openai.APIConnectionError):
        raise APIConnectionError(request=getattr(e, "request", None)) from e
    elif isinstance(e, openai.InternalServerError):
        raise InternalServerError(
            message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)
        ) from e
    elif isinstance(e, openai.NotFoundError):
        raise NotFoundError(message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)) from e
    elif isinstance(e, openai.PermissionDeniedError):
        raise PermissionDeniedError(
            message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)
        ) from e
    elif isinstance(e, openai.UnprocessableEntityError):
        raise UnprocessableEntityError(
            message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)
        ) from e
    elif isinstance(e, openai.APIStatusError):
        raise APIStatusError(message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)) from e
    raise LitelmError(message=msg) from e


# kwargs that are litellm-specific and must be stripped before passing to OpenAI SDK
_LITELLM_ONLY_KWARGS = {"cache", "num_retries", "retry_strategy", "caching"}


def _prepare_call(model, kwargs):
    """Parse model, build client kwargs, strip litellm-specific params."""
    num_retries = kwargs.pop("num_retries", 0)
    kwargs.pop("retry_strategy", None)
    kwargs.pop("cache", None)
    kwargs.pop("caching", None)
    kwargs.pop("custom_llm_provider", None)
    kwargs.pop("fallbacks", None)
    kwargs.pop("mock_timeout", None)

    api_key = kwargs.pop("api_key", None)
    api_base = kwargs.pop("api_base", None) or kwargs.pop("base_url", None)
    api_version = kwargs.pop("api_version", None)
    azure_ad_token_provider = kwargs.pop("azure_ad_token_provider", None)
    headers = kwargs.pop("headers", None)

    provider, model_name, base_url, resolved_api_key, resolved_api_version = parse_model(
        model, api_key=api_key, api_base=api_base, api_version=api_version
    )

    if headers:
        kwargs["extra_headers"] = headers

    return (
        provider,
        model_name,
        base_url,
        resolved_api_key,
        resolved_api_version or api_version,
        num_retries,
        azure_ad_token_provider,
        kwargs,
    )


def _add_additional_properties_false(schema):
    """Recursively add additionalProperties: false to object schemas (required by OpenAI strict mode)."""
    if not isinstance(schema, dict):
        return schema
    if schema.get("type") == "object" and "properties" in schema:
        schema["additionalProperties"] = False
    for key in ("properties", "$defs", "definitions"):
        container = schema.get(key)
        if isinstance(container, dict):
            for v in container.values():
                _add_additional_properties_false(v)
    if "items" in schema and isinstance(schema["items"], dict):
        _add_additional_properties_false(schema["items"])
    for variant_key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(variant_key)
        if isinstance(variants, list):
            for v in variants:
                _add_additional_properties_false(v)
    return schema


def _normalize_response_format(response_format):
    """Convert pydantic BaseModel classes to OpenAI json_schema dicts."""
    if response_format is None or isinstance(response_format, (dict, str)):
        return response_format
    if isinstance(response_format, type):
        try:
            from pydantic import BaseModel

            if issubclass(response_format, BaseModel):
                schema = response_format.model_json_schema()
                _add_additional_properties_false(schema)
                return {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "schema": schema,
                        "strict": True,
                    },
                }
        except ImportError:
            pass
    return response_format


def _wrap_context_window_error(e):
    """Convert BadRequestError to ContextWindowExceededError if about context length."""
    if is_context_window_error(str(e)):
        raise ContextWindowExceededError(
            message=str(e),
            response=getattr(e, "response", None),
            body=getattr(e, "body", None),
        ) from e
    raise e


def _fire_completion_success(model, provider, response, start_time, stream):
    """Build the success event and dispatch to any registered callbacks.

    Short-circuit on an empty registry — avoid constructing the event dict
    (and especially the latency calc) when no observer is listening.
    """
    if not success_callbacks:
        return
    fire_success(
        {
            "model": model,
            "provider": provider,
            "response": response,
            "latency_ms": (time.monotonic() - start_time) * 1000,
            "stream": stream,
        }
    )


def completion(model, messages=None, *, timeout=None, stream=False, shared_session=None, **kwargs):
    """Synchronous chat completion."""
    start_time = time.monotonic()
    mock = kwargs.pop("mock_response", None)
    n = kwargs.pop("n", None) or 1
    (provider, model_name, base_url, api_key, api_version, num_retries, azure_ad_token_provider, kwargs) = (
        _prepare_call(model, kwargs)
    )
    if "response_format" in kwargs:
        kwargs["response_format"] = _normalize_response_format(kwargs["response_format"])

    if mock is not None:
        content = str(mock) if mock is not True else "This is a mock request"
        if stream:
            return _mock_stream_sync(content, model_name)
        result = ModelResponse(
            ChatCompletion(
                id="mock",
                choices=[
                    Choice(
                        index=i, message=ChatCompletionMessage(role="assistant", content=content), finish_reason="stop"
                    )
                    for i in range(n)
                ],
                created=0,
                model=model_name,
                object="chat.completion",
                usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )
        )
        _fire_completion_success(model, provider, result, start_time, False)
        return result

    if n > 1:
        kwargs["n"] = n

    handler = get_handler(provider)
    if handler:
        result = handler.completion(
            model_name, messages, stream=stream, api_key=api_key, base_url=base_url, timeout=timeout, **kwargs
        )
        if not stream:
            _fire_completion_success(model, provider, result, start_time, False)
        return result

    client = get_sync_client(
        provider,
        base_url,
        api_key,
        max_retries=num_retries,
        api_version=api_version,
        azure_ad_token_provider=azure_ad_token_provider,
    )

    try:
        sdk_kwargs = dict(model=model_name, messages=messages, stream=stream, **kwargs)
        if timeout is not None:
            sdk_kwargs["timeout"] = timeout
        response = client.chat.completions.create(**sdk_kwargs)
    except _bad_request_errors as e:
        _wrap_context_window_error(e)
    except _openai_errors as e:
        _map_openai_error(e)

    if stream:
        return _wrap_stream_sync(response)
    result = ModelResponse(response)
    _fire_completion_success(model, provider, result, start_time, False)
    return result


async def acompletion(model, messages=None, *, timeout=None, stream=False, shared_session=None, **kwargs):
    """Async chat completion."""
    start_time = time.monotonic()
    mock = kwargs.pop("mock_response", None)
    n = kwargs.pop("n", None) or 1
    (provider, model_name, base_url, api_key, api_version, num_retries, azure_ad_token_provider, kwargs) = (
        _prepare_call(model, kwargs)
    )
    if "response_format" in kwargs:
        kwargs["response_format"] = _normalize_response_format(kwargs["response_format"])

    if mock is not None:
        content = str(mock) if mock is not True else "This is a mock request"
        if stream:
            return _mock_stream_async(content, model_name)
        result = ModelResponse(
            ChatCompletion(
                id="mock",
                choices=[
                    Choice(
                        index=i, message=ChatCompletionMessage(role="assistant", content=content), finish_reason="stop"
                    )
                    for i in range(n)
                ],
                created=0,
                model=model_name,
                object="chat.completion",
                usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )
        )
        _fire_completion_success(model, provider, result, start_time, False)
        return result

    if n > 1:
        kwargs["n"] = n

    handler = get_handler(provider)
    if handler:
        result = await handler.acompletion(
            model_name, messages, stream=stream, api_key=api_key, base_url=base_url, timeout=timeout, **kwargs
        )
        if not stream:
            _fire_completion_success(model, provider, result, start_time, False)
        return result

    client = get_async_client(
        provider,
        base_url,
        api_key,
        max_retries=num_retries,
        api_version=api_version,
        azure_ad_token_provider=azure_ad_token_provider,
    )

    try:
        sdk_kwargs = dict(model=model_name, messages=messages, stream=stream, **kwargs)
        if timeout is not None:
            sdk_kwargs["timeout"] = timeout
        response = await client.chat.completions.create(**sdk_kwargs)
    except _bad_request_errors as e:
        _wrap_context_window_error(e)
    except _openai_errors as e:
        _map_openai_error(e)

    if stream:
        return _wrap_stream_async(response)
    result = ModelResponse(response)
    _fire_completion_success(model, provider, result, start_time, False)
    return result


def mock_completion(model, messages, n=1, stream=False, **kwargs):
    """Create a mock completion response."""
    return completion(model, messages, stream=stream, mock_response=True, n=n, **kwargs)


def _mock_stream_sync(content, model):
    """Yield mock streaming chunks."""
    yield ModelResponseStream(
        ChatCompletionChunk(
            id="mock",
            model=model,
            choices=[ChunkChoice(delta=ChoiceDelta(role="assistant", content=content))],
        )
    )
    yield ModelResponseStream(
        ChatCompletionChunk(
            id="mock",
            model=model,
            choices=[ChunkChoice(finish_reason="stop")],
        )
    )


async def _mock_stream_async(content, model):
    """Yield mock streaming chunks (async)."""
    yield ModelResponseStream(
        ChatCompletionChunk(
            id="mock",
            model=model,
            choices=[ChunkChoice(delta=ChoiceDelta(role="assistant", content=content))],
        )
    )
    yield ModelResponseStream(
        ChatCompletionChunk(
            id="mock",
            model=model,
            choices=[ChunkChoice(finish_reason="stop")],
        )
    )


def _wrap_stream_sync(stream):
    """Wrap sync stream to yield ModelResponseStream objects."""
    for chunk in stream:
        yield ModelResponseStream(chunk)


async def _wrap_stream_async(stream):
    """Wrap async stream to yield ModelResponseStream objects."""
    async for chunk in stream:
        yield ModelResponseStream(chunk)


def stream_chunk_builder(chunks):
    """Build a ModelResponse from a list of ModelResponseStream chunks."""
    if not chunks:
        return ModelResponse(
            ChatCompletion(
                id="empty",
                choices=[],
                created=0,
                model="",
                object="chat.completion",
            )
        )

    # Gather content, tool calls, and usage from chunks
    content_parts = []
    role = "assistant"
    model = ""
    chunk_id = ""
    tool_calls_by_index = {}
    images_by_index = {}
    usage = None
    finish_reason = None
    reasoning_content_parts = []
    thinking_blocks = []
    current_thinking_parts = []
    current_thinking_signature = None
    provider_specific_fields = None

    for c in chunks:
        chunk = c._chunk if hasattr(c, "_chunk") else c
        chunk_id = chunk_id or chunk.id
        model = model or chunk.model

        if hasattr(chunk, "usage") and chunk.usage is not None:
            if usage is None:
                usage = CompletionUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    completion_tokens_details=getattr(chunk.usage, "completion_tokens_details", None),
                    prompt_tokens_details=getattr(chunk.usage, "prompt_tokens_details", None),
                )
            else:
                # Accumulate: take max of each field (Anthropic splits across chunks)
                usage.prompt_tokens = max(usage.prompt_tokens, chunk.usage.prompt_tokens)
                usage.completion_tokens = max(usage.completion_tokens, chunk.usage.completion_tokens)
                # Preserve API's total if it includes hidden categories (e.g. reasoning tokens)
                computed = usage.prompt_tokens + usage.completion_tokens
                usage.total_tokens = max(usage.total_tokens, chunk.usage.total_tokens, computed)
                # Take latest non-None token details
                ctd = getattr(chunk.usage, "completion_tokens_details", None)
                if ctd is not None:
                    usage.completion_tokens_details = ctd
                ptd = getattr(chunk.usage, "prompt_tokens_details", None)
                if ptd is not None:
                    usage.prompt_tokens_details = ptd

        if chunk.choices:
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason or finish_reason
            delta = choice.delta
            if delta:
                if delta.role:
                    role = delta.role
                if delta.content:
                    content_parts.append(delta.content)
                # Collect reasoning_content if present
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    reasoning_content_parts.append(rc)
                # Collect thinking_blocks if present
                tb = getattr(delta, "thinking_blocks", None)
                if tb:
                    for block in tb:
                        if block.get("thinking"):
                            current_thinking_parts.append(block["thinking"])
                        if block.get("signature"):
                            current_thinking_signature = block["signature"]
                            # Flush: signature terminates a thinking block
                            thinking_blocks.append(
                                {
                                    "type": "thinking",
                                    "thinking": "".join(current_thinking_parts),
                                    "signature": current_thinking_signature,
                                }
                            )
                            current_thinking_parts = []
                            current_thinking_signature = None
                if getattr(delta, "images", None):
                    for img in delta.images:
                        images_by_index[img["index"]] = img
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_by_index:
                            tool_calls_by_index[idx] = {
                                "id": tc.id or f"call_{__import__('uuid').uuid4().hex[:24]}",
                                "type": "function",
                                "function": {"name": tc.function.name or "", "arguments": ""},
                            }
                        else:
                            if tc.id:
                                tool_calls_by_index[idx]["id"] = tc.id
                            if tc.function.name:
                                tool_calls_by_index[idx]["function"]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_by_index[idx]["function"]["arguments"] += tc.function.arguments
                psf = getattr(delta, "provider_specific_fields", None)
                if psf and isinstance(psf, dict):
                    if provider_specific_fields is None:
                        provider_specific_fields = {}
                    for k, v in psf.items():
                        if isinstance(v, list):
                            provider_specific_fields.setdefault(k, []).extend(v)
                        else:
                            provider_specific_fields[k] = v

    # Flush any remaining thinking block (no signature received)
    if current_thinking_parts:
        block = {"type": "thinking", "thinking": "".join(current_thinking_parts)}
        if current_thinking_signature:
            block["signature"] = current_thinking_signature
        thinking_blocks.append(block)

    content = "".join(content_parts) or None
    tool_calls = None
    if tool_calls_by_index:
        tool_calls = [
            ChatCompletionMessageToolCall(
                id=tc["id"],
                type=tc["type"],
                function=Function(name=tc["function"]["name"], arguments=tc["function"]["arguments"]),
            )
            for _, tc in sorted(tool_calls_by_index.items())
        ]

    images = [v for _, v in sorted(images_by_index.items())] or None

    message = ChatCompletionMessage(
        role=role,
        content=content,
        tool_calls=tool_calls,
        reasoning_content="".join(reasoning_content_parts) if reasoning_content_parts else None,
        images=images,
        thinking_blocks=thinking_blocks or None,
        provider_specific_fields=provider_specific_fields,
    )

    if usage is None:
        usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    completion_obj = ChatCompletion(
        id=chunk_id or "chatcmpl-stream",
        choices=[Choice(index=0, message=message, finish_reason=finish_reason or "stop")],
        created=0,
        model=model,
        object="chat.completion",
        usage=usage,
    )

    return ModelResponse(completion_obj)
