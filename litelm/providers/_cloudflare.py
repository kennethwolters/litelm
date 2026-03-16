"""Cloudflare Workers AI handler using httpx.

Model strings: cloudflare/@cf/meta/llama-2-7b-chat-int8
Env vars: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN
"""

import json
import os
import time

from litelm._exceptions import (
    AuthenticationError,
    APIStatusError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from litelm._types import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionChunk,
    Choice,
    ChunkChoice,
    ChoiceDelta,
    CompletionUsage,
    ModelResponse,
    ModelResponseStream,
)

_BASE_URL = "https://api.cloudflare.com/client/v4/accounts"


def _get_config(api_key=None, base_url=None):
    """Resolve account_id and api_token from args or env."""
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not account_id:
        raise ValueError("CLOUDFLARE_ACCOUNT_ID environment variable is required for cloudflare/ models.")
    token = api_key or os.environ.get("CLOUDFLARE_API_TOKEN")
    if not token:
        raise ValueError("CLOUDFLARE_API_TOKEN environment variable is required for cloudflare/ models.")
    return account_id, token


def _build_url(account_id, model_name):
    """Build the Cloudflare AI run URL."""
    return f"{_BASE_URL}/{account_id}/ai/run/{model_name}"


def _build_request_body(messages, **kwargs):
    """Build Cloudflare-format request body from OpenAI messages."""
    body = {"messages": messages}

    for key in ("temperature", "max_tokens", "top_p"):
        if key in kwargs:
            val = kwargs.pop(key)
            if val is not None:
                body[key] = val

    max_completion_tokens = kwargs.pop("max_completion_tokens", None)
    if max_completion_tokens and "max_tokens" not in body:
        body["max_tokens"] = max_completion_tokens

    stream = kwargs.pop("stream", False)
    if stream:
        body["stream"] = True

    # Drop unsupported params
    for drop in ("tools", "tool_choice", "response_format", "frequency_penalty",
                 "presence_penalty", "seed", "logprobs", "top_logprobs", "n",
                 "stop", "extra_headers"):
        kwargs.pop(drop, None)

    return body, stream


def _parse_response(data, model_name):
    """Parse Cloudflare response JSON into ModelResponse."""
    result = data.get("result", {})
    response_text = result.get("response", "")

    message = ChatCompletionMessage(role="assistant", content=response_text)
    usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    completion = ChatCompletion(
        id=f"chatcmpl-cf-{int(time.time())}",
        choices=[Choice(index=0, message=message, finish_reason="stop")],
        created=int(time.time()),
        model=model_name,
        object="chat.completion",
        usage=usage,
    )
    return ModelResponse(completion)


def _parse_stream_line(line, model_name, chunk_id):
    """Parse a single SSE line from Cloudflare streaming response."""
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None

    data_str = line[5:].strip()
    if data_str == "[DONE]":
        delta = ChoiceDelta()
        choice = ChunkChoice(index=0, delta=delta, finish_reason="stop")
        chunk = ChatCompletionChunk(
            id=chunk_id,
            choices=[choice],
            created=int(time.time()),
            model=model_name,
            object="chat.completion.chunk",
        )
        return ModelResponseStream(chunk)

    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return None

    text = data.get("response", "")
    delta = ChoiceDelta(content=text, role="assistant")
    choice = ChunkChoice(index=0, delta=delta, finish_reason=None)
    chunk = ChatCompletionChunk(
        id=chunk_id,
        choices=[choice],
        created=int(time.time()),
        model=model_name,
        object="chat.completion.chunk",
    )
    return ModelResponseStream(chunk)


def _handle_error_response(response):
    """Raise appropriate litelm exception for error HTTP status."""
    status = response.status_code
    try:
        body = response.json()
        msg = json.dumps(body.get("errors", body))
    except Exception:
        msg = response.text

    if status == 401:
        raise AuthenticationError(message=msg, response=response, body=msg)
    elif status == 429:
        raise RateLimitError(message=msg, response=response, body=msg)
    elif status == 400:
        raise BadRequestError(message=msg, response=response, body=msg)
    elif status == 403:
        raise PermissionDeniedError(message=msg, response=response, body=msg)
    elif status == 404:
        raise NotFoundError(message=msg, response=response, body=msg)
    elif status == 422:
        raise UnprocessableEntityError(message=msg, response=response, body=msg)
    elif status >= 500:
        raise InternalServerError(message=msg, response=response, body=msg)
    else:
        raise APIStatusError(message=msg, response=response, body=msg)


def completion(model_name, messages, *, stream=False, api_key=None, base_url=None, timeout=None, **kwargs):
    """Synchronous Cloudflare Workers AI completion."""
    import httpx

    account_id, token = _get_config(api_key)
    url = _build_url(account_id, model_name)
    body, do_stream = _build_request_body(messages, stream=stream, **kwargs)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    http_timeout = timeout if timeout is not None else 120

    if do_stream:
        return _stream_sync(url, headers, body, model_name, http_timeout)

    with httpx.Client(timeout=http_timeout) as client:
        resp = client.post(url, json=body, headers=headers)

    if resp.status_code != 200:
        _handle_error_response(resp)

    return _parse_response(resp.json(), model_name)


def _stream_sync(url, headers, body, model_name, http_timeout=120):
    """Synchronous streaming generator for Cloudflare."""
    import httpx

    chunk_id = f"chatcmpl-cf-{int(time.time())}"
    with httpx.Client(timeout=http_timeout) as client:
        with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                resp.read()
                _handle_error_response(resp)
            for line in resp.iter_lines():
                chunk = _parse_stream_line(line, model_name, chunk_id)
                if chunk is not None:
                    yield chunk


async def acompletion(model_name, messages, *, stream=False, api_key=None, base_url=None, timeout=None, **kwargs):
    """Async Cloudflare Workers AI completion."""
    import httpx

    account_id, token = _get_config(api_key)
    url = _build_url(account_id, model_name)
    body, do_stream = _build_request_body(messages, stream=stream, **kwargs)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    http_timeout = timeout if timeout is not None else 120

    if do_stream:
        return _stream_async(url, headers, body, model_name, http_timeout)

    async with httpx.AsyncClient(timeout=http_timeout) as client:
        resp = await client.post(url, json=body, headers=headers)

    if resp.status_code != 200:
        _handle_error_response(resp)

    return _parse_response(resp.json(), model_name)


async def _stream_async(url, headers, body, model_name, http_timeout=120):
    """Async streaming generator for Cloudflare."""
    import httpx

    chunk_id = f"chatcmpl-cf-{int(time.time())}"
    async with httpx.AsyncClient(timeout=http_timeout) as client:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                await resp.aread()
                _handle_error_response(resp)
            async for line in resp.aiter_lines():
                chunk = _parse_stream_line(line, model_name, chunk_id)
                if chunk is not None:
                    yield chunk
