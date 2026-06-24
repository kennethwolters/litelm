"""Cloudflare Workers AI handler via Cloudflare's OpenAI-compatible endpoint.

Model strings: cloudflare/@cf/meta/llama-2-7b-chat-int8
Env vars: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN
"""

import os

from litelm._client_cache import get_async_client, get_sync_client
from litelm._completion import _map_openai_error, _openai_errors
from litelm._types import ModelResponse, ModelResponseStream

_BASE_URL = "https://api.cloudflare.com/client/v4/accounts"
_LEGACY_AI_RUN_SUFFIX = "/ai/run"
_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"


def _normalize_nonempty(value):
    if value is None:
        return None
    value = value.strip()
    return value or None


def _resolve_api_key(api_key=None):
    token = _normalize_nonempty(api_key) or _normalize_nonempty(os.environ.get("CLOUDFLARE_API_TOKEN"))
    if token is None:
        raise ValueError("CLOUDFLARE_API_TOKEN environment variable is required for cloudflare/ models.")
    return token


def _build_base_url(account_id):
    return f"{_BASE_URL}/{account_id}/ai/v1"


def _resolve_base_url(base_url=None):
    """Resolve Cloudflare's OpenAI-compatible base URL.

    Cloudflare's legacy Workers AI path was /ai/run/{model}. The OpenAI-compatible
    chat API lives under /ai/v1, so rewrite a configured /ai/run base.
    """
    base_url = _normalize_nonempty(base_url)
    if base_url:
        trimmed = base_url.rstrip("/")
        if trimmed.endswith(_CHAT_COMPLETIONS_SUFFIX):
            trimmed = trimmed[: -len(_CHAT_COMPLETIONS_SUFFIX)]
        if trimmed.endswith(_LEGACY_AI_RUN_SUFFIX):
            return f"{trimmed[: -len(_LEGACY_AI_RUN_SUFFIX)]}/ai/v1"
        return trimmed

    account_id = _normalize_nonempty(os.environ.get("CLOUDFLARE_ACCOUNT_ID"))
    if account_id is None:
        raise ValueError("CLOUDFLARE_ACCOUNT_ID environment variable is required for cloudflare/ models.")
    return _build_base_url(account_id)


def _prepare_sdk_kwargs(model_name, messages, stream, kwargs):
    sdk_kwargs = dict(model=model_name, messages=messages, stream=stream, **kwargs)
    if "max_completion_tokens" in sdk_kwargs and "max_tokens" not in sdk_kwargs:
        sdk_kwargs["max_tokens"] = sdk_kwargs.pop("max_completion_tokens")
    return sdk_kwargs


def completion(model_name, messages, *, stream=False, api_key=None, base_url=None, timeout=None, **kwargs):
    """Synchronous Cloudflare completion through the OpenAI-compatible API."""
    token = _resolve_api_key(api_key)
    resolved_base_url = _resolve_base_url(base_url)
    client = get_sync_client("cloudflare", resolved_base_url, token)

    sdk_kwargs = _prepare_sdk_kwargs(model_name, messages, stream, kwargs)
    if timeout is not None:
        sdk_kwargs["timeout"] = timeout

    try:
        response = client.chat.completions.create(**sdk_kwargs)
    except _openai_errors as e:
        _map_openai_error(e)

    if stream:
        return _wrap_stream_sync(response)
    return ModelResponse(response)


async def acompletion(model_name, messages, *, stream=False, api_key=None, base_url=None, timeout=None, **kwargs):
    """Async Cloudflare completion through the OpenAI-compatible API."""
    token = _resolve_api_key(api_key)
    resolved_base_url = _resolve_base_url(base_url)
    client = get_async_client("cloudflare", resolved_base_url, token)

    sdk_kwargs = _prepare_sdk_kwargs(model_name, messages, stream, kwargs)
    if timeout is not None:
        sdk_kwargs["timeout"] = timeout

    try:
        response = await client.chat.completions.create(**sdk_kwargs)
    except _openai_errors as e:
        _map_openai_error(e)

    if stream:
        return _wrap_stream_async(response)
    return ModelResponse(response)


def _wrap_stream_sync(stream):
    for chunk in stream:
        yield ModelResponseStream(chunk)


async def _wrap_stream_async(stream):
    async for chunk in stream:
        yield ModelResponseStream(chunk)
