"""Embedding functions wrapping the OpenAI SDK."""

from litelm._client_cache import get_async_client, get_sync_client
from litelm._completion import _map_openai_error, _openai_errors
from litelm._dispatch import get_handler
from litelm._providers import parse_model


class _EmbeddingItem:
    """Wraps SDK Embedding to support both attribute and dict access (DSPy uses data[i]["embedding"])."""

    __slots__ = ("_item",)

    def __init__(self, item):
        self._item = item

    def __getitem__(self, key):
        return getattr(self._item, key)

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._item, name)


class EmbeddingResponse:
    """Wraps SDK CreateEmbeddingResponse with dict-access on data items."""

    __slots__ = ("_response", "data")

    def __init__(self, response):
        self._response = response
        raw_data = response.data if hasattr(response, "data") else response.get("data", [])
        self.data = [d if isinstance(d, dict) else _EmbeddingItem(d) for d in raw_data]

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._response, name)


def embedding(model, input, *, timeout=None, caching=False, shared_session=None, **kwargs):
    """Synchronous embedding call."""
    kwargs.pop("cache", None)
    kwargs.pop("caching", None)
    kwargs.pop("account_id", None)
    api_key = kwargs.pop("api_key", None)
    api_base = kwargs.pop("api_base", None) or kwargs.pop("base_url", None)
    provider, model_name, base_url, resolved_api_key, api_version = parse_model(
        model, api_key=api_key, api_base=api_base
    )

    handler = get_handler(provider)
    if handler and hasattr(handler, "embedding"):
        return handler.embedding(model_name, input, api_key=resolved_api_key, base_url=base_url, **kwargs)

    client = get_sync_client(provider, base_url, resolved_api_key, api_version=api_version)
    sdk_kwargs = dict(model=model_name, input=input, **kwargs)
    if timeout is not None:
        sdk_kwargs["timeout"] = timeout
    try:
        return EmbeddingResponse(client.embeddings.create(**sdk_kwargs))
    except _openai_errors as e:
        _map_openai_error(e)


async def aembedding(model, input, *, timeout=None, caching=False, shared_session=None, **kwargs):
    """Async embedding call."""
    kwargs.pop("cache", None)
    kwargs.pop("caching", None)
    kwargs.pop("account_id", None)
    api_key = kwargs.pop("api_key", None)
    api_base = kwargs.pop("api_base", None) or kwargs.pop("base_url", None)
    provider, model_name, base_url, resolved_api_key, api_version = parse_model(
        model, api_key=api_key, api_base=api_base
    )

    handler = get_handler(provider)
    if handler and hasattr(handler, "aembedding"):
        return await handler.aembedding(model_name, input, api_key=resolved_api_key, base_url=base_url, **kwargs)

    client = get_async_client(provider, base_url, resolved_api_key, api_version=api_version)
    sdk_kwargs = dict(model=model_name, input=input, **kwargs)
    if timeout is not None:
        sdk_kwargs["timeout"] = timeout
    try:
        return EmbeddingResponse(await client.embeddings.create(**sdk_kwargs))
    except _openai_errors as e:
        _map_openai_error(e)
