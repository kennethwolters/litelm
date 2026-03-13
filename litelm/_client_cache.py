"""Thread-safe cache of OpenAI / AzureOpenAI clients (lazy import)."""

import threading

_lock = threading.Lock()
_sync_clients: dict[tuple, object] = {}
_async_clients: dict[tuple, object] = {}


def _require_openai():
    """Import openai SDK or raise a clear error."""
    try:
        import openai
        return openai
    except ImportError:
        raise ImportError(
            "The openai SDK is required for this provider. "
            "Install it with: pip install litelm[openai]"
        )


def get_sync_client(provider, base_url, api_key, max_retries=0, api_version=None):
    key = ("sync", provider, base_url, api_key, max_retries)
    if key not in _sync_clients:
        with _lock:
            if key not in _sync_clients:
                _sync_clients[key] = _make_client(
                    provider, base_url, api_key, max_retries, api_version, async_client=False
                )
    return _sync_clients[key]


def get_async_client(provider, base_url, api_key, max_retries=0, api_version=None):
    key = ("async", provider, base_url, api_key, max_retries)
    if key not in _async_clients:
        with _lock:
            if key not in _async_clients:
                _async_clients[key] = _make_client(
                    provider, base_url, api_key, max_retries, api_version, async_client=True
                )
    return _async_clients[key]


async def close_async_clients():
    """Close and clear all cached async clients."""
    with _lock:
        for client in _async_clients.values():
            await client.close()
        _async_clients.clear()


def _make_client(provider, base_url, api_key, max_retries, api_version, async_client):
    openai = _require_openai()
    if provider == "azure":
        cls = openai.AsyncAzureOpenAI if async_client else openai.AzureOpenAI
        return cls(
            azure_endpoint=base_url,
            api_key=api_key,
            api_version=api_version or "2024-02-01",
            max_retries=max_retries,
        )
    else:
        cls = openai.AsyncOpenAI if async_client else openai.OpenAI
        kwargs = {"api_key": api_key, "max_retries": max_retries}
        if base_url:
            kwargs["base_url"] = base_url
        return cls(**kwargs)
