"""Tests for the client cache module."""

import json
import socket

import httpx2
import openai
import pytest

import litelm
from litelm._client_cache import (
    _async_clients,
    _sync_clients,
    close_async_clients,
    get_async_client,
    get_sync_client,
)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear client caches before and after each test."""
    _sync_clients.clear()
    _async_clients.clear()
    yield
    _sync_clients.clear()
    _async_clients.clear()


@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.asyncio
async def test_httpx2_client_transport(provider):
    base = "https://example.test" if provider == "azure" else None
    sync_client = get_sync_client(provider, base, "test-key")
    async_client = get_async_client(provider, base, "test-key")
    try:
        assert isinstance(sync_client._client, httpx2.Client)
        assert isinstance(async_client._client, httpx2.AsyncClient)
        assert sync_client.timeout.read == 600
        assert async_client.timeout.read == 600
    finally:
        sync_client.close()
        await async_client.close()


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_httpx2_completion_round_trip(monkeypatch, async_call, stream):
    requests = []

    def handle(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer test-key"
        assert json.loads(request.content)["model"] == "test-model"
        assert request.extensions["timeout"]["read"] == 7
        if stream:
            chunk = {
                "id": "test",
                "model": "test-model",
                "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}],
            }
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            )
        return httpx2.Response(
            200,
            json={
                "id": "test",
                "model": "test-model",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                ],
            },
        )

    factory_name = "DefaultAsyncHttpx2Client" if async_call else "DefaultHttpx2Client"
    factory = getattr(openai, factory_name)
    monkeypatch.setattr(openai, factory_name, lambda **kw: factory(transport=httpx2.MockTransport(handle), **kw))
    kwargs = dict(messages=[{"role": "user", "content": "hi"}], api_key="test-key", stream=stream, timeout=7)
    try:
        # Fail before any network I/O if a regression restores the old transport.
        client = (get_async_client if async_call else get_sync_client)("openai", None, "test-key")
        assert isinstance(client._client, httpx2.AsyncClient if async_call else httpx2.Client)
        result = (
            await litelm.acompletion("openai/test-model", **kwargs)
            if async_call
            else litelm.completion("openai/test-model", **kwargs)
        )
        if stream:
            chunks = [chunk async for chunk in result] if async_call else list(result)
            assert chunks[0].choices[0].delta.content == "hello"
        else:
            assert result.choices[0].message.content == "hello"
        assert len(requests) == 1
        assert str(requests[0].url) == "https://api.openai.com/v1/chat/completions"
    finally:
        for client in _sync_clients.values():
            client.close()
        await close_async_clients()


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("failure", ["rate_limit", "timeout"])
@pytest.mark.asyncio
async def test_httpx2_errors_are_mapped(monkeypatch, async_call, failure):
    def handle(request):
        if failure == "timeout":
            raise httpx2.ReadTimeout("timed out", request=request)
        return httpx2.Response(429, headers={"retry-after": "9"}, json={"error": {"message": "slow down"}})

    factory_name = "DefaultAsyncHttpx2Client" if async_call else "DefaultHttpx2Client"
    factory = getattr(openai, factory_name)
    monkeypatch.setattr(openai, factory_name, lambda **kw: factory(transport=httpx2.MockTransport(handle), **kw))
    try:
        client = (get_async_client if async_call else get_sync_client)("openai", None, "test-key")
        assert isinstance(client._client, httpx2.AsyncClient if async_call else httpx2.Client)
        with pytest.raises(litelm.Timeout if failure == "timeout" else litelm.RateLimitError) as exc:
            if async_call:
                await litelm.acompletion("openai/test", messages=[], api_key="test-key")
            else:
                litelm.completion("openai/test", messages=[], api_key="test-key")
        if failure == "rate_limit":
            assert exc.value.litellm_response_headers["retry-after"] == "9"
    finally:
        for client in _sync_clients.values():
            client.close()
        await close_async_clients()


@pytest.fixture
def owned_wire(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected default transport or network connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(openai, "DefaultHttpx2Client", forbidden)
    monkeypatch.setattr(openai, "DefaultAsyncHttpx2Client", forbidden)


@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_completion_borrows_owned_http_client(owned_wire, provider, async_call, stream):
    requests = []

    def handle(request):
        requests.append(request)
        assert "http_client" not in json.loads(request.content)
        if json.loads(request.content).get("stream"):
            chunk = {
                "id": "test",
                "model": "test",
                "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}],
            }
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            )
        return httpx2.Response(
            200,
            json={
                "id": "test",
                "model": "test",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                ],
            },
        )

    cls = httpx2.AsyncClient if async_call else httpx2.Client
    client = cls(transport=httpx2.MockTransport(handle), trust_env=False)
    kwargs = dict(
        messages=[],
        api_key="test-key",
        api_base="https://provider.test",
        api_version="2024-02-01",
        http_client=client,
        num_retries=0,
        stream=stream,
    )
    try:
        result = (
            await litelm.acompletion(f"{provider}/test", **kwargs)
            if async_call
            else litelm.completion(f"{provider}/test", **kwargs)
        )
        if stream:
            chunks = [chunk async for chunk in result] if async_call else list(result)
            assert chunks[0].choices[0].delta.content == "hello"
        else:
            assert result.choices[0].message.content == "hello"
        assert len(requests) == 1
        assert requests[0].url.host == "provider.test"
        assert not _sync_clients and not _async_clients
        await close_async_clients()
        assert not client.is_closed
        # A second call reuses the caller's pool, not a cached SDK wrapper.
        result = (
            await litelm.acompletion(f"{provider}/test", **{**kwargs, "stream": False})
            if async_call
            else litelm.completion(f"{provider}/test", **{**kwargs, "stream": False})
        )
        assert result.choices[0].message.content == "hello"
        assert len(requests) == 2 and not client.is_closed
    finally:
        await client.aclose() if async_call else client.close()


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("status", [400, 429, 503, "timeout"])
@pytest.mark.asyncio
async def test_owned_completion_transport_preserves_errors_without_retry(owned_wire, async_call, status):
    attempts = []

    def handle(request):
        attempts.append(request)
        if status == "timeout":
            raise httpx2.ReadTimeout("timed out", request=request)
        return httpx2.Response(status, json={"error": {"message": "test failure"}})

    client = (httpx2.AsyncClient if async_call else httpx2.Client)(transport=httpx2.MockTransport(handle))
    expected = {
        400: litelm.BadRequestError,
        429: litelm.RateLimitError,
        503: litelm.InternalServerError,
        "timeout": litelm.Timeout,
    }[status]
    try:
        with pytest.raises(expected):
            kwargs = dict(messages=[], api_key="test-key", http_client=client, num_retries=0)
            if async_call:
                await litelm.acompletion("openai/test", **kwargs)
            else:
                litelm.completion("openai/test", **kwargs)
        assert len(attempts) == 1
        assert not client.is_closed
        assert not _sync_clients and not _async_clients
    finally:
        await client.aclose() if async_call else client.close()


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("provider", ["anthropic", "bedrock", "cloudflare", "mistral"])
@pytest.mark.asyncio
async def test_custom_completion_handler_rejects_owned_transport(owned_wire, async_call, provider):
    with pytest.raises(ValueError, match="http_client.*not supported"):
        kwargs = dict(messages=[], api_key="test-key", http_client=object())
        if async_call:
            await litelm.acompletion(f"{provider}/test", **kwargs)
        else:
            litelm.completion(f"{provider}/test", **kwargs)


def test_sync_client_creation():
    """get_sync_client returns an OpenAI client."""
    client = get_sync_client("openai", None, "sk-test")
    assert client is not None


def test_sync_client_cache_hit():
    """Same params return the same cached client."""
    c1 = get_sync_client("openai", None, "sk-test")
    c2 = get_sync_client("openai", None, "sk-test")
    assert c1 is c2


def test_sync_client_cache_miss_different_key():
    """Different api_key returns a different client."""
    c1 = get_sync_client("openai", None, "sk-a")
    c2 = get_sync_client("openai", None, "sk-b")
    assert c1 is not c2


def test_sync_client_cache_miss_different_base_url():
    """Different base_url returns a different client."""
    c1 = get_sync_client("groq", "https://api.groq.com/openai/v1", "sk-test")
    c2 = get_sync_client("groq", "https://other.groq.com/v1", "sk-test")
    assert c1 is not c2


def test_sync_client_cache_miss_different_retries():
    """Different max_retries returns a different client."""
    c1 = get_sync_client("openai", None, "sk-test", max_retries=0)
    c2 = get_sync_client("openai", None, "sk-test", max_retries=3)
    assert c1 is not c2


def test_async_client_creation():
    """get_async_client returns an AsyncOpenAI client."""
    client = get_async_client("openai", None, "sk-test")
    assert client is not None


def test_async_client_cache_hit():
    """Same params return the same cached async client."""
    c1 = get_async_client("openai", None, "sk-test")
    c2 = get_async_client("openai", None, "sk-test")
    assert c1 is c2


def test_azure_sync_client():
    """Azure provider creates AzureOpenAI client."""
    import openai

    client = get_sync_client("azure", "https://my.azure.com", "sk-az", api_version="2024-02-01")
    assert isinstance(client, openai.AzureOpenAI)


def test_azure_async_client():
    """Azure provider creates AsyncAzureOpenAI client."""
    import openai

    client = get_async_client("azure", "https://my.azure.com", "sk-az", api_version="2024-02-01")
    assert isinstance(client, openai.AsyncAzureOpenAI)


def test_azure_sync_different_api_version():
    """Different api_version returns different client (Bug 1 regression)."""
    c1 = get_sync_client("azure", "https://test.openai.azure.com", "sk-az", api_version="2024-02-01")
    c2 = get_sync_client("azure", "https://test.openai.azure.com", "sk-az", api_version="2025-01-01")
    assert c1 is not c2


def test_azure_async_different_api_version():
    """Different api_version returns different async client (Bug 1 regression)."""
    c1 = get_async_client("azure", "https://test.openai.azure.com", "sk-az", api_version="2024-02-01")
    c2 = get_async_client("azure", "https://test.openai.azure.com", "sk-az", api_version="2025-01-01")
    assert c1 is not c2


def test_azure_same_api_version_cached():
    """Same api_version returns same cached client."""
    c1 = get_sync_client("azure", "https://test.openai.azure.com", "sk-az", api_version="2024-02-01")
    c2 = get_sync_client("azure", "https://test.openai.azure.com", "sk-az", api_version="2024-02-01")
    assert c1 is c2


@pytest.mark.asyncio
async def test_close_async_clients():
    """close_async_clients clears the async cache."""
    get_async_client("openai", None, "sk-test")
    assert len(_async_clients) == 1
    await close_async_clients()
    assert len(_async_clients) == 0


def _fake_token():
    return "fake-token"


def test_azure_ad_token_provider_forwarded():
    """azure_ad_token_provider is passed to AzureOpenAI constructor."""
    client = get_sync_client("azure", "https://my.azure.com", None, azure_ad_token_provider=_fake_token)
    assert client._azure_ad_token_provider is _fake_token


def test_azure_ad_token_provider_forwarded_async():
    """azure_ad_token_provider is passed to AsyncAzureOpenAI constructor."""
    client = get_async_client("azure", "https://my.azure.com", None, azure_ad_token_provider=_fake_token)
    assert client._azure_ad_token_provider is _fake_token


def _token_a():
    return "token-a"


def _token_b():
    return "token-b"


def test_azure_different_token_providers_not_cached():
    """Different azure_ad_token_provider functions produce different clients."""
    c1 = get_sync_client("azure", "https://x.azure.com", None, azure_ad_token_provider=_token_a)
    c2 = get_sync_client("azure", "https://x.azure.com", None, azure_ad_token_provider=_token_b)
    assert c1 is not c2


def test_azure_same_token_provider_cached():
    """Same azure_ad_token_provider function reuses cached client."""
    c1 = get_sync_client("azure", "https://x.azure.com", None, azure_ad_token_provider=_fake_token)
    c2 = get_sync_client("azure", "https://x.azure.com", None, azure_ad_token_provider=_fake_token)
    assert c1 is c2


def test_azure_none_token_provider_backwards_compat():
    """Omitting azure_ad_token_provider works as before."""
    import openai

    client = get_sync_client("azure", "https://my.azure.com", "sk-az", api_version="2024-02-01")
    assert isinstance(client, openai.AzureOpenAI)
