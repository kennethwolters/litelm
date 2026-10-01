"""Public completion contracts through real SDKs and offline HTTPX2 transports."""

import asyncio
import json
import socket
from types import SimpleNamespace
from unittest import mock

import anthropic
import botocore.session
import httpx2
import openai
import pytest
from botocore.credentials import Credentials

import litelm
from litelm import _client_cache
from litelm._exceptions import LitelmError
from litelm.providers import _bedrock

PROVIDERS = ["openai", "mistral", "cloudflare", "anthropic", "bedrock"]


@pytest.fixture
async def wire(monkeypatch):
    """Replace only transports/credentials; retain actual SDK retries and decoding."""
    state = SimpleNamespace(requests=[], respond=None)
    transports = []
    _client_cache._sync_clients.clear()
    _client_cache._async_clients.clear()
    _bedrock._client_cache.clear()

    def no_network(*args, **kwargs):
        raise AssertionError("Unexpected real network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)

    def handle(request):
        assert request.url.host == "provider.test"
        body = json.loads(request.content)
        assert not {"num_retries", "max_retries"} & body.keys()
        state.requests.append(request)
        return state.respond(request)

    for name in ("DefaultHttpx2Client", "DefaultAsyncHttpx2Client"):
        factory = getattr(openai, name)

        def create(factory=factory, **kwargs):
            client = factory(transport=httpx2.MockTransport(handle), **kwargs)
            transports.append(client)
            return client

        monkeypatch.setattr(openai, name, create)

    for name, transport_cls in (("Anthropic", httpx2.Client), ("AsyncAnthropic", httpx2.AsyncClient)):
        factory = getattr(anthropic, name)

        def create(factory=factory, transport_cls=transport_cls, **kwargs):
            client = transport_cls(transport=httpx2.MockTransport(handle))
            transports.append(client)
            return factory(http_client=client, **kwargs)

        monkeypatch.setattr(anthropic, name, create)

    session = mock.Mock()
    session.get_credentials.return_value = Credentials("test-access", "test-secret", "test-session")
    monkeypatch.setattr(botocore.session, "get_session", lambda: session)
    # Count retry attempts without sleeping; backoff timing is the SDK's responsibility.
    for sdk in (openai, anthropic):
        monkeypatch.setattr(sdk._base_client.BaseClient, "_calculate_retry_timeout", lambda *a, **kw: 0)
    yield state
    for client in transports:
        if isinstance(client, httpx2.AsyncClient):
            await client.aclose()
        else:
            client.close()
    _client_cache._sync_clients.clear()
    _client_cache._async_clients.clear()
    _bedrock._client_cache.clear()


async def call(provider, async_call, **kwargs):
    params = dict(
        model=f"{provider}/test-model",
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
        api_base="https://provider.test/v1",
        max_tokens=8,
        **kwargs,
    )
    return await litelm.acompletion(**params) if async_call else litelm.completion(**params)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    "message,code,expected",
    [
        ("invalid image payload", None, litelm.BadRequestError),
        ("invalid token", None, litelm.BadRequestError),
        ("tool name length must be less than 64", None, litelm.BadRequestError),
        ("context parameter must be a string", None, litelm.BadRequestError),
        ("max_tokens must be less than 64000", None, litelm.BadRequestError),
        ("maximum context length exceeded", None, litelm.ContextWindowExceededError),
        ("prompt is too long: 250000 tokens > 200000 maximum", None, litelm.ContextWindowExceededError),
        ("input length and max_tokens exceed context limit", None, litelm.ContextWindowExceededError),
        ("Request rejected", "context_length_exceeded", litelm.ContextWindowExceededError),
    ],
)
async def test_bad_request_mapping(wire, provider, async_call, message, code, expected):
    body = {"error": {"message": message, "type": "invalid_request_error", "code": code}}
    wire.respond = lambda request: httpx2.Response(400, headers={"x-request-id": "test-id"}, json=body)
    with pytest.raises(expected) as caught:
        await call(provider, async_call)
    assert type(caught.value) is expected
    assert caught.value.body == body or caught.value.body == body["error"]
    assert caught.value.litellm_response_headers["x-request-id"] == "test-id"
    assert caught.value.__cause__ is not None
    assert len(wire.requests) == 1


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure", [400, 429, 503, "timeout"])
async def test_retry_controls_reach_provider_clients(wire, provider, async_call, stream, failure):
    def respond(request):
        if failure == "timeout":
            raise httpx2.ReadTimeout("synthetic timeout", request=request)
        return httpx2.Response(failure, json={"error": {"type": "api_error", "message": "try again"}})

    wire.respond = respond
    expected = {
        400: litelm.BadRequestError,
        429: litelm.RateLimitError,
        503: litelm.InternalServerError,
        "timeout": litelm.Timeout,
    }[failure]
    # Same provider/key/endpoint, changing retry count: cached clients must not retain the old policy.
    for options, attempts in [
        ({}, 1),
        ({"num_retries": 0}, 1),
        ({"num_retries": 2}, 3),
        ({"max_retries": 2}, 3),
        ({"num_retries": 0, "max_retries": 2}, 1),
    ]:
        wire.requests.clear()
        with pytest.raises(expected):
            result = await call(provider, async_call, stream=stream, **options)
            if stream:
                if async_call:
                    async for _ in result:
                        pass
                else:
                    list(result)
        assert len(wire.requests) == (1 if failure == 400 else attempts)


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("primary", [httpx2.ReadError, asyncio.CancelledError, None])
async def test_wrapper_cleanup_does_not_replace_primary_failure(async_call, primary):
    from litelm._completion import _wrap_stream_async, _wrap_stream_sync

    class Source:
        def __iter__(self):
            if primary:
                raise primary("original failure")
            return iter(())

        async def __aiter__(self):
            for part in self:
                yield part

    source = Source()
    close = mock.AsyncMock if async_call else mock.Mock
    source.close = close(side_effect=RuntimeError("cleanup failed"))
    expected = litelm.APIConnectionError if primary is httpx2.ReadError else primary or RuntimeError
    with pytest.raises(expected):
        if async_call:
            async for _ in _wrap_stream_async(source):
                pass
        else:
            list(_wrap_stream_sync(source))
    source.close.assert_called_once()


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_cancelling_active_stream_closes_without_retry(wire, provider):
    entered = asyncio.Event()

    class WaitingStream(httpx2.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield stream_prefix(provider)
            entered.set()
            await asyncio.Future()

        async def aclose(self):
            self.closed = True

    source = WaitingStream()
    wire.respond = lambda request: httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=source)

    async def consume():
        stream = await call(provider, True, stream=True, num_retries=2)
        async for _ in stream:
            pass

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert source.closed
    assert len(wire.requests) == 1


def stream_prefix(provider):
    if provider == "anthropic":
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "test",
                    "type": "message",
                    "role": "assistant",
                    "model": "test-model",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}},
        ]
        return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
    return b'data: {"id":"test","choices":[{"index":0,"delta":{"content":"hello"}}]}\n\n'


class FaultStream(httpx2.SyncByteStream, httpx2.AsyncByteStream):
    def __init__(self, prefix, failure):
        self.prefix = prefix
        self.failure = failure
        self.closed = False

    def __iter__(self):
        if self.prefix:
            yield self.prefix
        if self.failure:
            raise self.failure("synthetic failure")

    async def __aiter__(self):
        for part in self:
            yield part

    def close(self):
        self.closed = True

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("with_prefix", [False, True])
@pytest.mark.parametrize(
    "failure,expected",
    [
        (httpx2.ReadError, litelm.APIConnectionError),
        (httpx2.ReadTimeout, litelm.Timeout),
        (ValueError, ValueError),  # Programming/decoding errors must not become retryable connection errors.
        (asyncio.CancelledError, asyncio.CancelledError),
    ],
)
async def test_stream_errors_preserve_prefix_map_transport_and_close(
    wire, provider, async_call, with_prefix, failure, expected
):
    source = FaultStream(stream_prefix(provider) if with_prefix else b"", failure)
    wire.respond = lambda request: httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=source)
    result = await call(provider, async_call, stream=True, num_retries=2)
    text = []
    with pytest.raises(expected) as caught:
        if async_call:
            async for chunk in result:
                text.append(chunk.choices[0].delta.content or "")
        else:
            for chunk in result:
                text.append(chunk.choices[0].delta.content or "")
    assert "".join(text) == ("hello" if with_prefix else "")
    if issubclass(expected, LitelmError):
        assert isinstance(caught.value.__cause__, failure)
        assert "synthetic failure" in str(caught.value)
    assert len(wire.requests) == 1  # Never retry/replay an already-created stream.
    assert source.closed


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("early_close", [False, True])
async def test_stream_closes_on_exhaustion_or_explicit_close(wire, provider, async_call, early_close):
    source = FaultStream(stream_prefix(provider), None)
    wire.respond = lambda request: httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=source)
    result = await call(provider, async_call, stream=True)
    if async_call:
        if early_close:
            await result.__anext__()
            await result.aclose()
        else:
            async for _ in result:
                pass
    else:
        if early_close:
            next(result)
            result.close()
        else:
            list(result)
    assert source.closed
    assert len(wire.requests) == 1
