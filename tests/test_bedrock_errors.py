"""Tests for Bedrock error mapping — openai exceptions → litelm exceptions."""

import datetime
import json
from unittest import mock

import httpx2
import openai
import pytest

from litelm._exceptions import (
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    RateLimitError,
    Timeout,
)


def _make_openai_error(cls_name, message="test error"):
    """Create a mock openai exception with the given class name."""
    openai = pytest.importorskip("openai")
    cls = getattr(openai, cls_name)
    # openai errors need specific constructor args
    if cls_name == "APITimeoutError":
        return cls(request=mock.MagicMock())
    if cls_name == "APIConnectionError":
        return cls(request=mock.MagicMock())
    return cls(
        message=message,
        response=mock.MagicMock(status_code=400),
        body={"error": {"message": message}},
    )


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_sigv4_httpx2_round_trip_and_credential_refresh(monkeypatch, async_call, stream):
    import botocore.session
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import ReadOnlyCredentials

    from litelm.providers import _bedrock

    credentials = [ReadOnlyCredentials(f"test-key-{i}", f"test-secret-{i}", f"test-token-{i}") for i in range(2)]
    provider = mock.Mock()
    provider.get_frozen_credentials.side_effect = credentials
    session = mock.Mock()
    session.get_credentials.return_value = provider
    monkeypatch.setattr(botocore.session, "get_session", lambda: session)
    monkeypatch.setattr("botocore.auth.get_current_datetime", lambda: datetime.datetime(2026, 9, 23, 12, 0, 0))
    requests = []

    def handle(request):
        current = credentials[len(requests)]
        requests.append(request)
        assert request.headers["x-amz-security-token"] == current.token
        assert "us-west-2/bedrock/aws4_request" in request.headers["authorization"]
        assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256 ")
        assert json.loads(request.content)["messages"][0]["content"] == "hello café"
        unsigned = AWSRequest(
            method=request.method, url=str(request.url), headers=dict(request.headers), data=request.content
        )
        SigV4Auth(current, "bedrock", "us-west-2").add_auth(unsigned)
        assert request.headers["authorization"] == unsigned.headers["Authorization"]
        if stream:
            chunk = {"id": "test", "choices": [{"index": 0, "delta": {"content": "hello"}}]}
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            )
        return httpx2.Response(
            200,
            json={
                "id": "test",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                ],
            },
        )

    factory_name = "DefaultAsyncHttpx2Client" if async_call else "DefaultHttpx2Client"
    factory = getattr(openai, factory_name)
    monkeypatch.setattr(openai, factory_name, lambda **kw: factory(transport=httpx2.MockTransport(handle), **kw))
    # Build the real SDK + transport, but never contact AWS or load actual credentials.
    client = _bedrock._create_openai_client("https://bedrock.test/openai/v1", "us-west-2", async_call)
    monkeypatch.setattr(_bedrock, "_get_openai_client", lambda *args, **kwargs: client)
    try:
        assert isinstance(client._client, httpx2.AsyncClient if async_call else httpx2.Client)
        for _ in range(2):
            kwargs = dict(messages=[{"role": "user", "content": "hello café"}], stream=stream, max_tokens=8)
            result = (
                await _bedrock.acompletion("test-model", **kwargs)
                if async_call
                else _bedrock.completion("test-model", **kwargs)
            )
            if stream:
                chunks = [chunk async for chunk in result] if async_call else list(result)
                assert chunks[0].choices[0].delta.content == "hello"
            else:
                assert result.choices[0].message.content == "hello"
        assert len(requests) == 2
        assert provider.get_frozen_credentials.call_count == 2
    finally:
        if async_call:
            await client.close()
        else:
            client.close()


class TestBedrockErrorMapping:
    def setup_method(self):
        from litelm.providers import _bedrock

        self.mod = _bedrock

    def test_bad_request_maps(self):
        err = _make_openai_error("BadRequestError", "invalid request")
        with pytest.raises(BadRequestError, match="invalid request"):
            self.mod._map_error(err)

    def test_rate_limit_maps(self):
        err = _make_openai_error("RateLimitError", "rate limited")
        with pytest.raises(RateLimitError):
            self.mod._map_error(err)

    def test_auth_error_maps(self):
        err = _make_openai_error("AuthenticationError", "bad key")
        with pytest.raises(AuthenticationError):
            self.mod._map_error(err)

    def test_timeout_maps(self):
        err = _make_openai_error("APITimeoutError")
        with pytest.raises(Timeout):
            self.mod._map_error(err)

    def test_context_window_from_bad_request(self):
        err = _make_openai_error("BadRequestError", "maximum context length exceeded")
        with pytest.raises(ContextWindowExceededError):
            self.mod._map_error(err)

    def test_context_window_token_keyword(self):
        err = _make_openai_error("BadRequestError", "too many tokens in request")
        with pytest.raises(ContextWindowExceededError):
            self.mod._map_error(err)
