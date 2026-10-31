"""Resource, privacy and early-validation boundaries for native decision calls."""

import json

import httpx2
import pytest

import litelm
from tests.test_systemone import ANSWERS, QUESTIONS, invoke


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    "params",
    [
        {"state": float("inf")},
        {"state": object()},
        {"model": "openai/model"},
        {"model": "typesafe/"},
        {"num_retries": -1},
        {"num_retries": True},
        {"num_retries": 11},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"api_base": "http://remote.test"},
        {"api_base": "https://user:password@provider.test"},
        {"api_base": "https://provider.test?api-key=secret"},
        {"questions": {}},
        {"questions": {"x": {"type": "noul"}}},
        {"questions": {"x": {"type": "score", "criteria": ["Only one"]}}},
        {"questions": {"x": {"type": "choice", "criteria": {"a": "A", "b": "B"}, "tools": []}}},
        {"state": "x" * 2_000_000},
    ],
)
@pytest.mark.asyncio
async def test_request_validation_precedes_network(async_call, params):
    with pytest.raises(litelm.BadRequestError) as caught:
        await invoke(async_call, lambda request: pytest.fail("Invalid request reached transport"), **params)
    assert caught.value.__context__ is None and caught.value.__cause__ is None


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    "content", [b"not json", b"[]", b"null", b"{}", b"x" * 1_000_001, b'{"answers": {}, "answers": {}}']
)
@pytest.mark.asyncio
async def test_invalid_success_and_size_limit(async_call, content):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx2.Response(200, content=content)

    with pytest.raises(litelm.APIResponseValidationError) as caught:
        await invoke(async_call, handle, num_retries=1)
    assert len(calls) == 1
    assert caught.value.__context__ is None and caught.value.__cause__ is None


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    "error,expected", [(httpx2.ConnectError, litelm.APIConnectionError), (httpx2.ReadTimeout, litelm.Timeout)]
)
@pytest.mark.asyncio
async def test_transport_errors_are_sanitized(async_call, error, expected):
    def handle(request):
        raise error("private-input secret-key", request=request)

    with pytest.raises(expected) as caught:
        await invoke(async_call, handle)
    assert "secret-key" not in str(caught.value)
    assert caught.value.request is None


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.asyncio
async def test_no_redirects_no_callbacks_and_client_stays_owned(async_call, monkeypatch):
    calls, callbacks = [], []
    monkeypatch.setattr(litelm, "success_callback", [lambda *args: callbacks.append(args)])
    monkeypatch.setattr(litelm, "failure_callback", [lambda *args: callbacks.append(args)])

    def handle(request):
        calls.append(request)
        return httpx2.Response(307, headers={"location": "https://elsewhere.test"})

    params = dict(model="typesafe/test", state="Beispiel", questions=QUESTIONS, api_key="test")
    if async_call:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle), follow_redirects=True) as client:
            with pytest.raises(litelm.APIStatusError):
                await litelm.asystemone(**params, http_client=client)
            assert not client.is_closed
    else:
        with httpx2.Client(transport=httpx2.MockTransport(handle), follow_redirects=True) as client:
            with pytest.raises(litelm.APIStatusError):
                litelm.systemone(**params, http_client=client)
            assert not client.is_closed
    assert len(calls) == 1 and not callbacks


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.asyncio
async def test_typed_questions_emit_defaults_and_preserve_nulls(async_call):
    def handle(request):
        payload = json.loads(request.content)
        assert payload["state"] is None
        assert payload["questions"] == {"flag": {"type": "noul", "criteria": {"true": None, "false": "Nein"}}}
        return httpx2.Response(200, json={"answers": {"flag": ANSWERS["flag"]}})

    response = await invoke(
        async_call,
        handle,
        state=None,
        questions={"flag": litelm.NoulQuestion(criteria={"true": None, "false": "Nein"})},
    )
    assert response.answers["flag"].noul == 0.7


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.asyncio
async def test_usage_is_real_and_serializable(async_call):
    response = await invoke(
        async_call,
        lambda request: httpx2.Response(
            200,
            json={
                "answers": ANSWERS,
                "usage": {"input_tokens": 12, "output_tokens": 3},
            },
        ),
    )
    assert response.usage.input_tokens == 12 and response.usage.total_tokens is None
    assert litelm.SystemOneResponse.model_validate_json(response.model_dump_json()) == response


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.asyncio
async def test_compressed_response_is_rejected_without_reading_body(async_call):
    class UnreadBody(httpx2.SyncByteStream, httpx2.AsyncByteStream):
        def __iter__(self):
            pytest.fail("Compressed body was read before validation")
            yield b""

        async def __aiter__(self):
            pytest.fail("Compressed body was read before validation")
            yield b""

    def handle(request):
        assert request.headers["accept-encoding"] == "identity"
        return httpx2.Response(200, headers={"content-encoding": "gzip"}, stream=UnreadBody())

    with pytest.raises(litelm.APIResponseValidationError, match="encoding"):
        await invoke(async_call, handle)


@pytest.mark.asyncio
async def test_async_token_callback_and_failure():
    async def token():
        return "entra-token"

    def handle(request):
        assert request.headers["authorization"] == "Bearer entra-token"
        assert "api-key" not in request.headers
        return httpx2.Response(200, json={"answers": ANSWERS})

    await invoke(True, handle, "azure_ai", api_key=None, azure_ad_token_provider=token)

    def failed_token():
        raise RuntimeError("secret-key private-input")

    with pytest.raises(litelm.AuthenticationError, match="token provider failed") as caught:
        await invoke(
            True,
            lambda request: pytest.fail("Token failure reached transport"),
            "azure_ai",
            api_key=None,
            azure_ad_token_provider=failed_token,
        )
    assert caught.value.__context__ is None and caught.value.__cause__ is None
