"""Native decision contracts through HTTPX2, without provider SDK or network mocks."""

import json
import traceback

import httpx2
import pytest

import litelm

QUESTIONS = {
    "flag": {"type": "noul", "instructions": "Ist der Text relevant?"},
    "label": {"type": "choice", "criteria": {"a": "Erste Option", "b": "Zweite Option"}},
    "rating": {"type": "score", "criteria": ["Schlecht", "Mittel", "Gut"]},
}
ANSWERS = {
    "flag": {"type": "noul", "noul": 0.7},
    "label": {"type": "choice", "choice": "a", "confidence": 0.6, "probabilities": {"a": 0.8, "b": 0.2}},
    "rating": {
        "type": "score",
        "score": 1.5,
        "confidence": 0.6,
        "legend": {"0": "Schlecht", "1": "Mittel", "2": "Gut"},
        "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
    },
}


async def invoke(async_call, handle, provider="typesafe", **overrides):
    params = dict(
        model=f"{provider}/deployment",
        state={"text": "Beispiel"},
        questions=QUESTIONS,
        api_key="secret-key",
        api_base="https://provider.test",
    )
    params.update(overrides)
    cls = httpx2.AsyncClient if async_call else httpx2.Client
    client = cls(transport=httpx2.MockTransport(handle))
    try:
        if async_call:
            return await litelm.asystemone(**params, http_client=client)
        return litelm.systemone(**params, http_client=client)
    finally:
        if async_call:
            await client.aclose()
        else:
            client.close()


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    "provider,path,auth",
    [
        ("typesafe", "/v1/systemone", "authorization"),
        ("azure_ai", "/providers/microsoft/v1/systemone", "api-key"),
    ],
)
@pytest.mark.asyncio
async def test_wire_contract_and_typed_roundtrip(async_call, provider, path, auth):
    def handle(request):
        assert request.url.path == path
        assert request.headers[auth] == ("Bearer secret-key" if auth == "authorization" else "secret-key")
        assert json.loads(request.content) == {
            "model": "deployment",
            "state": {"text": "Beispiel"},
            "questions": QUESTIONS,
        }
        return httpx2.Response(200, json={"model": "underlying-model", "answers": ANSWERS})

    response = await invoke(async_call, handle, provider)
    assert isinstance(response, litelm.SystemOneResponse)
    assert response.answers["flag"].noul == 0.7
    assert response.answers["label"].probabilities == {"a": 0.8, "b": 0.2}
    assert response.answers["rating"].score == 1.5
    assert response.usage is None  # Missing usage is not zero-cost inference.
    assert litelm.SystemOneResponse.model_validate(response.model_dump()).answers == response.answers


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    "status,error",
    [
        (400, litelm.BadRequestError),
        (401, litelm.AuthenticationError),
        (403, litelm.PermissionDeniedError),
        (404, litelm.NotFoundError),
        (422, litelm.UnprocessableEntityError),
        (429, litelm.RateLimitError),
        (500, litelm.InternalServerError),
        (503, litelm.ServiceUnavailableError),
    ],
)
@pytest.mark.asyncio
async def test_sanitized_errors_and_no_implicit_retry(async_call, status, error):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx2.Response(status, text="private-input secret-key", headers={"retry-after": "2"})

    with pytest.raises(error) as caught:
        await invoke(async_call, handle)
    assert len(calls) == 1
    rendered = "".join(traceback.format_exception(caught.value))
    assert "private-input" not in rendered and "secret-key" not in rendered
    assert caught.value.body is None and caught.value.request is None
    assert caught.value.response.headers == {"retry-after": "2"}


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.asyncio
async def test_explicit_retry_refreshes_entra_token(async_call, monkeypatch):
    calls, tokens = [], iter(["first-token", "second-token"])

    def handle(request):
        calls.append(request.headers["authorization"])
        return httpx2.Response(429 if len(calls) == 1 else 200, json={"answers": ANSWERS}, headers={"retry-after": "0"})

    response = await invoke(
        async_call, handle, "azure_ai", api_key=None, azure_ad_token_provider=lambda: next(tokens), num_retries=1
    )
    assert response.answers["flag"].noul == 0.7
    assert calls == ["Bearer first-token", "Bearer second-token"]


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["answers"].pop("flag"),
        lambda d: d["answers"]["flag"].update(noul=True),
        lambda d: d["answers"]["flag"].update(noul=float("nan")),
        lambda d: d["answers"]["label"].update(choice="unknown"),
        lambda d: d["answers"]["label"].update(probabilities={"a": 0.8}),
        lambda d: d["answers"]["label"].update(probabilities={"a": 0.8, "b": 0.8}),
        lambda d: d["answers"]["rating"].update(probabilities={"0": 0.4, "1": 0.6}),
    ],
)
@pytest.mark.asyncio
async def test_malformed_success_is_not_retried(async_call, mutate):
    calls = []

    def handle(request):
        calls.append(request)
        payload = json.loads(json.dumps({"answers": ANSWERS}))
        mutate(payload)
        return httpx2.Response(200, content=json.dumps(payload).encode())

    with pytest.raises(litelm.APIResponseValidationError):
        await invoke(async_call, handle, num_retries=1)
    assert len(calls) == 1


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.asyncio
async def test_invalid_request_fails_before_transport(async_call):
    def forbidden(request):
        pytest.fail("Invalid questions reached the provider")

    with pytest.raises(litelm.BadRequestError):
        await invoke(async_call, forbidden, questions={"bad": {"type": "choice", "criteria": {}}})


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.asyncio
async def test_unsupported_generation_controls_are_not_silently_ignored(async_call):
    def forbidden(request):
        pytest.fail("Unsupported control reached provider")

    with pytest.raises(TypeError):
        await invoke(async_call, forbidden, temperature=0)


def test_native_operation_does_not_change_generic_provider_routing():
    from litelm._providers import parse_model

    for provider in ("typesafe", "azure_ai"):
        model = f"{provider}/deployment"
        assert litelm.supports_systemone(model)
        assert parse_model(model, api_base="https://provider.test", api_key="key")[:2] == (provider, "deployment")
        assert litelm.get_llm_provider(model, api_base="https://provider.test", api_key="key")[:2] == (
            "deployment",
            provider,
        )
