"""Direct System One transport for TypeSafe and Microsoft Foundry decision models."""

import asyncio
import inspect
import json
import math
import os
import time
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit

import httpx2
from pydantic import ValidationError

from litelm._decision_types import SystemOneRequest, SystemOneResponse
from litelm._exceptions import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    UnprocessableEntityError,
)

_MAX_REQUEST_BYTES = 2_000_000
_MAX_RESPONSE_BYTES = 1_000_000
_PROVIDERS = {"typesafe", "azure_ai"}
_RETRYABLE = {408, 429, 500, 502, 503, 504}


def supports_systemone(model=None, *, custom_llm_provider=None):
    """Whether this provider exposes the native decision operation (not chat)."""
    provider = custom_llm_provider or (model.split("/", 1)[0] if isinstance(model, str) else None)
    return provider in _PROVIDERS


def _prepare(model, state, questions, api_base, timeout, num_retries):
    if not isinstance(model, str) or "/" not in model:
        raise BadRequestError("System One requires a provider/model identifier")
    provider, name = model.split("/", 1)
    if provider not in _PROVIDERS:
        raise BadRequestError("Unsupported System One provider")
    if type(num_retries) is not int or not 0 <= num_retries <= 10:
        raise BadRequestError("num_retries must be an integer from 0 to 10")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise BadRequestError("timeout must be a finite positive number")
    try:
        request = SystemOneRequest(model=name, state=state, questions=questions)
        # Preserve explicit nulls and emit default type discriminators for typed questions.
        payload = {
            "model": request.model,
            "state": request.state,
            "questions": {
                name: {"type": question.type, **question.model_dump(exclude_unset=True)}
                for name, question in request.questions.items()
            },
        }
        body = json.dumps(payload, allow_nan=False, ensure_ascii=False).encode()
        if len(body) > _MAX_REQUEST_BYTES:
            raise ValueError
    except (ValidationError, ValueError, TypeError):
        request = None
    if request is None:
        raise BadRequestError("Invalid or oversized System One request", model=model, llm_provider=provider)
    base = api_base or os.getenv(f"{provider.upper()}_API_BASE")
    base = base or ("https://api.typesafe.ai" if provider == "typesafe" else None)
    try:
        url = urlsplit(base or "")
        local_http = url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}
        if (url.scheme != "https" and not local_http) or not url.hostname:
            raise ValueError
        if url.username or url.password or url.query or url.fragment:
            raise ValueError
        path = url.path.rstrip("/")
        suffix = "/v1/systemone" if provider == "typesafe" else "/providers/microsoft/v1/systemone"
        if not path.endswith(suffix):
            if provider == "typesafe" and path.endswith("/v1"):
                path = path[:-3]
            if provider == "azure_ai" and path:
                # Foundry needs its resource endpoint, not a project or OpenAI route.
                raise ValueError
            path += suffix
        endpoint = urlunsplit((url.scheme, url.netloc, path, "", ""))
    except (ValueError, TypeError):
        pass
    else:
        return provider, request, endpoint, body
    raise BadRequestError("Use a credential-free resource API base or full System One endpoint")


def _credentials(provider, api_key, token_provider):
    if token_provider is not None:
        if provider != "azure_ai" or api_key is not None or not callable(token_provider):
            raise BadRequestError("Use either an Azure token provider or an API key")
        return None
    key = api_key if api_key is not None else os.getenv(f"{provider.upper()}_API_KEY")
    if not isinstance(key, str) or not key.strip():
        raise AuthenticationError("Missing System One credentials", llm_provider=provider)
    return key


def _headers(provider, key, token=None):
    credential = token if token is not None else key
    if not isinstance(credential, str) or not credential or any(not 33 <= ord(c) <= 126 for c in credential):
        raise AuthenticationError("Invalid System One credentials", llm_provider=provider)
    headers = {"Content-Type": "application/json", "Accept": "application/json", "Accept-Encoding": "identity"}
    if provider == "azure_ai" and token is None:
        headers["api-key"] = credential
    else:
        headers["Authorization"] = f"Bearer {credential}"
    return headers


def _token(callback):
    try:
        value = callback()
        if inspect.isawaitable(value):
            if inspect.iscoroutine(value):
                value.close()
            raise ValueError
        if not isinstance(value, str) or not value:
            raise ValueError
        return value
    except Exception:
        pass
    raise AuthenticationError("System One token provider failed")


def _raise_without_context(error):
    # Python 3.10 can retain a worker exception across an await even after its
    # except block. Clear it while unwinding, not only via `raise from None`.
    try:
        raise error from None
    finally:
        error.__context__ = None


async def _atoken(callback):
    try:
        value = await asyncio.to_thread(callback)
        if inspect.isawaitable(value):
            value = await value
        if not isinstance(value, str) or not value:
            raise ValueError
        return value
    except Exception:
        pass
    _raise_without_context(AuthenticationError("System One token provider failed"))


def _status_error(response, model, provider):
    status = response.status_code
    error = {
        400: BadRequestError,
        401: AuthenticationError,
        403: PermissionDeniedError,
        404: NotFoundError,
        408: Timeout,
        422: UnprocessableEntityError,
        429: RateLimitError,
        503: ServiceUnavailableError,
    }.get(status, InternalServerError if status >= 500 else APIStatusError)
    headers = {}
    retry_after = response.headers.get("retry-after")
    try:
        if retry_after is not None and math.isfinite(float(retry_after)) and 0 <= float(retry_after) <= 30:
            headers["retry-after"] = retry_after
    except ValueError:
        pass
    # Never attach the HTTP response/request, provider error body or arbitrary headers.
    return error(
        f"System One returned HTTP {status}",
        model=model,
        llm_provider=provider,
        response=SimpleNamespace(status_code=status, headers=headers),
    )


def _delay(error, attempt):
    return float(error.response.headers.get("retry-after", min(0.25 * 2**attempt, 8)))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _decode(data, request, provider):
    try:
        payload = json.loads(data, object_pairs_hook=_unique_object)
        return SystemOneResponse.model_validate(payload).validate_for(request)
    except (ValidationError, ValueError, TypeError, UnicodeDecodeError):
        pass
    raise APIResponseValidationError("Invalid System One response", model=request.model, llm_provider=provider)


def _check_encoding(response, request, provider):
    if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
        raise APIResponseValidationError(
            "Unsupported System One response encoding", model=request.model, llm_provider=provider
        )


def _too_large(request, provider):
    return APIResponseValidationError(
        "System One response exceeded byte limit", model=request.model, llm_provider=provider
    )


def systemone(
    model,
    *,
    state,
    questions,
    api_key=None,
    api_base=None,
    azure_ad_token_provider=None,
    timeout=30.0,
    num_retries=0,
    http_client=None,
):
    """Return typed noul/choice/score evidence without generating text.

    API keys use TYPESAFE_API_KEY or AZURE_AI_API_KEY; Foundry also requires
    AZURE_AI_API_BASE. An Azure Entra callback is refreshed on every attempt.
    Retries are explicit (0 by default, at most 10), and apply only to transport
    failures or HTTP 408/429/500/502/503/504, never malformed successes.
    Supplied HTTPX2 clients remain caller-owned; redirects are never followed.
    Request/response bodies are bounded to 2 MB/1 MB. No completion callbacks run.
    """
    provider, request, endpoint, body = _prepare(model, state, questions, api_base, timeout, num_retries)
    key = _credentials(provider, api_key, azure_ad_token_provider)
    if http_client is not None and not isinstance(http_client, httpx2.Client):
        raise BadRequestError("http_client must be an HTTPX2 Client")
    context = nullcontext(http_client) if http_client is not None else httpx2.Client()
    with context as client:
        for attempt in range(num_retries + 1):
            token = _token(azure_ad_token_provider) if azure_ad_token_provider is not None else None
            headers = _headers(provider, key, token)
            try:
                with client.stream(
                    "POST", endpoint, content=body, headers=headers, timeout=timeout, follow_redirects=False
                ) as response:
                    if response.status_code != 200:
                        error = _status_error(response, model, provider)
                    else:
                        _check_encoding(response, request, provider)
                        data = bytearray()
                        for chunk in response.iter_bytes():
                            if len(data) + len(chunk) > _MAX_RESPONSE_BYTES:
                                raise _too_large(request, provider)
                            data.extend(chunk)
                        return _decode(data, request, provider)
            except httpx2.TimeoutException:
                error = Timeout("System One request timed out", model=model, llm_provider=provider)
            except httpx2.RequestError:
                error = APIConnectionError("System One transport failed", model=model, llm_provider=provider)
            if attempt == num_retries or (
                error.status_code not in _RETRYABLE and not isinstance(error, APIConnectionError)
            ):
                raise error from None
            time.sleep(_delay(error, attempt))


@asynccontextmanager
async def _async_client(client):
    if client is not None:
        yield client
    else:
        async with httpx2.AsyncClient() as owned:
            yield owned


async def asystemone(
    model,
    *,
    state,
    questions,
    api_key=None,
    api_base=None,
    azure_ad_token_provider=None,
    timeout=30.0,
    num_retries=0,
    http_client=None,
):
    """Async equivalent of systemone; accepts sync or async Entra callbacks."""
    provider, request, endpoint, body = _prepare(model, state, questions, api_base, timeout, num_retries)
    key = _credentials(provider, api_key, azure_ad_token_provider)
    if http_client is not None and not isinstance(http_client, httpx2.AsyncClient):
        raise BadRequestError("http_client must be an HTTPX2 AsyncClient")
    async with _async_client(http_client) as client:
        for attempt in range(num_retries + 1):
            token = await _atoken(azure_ad_token_provider) if azure_ad_token_provider is not None else None
            headers = _headers(provider, key, token)
            try:
                async with client.stream(
                    "POST", endpoint, content=body, headers=headers, timeout=timeout, follow_redirects=False
                ) as response:
                    if response.status_code != 200:
                        error = _status_error(response, model, provider)
                    else:
                        _check_encoding(response, request, provider)
                        data = bytearray()
                        async for chunk in response.aiter_bytes():
                            if len(data) + len(chunk) > _MAX_RESPONSE_BYTES:
                                raise _too_large(request, provider)
                            data.extend(chunk)
                        return _decode(data, request, provider)
            except httpx2.TimeoutException:
                error = Timeout("System One request timed out", model=model, llm_provider=provider)
            except httpx2.RequestError:
                error = APIConnectionError("System One transport failed", model=model, llm_provider=provider)
            if attempt == num_retries or (
                error.status_code not in _RETRYABLE and not isinstance(error, APIConnectionError)
            ):
                raise error from None
            await asyncio.sleep(_delay(error, attempt))
