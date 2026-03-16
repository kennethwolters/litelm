"""AWS Bedrock handler using boto3 for SigV4 auth and Bedrock's OpenAI-compat endpoint.

Model strings: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
"""

import os
import threading

from litelm._exceptions import (
    AuthenticationError,
    APIConnectionError,
    APIStatusError,
    BadRequestError,
    ContextWindowExceededError,
    InternalServerError,
    LitelmError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    Timeout,
    UnprocessableEntityError,
    is_context_window_error,
)
from litelm._types import ModelResponse, ModelResponseStream


def _get_boto3():
    try:
        import boto3
        return boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for bedrock/ models. "
            "Install it with: pip install litelm[bedrock]"
        )


def _require_openai():
    try:
        import openai
        return openai
    except ImportError:
        raise ImportError(
            "The openai SDK is required for bedrock/ models. "
            "Install it with: pip install litelm[bedrock]"
        )


def _get_bedrock_url(region=None):
    """Build the Bedrock OpenAI-compat base URL from region."""
    region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    return f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1"


_client_lock = threading.Lock()
_client_cache: dict[tuple, object] = {}


def _get_openai_client(base_url, region, async_client=False):
    """Get or create an OpenAI client with Bedrock auth via httpx auth hook."""
    region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    base_url = base_url or _get_bedrock_url(region)
    key = (base_url, region, async_client)

    if key not in _client_cache:
        with _client_lock:
            if key not in _client_cache:
                _client_cache[key] = _create_openai_client(base_url, region, async_client)
    return _client_cache[key]


def _create_openai_client(base_url, region, async_client):
    """Create a new OpenAI client with Bedrock SigV4 auth."""
    openai = _require_openai()
    _get_boto3()
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    import botocore.session
    import httpx

    session = botocore.session.get_session()
    credentials = session.get_credentials()
    if credentials is None:
        raise ValueError("No AWS credentials found. Configure AWS credentials for Bedrock access.")
    frozen_credentials = credentials.get_frozen_credentials()

    class SigV4AuthFlow(httpx.Auth):
        def auth_flow(self, request):
            aws_request = AWSRequest(
                method=request.method,
                url=str(request.url),
                headers=dict(request.headers),
                data=request.content,
            )
            SigV4Auth(frozen_credentials, "bedrock", region).add_auth(aws_request)
            for key, value in aws_request.headers.items():
                request.headers[key] = value
            yield request

    if async_client:
        http_client = httpx.AsyncClient(auth=SigV4AuthFlow())
        return openai.AsyncOpenAI(
            base_url=base_url,
            api_key="bedrock",  # dummy, SigV4 handles auth
            http_client=http_client,
        )
    else:
        http_client = httpx.Client(auth=SigV4AuthFlow())
        return openai.OpenAI(
            base_url=base_url,
            api_key="bedrock",  # dummy, SigV4 handles auth
            http_client=http_client,
        )


def _map_error(e):
    """Map openai SDK exceptions to litelm exception types."""
    openai = _require_openai()
    msg = str(e)

    if isinstance(e, openai.BadRequestError):
        if is_context_window_error(msg):
            raise ContextWindowExceededError(
                message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)
            ) from e
        raise BadRequestError(
            message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)
        ) from e
    elif isinstance(e, openai.RateLimitError):
        raise RateLimitError(
            message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)
        ) from e
    elif isinstance(e, openai.AuthenticationError):
        raise AuthenticationError(
            message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)
        ) from e
    elif isinstance(e, openai.APITimeoutError):
        raise Timeout(request=getattr(e, "request", None)) from e
    elif isinstance(e, openai.APIConnectionError):
        raise APIConnectionError(request=getattr(e, "request", None)) from e
    elif isinstance(e, openai.InternalServerError):
        raise InternalServerError(
            message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)
        ) from e
    elif isinstance(e, openai.NotFoundError):
        raise NotFoundError(
            message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)
        ) from e
    elif isinstance(e, openai.PermissionDeniedError):
        raise PermissionDeniedError(
            message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)
        ) from e
    elif isinstance(e, openai.UnprocessableEntityError):
        raise UnprocessableEntityError(
            message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)
        ) from e
    elif isinstance(e, openai.APIStatusError):
        raise APIStatusError(
            message=msg, response=getattr(e, "response", None), body=getattr(e, "body", None)
        ) from e
    raise LitelmError(message=msg) from e


def completion(model_name, messages, *, stream=False, api_key=None, base_url=None, timeout=None, **kwargs):
    """Synchronous Bedrock chat completion via OpenAI-compat endpoint."""
    openai = _require_openai()
    region = kwargs.pop("region", None)
    base_url = base_url or _get_bedrock_url(region)
    client = _get_openai_client(base_url, region, async_client=False)

    sdk_kwargs = dict(model=model_name, messages=messages, stream=stream, **kwargs)
    if timeout is not None:
        sdk_kwargs["timeout"] = timeout

    try:
        response = client.chat.completions.create(**sdk_kwargs)
    except openai.APIError as e:
        _map_error(e)

    if stream:
        return _wrap_stream_sync(response)
    return ModelResponse(response)


async def acompletion(model_name, messages, *, stream=False, api_key=None, base_url=None, timeout=None, **kwargs):
    """Async Bedrock chat completion via OpenAI-compat endpoint."""
    openai = _require_openai()
    region = kwargs.pop("region", None)
    base_url = base_url or _get_bedrock_url(region)
    client = _get_openai_client(base_url, region, async_client=True)

    sdk_kwargs = dict(model=model_name, messages=messages, stream=stream, **kwargs)
    if timeout is not None:
        sdk_kwargs["timeout"] = timeout

    try:
        response = await client.chat.completions.create(**sdk_kwargs)
    except openai.APIError as e:
        _map_error(e)

    if stream:
        return _wrap_stream_async(response)
    return ModelResponse(response)


def _wrap_stream_sync(stream):
    for chunk in stream:
        yield ModelResponseStream(chunk)


async def _wrap_stream_async(stream):
    async for chunk in stream:
        yield ModelResponseStream(chunk)
