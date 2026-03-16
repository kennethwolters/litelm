"""OpenAI Responses API wrapper."""

from litelm._client_cache import get_async_client, get_sync_client
from litelm._completion import _map_openai_error, _openai_errors
from litelm._dispatch import get_handler
from litelm._providers import parse_model


def responses(
    model,
    *,
    input=None,
    previous_response_id=None,
    cache=None,
    num_retries=0,
    retry_strategy=None,
    headers=None,
    shared_session=None,
    **kwargs,
):
    """Synchronous responses API call."""
    kwargs.pop("caching", None)
    api_key = kwargs.pop("api_key", None)
    api_base = kwargs.pop("api_base", None) or kwargs.pop("base_url", None)
    provider, model_name, base_url, resolved_api_key, api_version = parse_model(
        model, api_key=api_key, api_base=api_base
    )

    if input is not None:
        kwargs["input"] = input
    if previous_response_id is not None:
        kwargs["previous_response_id"] = previous_response_id

    handler = get_handler(provider)
    if handler and hasattr(handler, "responses"):
        return handler.responses(model_name, api_key=resolved_api_key, base_url=base_url, headers=headers, **kwargs)

    client = get_sync_client(provider, base_url, resolved_api_key, max_retries=num_retries, api_version=api_version)
    extra_headers = headers or None
    try:
        return client.responses.create(model=model_name, extra_headers=extra_headers, **kwargs)
    except _openai_errors as e:
        _map_openai_error(e)


async def aresponses(
    model,
    *,
    input=None,
    previous_response_id=None,
    cache=None,
    num_retries=0,
    retry_strategy=None,
    headers=None,
    shared_session=None,
    **kwargs,
):
    """Async responses API call."""
    kwargs.pop("caching", None)
    api_key = kwargs.pop("api_key", None)
    api_base = kwargs.pop("api_base", None) or kwargs.pop("base_url", None)
    provider, model_name, base_url, resolved_api_key, api_version = parse_model(
        model, api_key=api_key, api_base=api_base
    )

    if input is not None:
        kwargs["input"] = input
    if previous_response_id is not None:
        kwargs["previous_response_id"] = previous_response_id

    handler = get_handler(provider)
    if handler and hasattr(handler, "aresponses"):
        return await handler.aresponses(
            model_name, api_key=resolved_api_key, base_url=base_url, headers=headers, **kwargs
        )

    client = get_async_client(provider, base_url, resolved_api_key, max_retries=num_retries, api_version=api_version)
    extra_headers = headers or None
    try:
        return await client.responses.create(model=model_name, extra_headers=extra_headers, **kwargs)
    except _openai_errors as e:
        _map_openai_error(e)
