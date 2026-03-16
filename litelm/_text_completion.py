"""Text completion (legacy completions API)."""

from litelm._client_cache import get_async_client, get_sync_client
from litelm._completion import (
    _bad_request_errors,
    _map_openai_error,
    _openai_errors,
    _prepare_call,
    _wrap_context_window_error,
)


class TextCompletionResponse:
    """Wraps OpenAI Completion response with DSPy-expected attrs."""

    __slots__ = ("_completion", "cache_hit", "_hidden_params")

    def __init__(self, completion, cache_hit=False):
        self._completion = completion
        self.cache_hit = cache_hit
        self._hidden_params = {}

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._completion, name)


def _strip_text_prefix(model):
    """Strip text-completion-openai/ prefix → openai/ for parse_model."""
    if model.startswith("text-completion-openai/"):
        return "openai/" + model[len("text-completion-openai/") :]
    return model


def text_completion(model, prompt, *, timeout=None, shared_session=None, **kwargs):
    """Synchronous text completion (legacy completions API)."""
    mock = kwargs.pop("mock_response", None)
    model = _strip_text_prefix(model)
    provider, model_name, base_url, api_key, api_version, num_retries, kwargs = _prepare_call(model, kwargs)
    if mock is not None:
        from openai.types import Completion, CompletionChoice, CompletionUsage

        content = str(mock) if mock is not True else "mock"
        return TextCompletionResponse(
            Completion(
                id="mock",
                choices=[CompletionChoice(index=0, text=content, finish_reason="stop")],
                created=0,
                model=model_name,
                object="text_completion",
                usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )
        )
    client = get_sync_client(provider, base_url, api_key, max_retries=num_retries, api_version=api_version)
    sdk_kwargs = dict(model=model_name, prompt=prompt, **kwargs)
    if timeout is not None:
        sdk_kwargs["timeout"] = timeout
    try:
        response = client.completions.create(**sdk_kwargs)
    except _bad_request_errors as e:
        _wrap_context_window_error(e)
    except _openai_errors as e:
        _map_openai_error(e)
    return TextCompletionResponse(response)


async def atext_completion(model, prompt, *, timeout=None, shared_session=None, **kwargs):
    """Async text completion (legacy completions API)."""
    mock = kwargs.pop("mock_response", None)
    model = _strip_text_prefix(model)
    provider, model_name, base_url, api_key, api_version, num_retries, kwargs = _prepare_call(model, kwargs)
    if mock is not None:
        from openai.types import Completion, CompletionChoice, CompletionUsage

        content = str(mock) if mock is not True else "mock"
        return TextCompletionResponse(
            Completion(
                id="mock",
                choices=[CompletionChoice(index=0, text=content, finish_reason="stop")],
                created=0,
                model=model_name,
                object="text_completion",
                usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )
        )
    client = get_async_client(provider, base_url, api_key, max_retries=num_retries, api_version=api_version)
    sdk_kwargs = dict(model=model_name, prompt=prompt, **kwargs)
    if timeout is not None:
        sdk_kwargs["timeout"] = timeout
    try:
        response = await client.completions.create(**sdk_kwargs)
    except _bad_request_errors as e:
        _wrap_context_window_error(e)
    except _openai_errors as e:
        _map_openai_error(e)
    return TextCompletionResponse(response)
