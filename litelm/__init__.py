"""litelm — thin OpenAI-SDK wrapper replacing litellm in DSPy."""

__version__ = "0.1.0"

from litelm._client_cache import close_async_clients as close_litelm_async_clients
from litelm._completion import acompletion, completion, stream_chunk_builder
from litelm._embedding import aembedding, embedding
from litelm._text_completion import atext_completion, text_completion
from litelm._exceptions import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadGatewayError,
    BadRequestError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    UnprocessableEntityError,
)
from litelm._responses import aresponses, responses
from litelm._types import (
    ChatCompletionMessageToolCall,
    Choices,
    Delta,
    Function,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)


# Capability stubs — always return True (DSPy only targets OpenAI-compatible providers)
def supports_function_calling(model=None, **kwargs):
    return True


def supports_response_schema(model=None, custom_llm_provider=None, **kwargs):
    return True


def supports_reasoning(model=None, **kwargs):
    """Check if model supports reasoning based on model name patterns."""
    if model is None:
        return False
    model_lower = model.lower()
    # Extract model name after provider prefix
    if "/" in model_lower:
        model_lower = model_lower.split("/", 1)[1]
    # Known reasoning model patterns
    reasoning_prefixes = ("o1", "o3", "o4", "gpt-5", "deepseek-reasoner")
    if any(model_lower.startswith(p) for p in reasoning_prefixes):
        return True
    # Claude 3.7+ models support extended thinking
    if "claude" in model_lower and any(v in model_lower for v in ("3-7", "3.7", "claude-4", "claude-5")):
        return True
    return False


def get_supported_openai_params(model=None, custom_llm_provider=None, **kwargs):
    """Return list of supported OpenAI params for known providers. None for unknown."""
    from litelm._providers import PROVIDERS

    provider = custom_llm_provider
    if provider is None and model and "/" in model:
        provider = model.split("/", 1)[0]
    if provider and provider not in PROVIDERS:
        return None
    return [
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "stream",
        "tools",
        "tool_choice",
        "response_format",
        "seed",
        "logprobs",
        "top_logprobs",
        "n",
    ]



# DSPy compatibility — sets these at startup
telemetry = False
cache = None
suppress_debug_info = False
add_function_to_prompt = False


def get_secret(secret_name, **kwargs):
    """Retrieve secret from env var."""
    import os
    return os.environ.get(secret_name)

__all__ = [
    "acompletion",
    "aembedding",
    "aresponses",
    "atext_completion",
    "APIConnectionError",
    "APIStatusError",
    "AuthenticationError",
    "BadGatewayError",
    "BadRequestError",
    "close_litelm_async_clients",
    "ContentPolicyViolationError",
    "completion",
    "ContextWindowExceededError",
    "embedding",
    "get_secret",
    "get_supported_openai_params",
    "InternalServerError",
    "ModelResponse",
    "ModelResponseStream",
    "NotFoundError",
    "PermissionDeniedError",
    "RateLimitError",
    "ServiceUnavailableError",
    "UnprocessableEntityError",
    "responses",
    "stream_chunk_builder",
    "supports_function_calling",
    "supports_reasoning",
    "supports_response_schema",
    "text_completion",
    "Timeout",
]
