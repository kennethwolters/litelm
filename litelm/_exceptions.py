"""Exception types — own hierarchy, no openai dependency."""


class _DefaultResponse:
    """Minimal response stub when no real response is provided."""
    __slots__ = ("status_code", "headers")

    def __init__(self, status_code=0):
        self.status_code = status_code
        self.headers = {}


class LitelmError(Exception):
    """Base exception for all litelm errors."""

    _default_status_code = 0

    def __init__(self, message="", response=None, body=None, request=None, **kwargs):
        self.message = message
        if response is None:
            response = _DefaultResponse(self._default_status_code)
        self.response = response
        self.body = body
        self.request = request
        self.status_code = getattr(response, "status_code", self._default_status_code) or self._default_status_code
        self.llm_provider = kwargs.get("llm_provider")
        self.model = kwargs.get("model")
        self.max_retries = kwargs.get("max_retries")
        self.num_retries = kwargs.get("num_retries")
        super().__init__(message)

    def __str__(self):
        cls = type(self).__name__
        parts = [f"{cls}: {self.message}"]
        if self.num_retries is not None:
            parts.append(f"LiteLLM Retried: {self.num_retries} times")
        if self.max_retries is not None:
            parts.append(f"LiteLLM Max Retries: {self.max_retries}")
        return ", ".join(parts) if len(parts) > 1 else parts[0]


class RateLimitError(LitelmError):
    _default_status_code = 429


class AuthenticationError(LitelmError):
    _default_status_code = 401


class BadRequestError(LitelmError):
    _default_status_code = 400


class Timeout(LitelmError):
    _default_status_code = 408


class InternalServerError(LitelmError):
    _default_status_code = 500


class APIConnectionError(LitelmError):
    pass


class APIStatusError(LitelmError):
    pass


class PermissionDeniedError(LitelmError):
    _default_status_code = 403


class NotFoundError(LitelmError):
    _default_status_code = 404


class UnprocessableEntityError(LitelmError):
    _default_status_code = 422


class ServiceUnavailableError(LitelmError):
    _default_status_code = 503


class ContentPolicyViolationError(BadRequestError):
    pass


class BadGatewayError(LitelmError):
    _default_status_code = 502


class ContextWindowExceededError(BadRequestError):
    """Raised when a request exceeds the model's context window."""

    def __init__(self, message="Context window exceeded", *args, **kwargs):
        # Accept and ignore extra positional args (llm_provider, model) for litellm compat
        kwargs.pop("request", None)
        super().__init__(message=message, **kwargs)


LITELLM_EXCEPTION_TYPES = [
    LitelmError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
    Timeout,
    InternalServerError,
    APIConnectionError,
    APIStatusError,
    PermissionDeniedError,
    NotFoundError,
    UnprocessableEntityError,
    ServiceUnavailableError,
    ContentPolicyViolationError,
    BadGatewayError,
    ContextWindowExceededError,
]


def is_context_window_error(msg: str) -> bool:
    """Check if an error message indicates a context window / token limit error."""
    low = msg.lower()
    return any(kw in low for kw in ("context", "token", "length", "too long"))
