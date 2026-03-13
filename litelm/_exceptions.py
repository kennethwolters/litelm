"""Exception types — own hierarchy, no openai dependency."""


class LitelmError(Exception):
    """Base exception for all litelm errors."""

    def __init__(self, message="", response=None, body=None, request=None):
        self.message = message
        self.response = response
        self.body = body
        self.request = request
        super().__init__(message)


class RateLimitError(LitelmError):
    pass


class AuthenticationError(LitelmError):
    pass


class BadRequestError(LitelmError):
    pass


class Timeout(LitelmError):
    pass


class InternalServerError(LitelmError):
    pass


class APIConnectionError(LitelmError):
    pass


class APIStatusError(LitelmError):
    pass


class PermissionDeniedError(LitelmError):
    pass


class NotFoundError(LitelmError):
    pass


class UnprocessableEntityError(LitelmError):
    pass


class ServiceUnavailableError(LitelmError):
    pass


class ContentPolicyViolationError(BadRequestError):
    pass


class BadGatewayError(LitelmError):
    pass


class ContextWindowExceededError(BadRequestError):
    """Raised when a request exceeds the model's context window."""

    def __init__(self, message="Context window exceeded", *args, **kwargs):
        # Accept and ignore extra positional args (llm_provider, model) for litellm compat
        kwargs.pop("request", None)
        super().__init__(message=message, **kwargs)


def is_context_window_error(msg: str) -> bool:
    """Check if an error message indicates a context window / token limit error."""
    low = msg.lower()
    return any(kw in low for kw in ("context", "token", "length", "too long"))
