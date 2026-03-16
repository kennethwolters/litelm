"""Tests for exception types and compatibility."""

from litelm._exceptions import (
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    LitelmError,
    RateLimitError,
    Timeout,
    InternalServerError,
    APIConnectionError,
)


def test_exception_hierarchy():
    assert issubclass(RateLimitError, LitelmError)
    assert issubclass(AuthenticationError, LitelmError)
    assert issubclass(BadRequestError, LitelmError)
    assert issubclass(Timeout, LitelmError)
    assert issubclass(InternalServerError, LitelmError)
    assert issubclass(APIConnectionError, LitelmError)
    assert issubclass(ContextWindowExceededError, BadRequestError)


def test_exception_attributes():
    e = RateLimitError(message="rate limited", response="resp", body="body")
    assert e.message == "rate limited"
    assert e.response == "resp"
    assert e.body == "body"
    assert str(e) == "RateLimitError: rate limited"


def test_context_window_error_basic():
    e = ContextWindowExceededError("too long")
    assert str(e) == "ContextWindowExceededError: too long"


def test_context_window_error_default_message():
    e = ContextWindowExceededError()
    assert "Context window exceeded" in str(e)


def test_context_window_error_extra_kwargs():
    # Should accept and ignore litellm-style kwargs without raising
    e = ContextWindowExceededError("msg", response=object(), body={"error": "x"})
    assert str(e) == "ContextWindowExceededError: msg"
