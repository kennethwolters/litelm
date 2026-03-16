"""Tests for Bedrock error mapping — openai exceptions → litelm exceptions."""

from unittest import mock

import pytest

from litelm._exceptions import (
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    RateLimitError,
    Timeout,
)


def _make_openai_error(cls_name, message="test error"):
    """Create a mock openai exception with the given class name."""
    openai = pytest.importorskip("openai")
    cls = getattr(openai, cls_name)
    # openai errors need specific constructor args
    if cls_name == "APITimeoutError":
        return cls(request=mock.MagicMock())
    if cls_name == "APIConnectionError":
        return cls(request=mock.MagicMock())
    return cls(
        message=message,
        response=mock.MagicMock(status_code=400),
        body={"error": {"message": message}},
    )


class TestBedrockErrorMapping:
    def setup_method(self):
        from litelm.providers import _bedrock

        self.mod = _bedrock

    def test_bad_request_maps(self):
        err = _make_openai_error("BadRequestError", "invalid request")
        with pytest.raises(BadRequestError, match="invalid request"):
            self.mod._map_error(err)

    def test_rate_limit_maps(self):
        err = _make_openai_error("RateLimitError", "rate limited")
        with pytest.raises(RateLimitError):
            self.mod._map_error(err)

    def test_auth_error_maps(self):
        err = _make_openai_error("AuthenticationError", "bad key")
        with pytest.raises(AuthenticationError):
            self.mod._map_error(err)

    def test_timeout_maps(self):
        err = _make_openai_error("APITimeoutError")
        with pytest.raises(Timeout):
            self.mod._map_error(err)

    def test_context_window_from_bad_request(self):
        err = _make_openai_error("BadRequestError", "maximum context length exceeded")
        with pytest.raises(ContextWindowExceededError):
            self.mod._map_error(err)

    def test_context_window_token_keyword(self):
        err = _make_openai_error("BadRequestError", "too many tokens in request")
        with pytest.raises(ContextWindowExceededError):
            self.mod._map_error(err)
