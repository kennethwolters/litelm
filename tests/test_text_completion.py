"""Tests for TextCompletionResponse attribute handling."""

import copy

from litelm._text_completion import text_completion, TextCompletionResponse


def test_text_completion_response_deepcopy_usage_assignment():
    """deepcopy + usage reassignment must work (dspy-lite cache.py pattern)."""
    resp = text_completion("openai/gpt-4o-mini", prompt="hi", mock_response="hello")
    clone = copy.deepcopy(resp)
    clone.usage = {}
    assert clone.usage == {}


def test_text_completion_response_arbitrary_attr_assignment():
    """Arbitrary attr assignment must work (no __slots__ restriction)."""
    resp = text_completion("openai/gpt-4o-mini", prompt="hi", mock_response="hello")
    resp.foo = "bar"
    assert resp.foo == "bar"
