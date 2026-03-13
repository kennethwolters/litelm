"""Tests for capability detection stubs."""

import litelm


def test_supports_function_calling():
    assert litelm.supports_function_calling() is True
    assert litelm.supports_function_calling(model="gpt-4o") is True


def test_supports_response_schema():
    assert litelm.supports_response_schema() is True
    assert litelm.supports_response_schema(model="gpt-4o") is True


def test_supports_reasoning_openai():
    assert litelm.supports_reasoning(model="o1-preview") is True
    assert litelm.supports_reasoning(model="o3-mini") is True
    assert litelm.supports_reasoning(model="o4-mini") is True
    assert litelm.supports_reasoning(model="gpt-5") is True
    assert litelm.supports_reasoning(model="openai/o3-mini") is True


def test_supports_reasoning_anthropic():
    assert litelm.supports_reasoning(model="claude-3.7-sonnet") is True
    assert litelm.supports_reasoning(model="anthropic/claude-4-opus") is True
    assert litelm.supports_reasoning(model="anthropic/claude-5-sonnet") is True


def test_supports_reasoning_deepseek():
    assert litelm.supports_reasoning(model="deepseek-reasoner") is True


def test_supports_reasoning_false():
    assert litelm.supports_reasoning(model="gpt-4o") is False
    assert litelm.supports_reasoning(model="claude-3-5-sonnet") is False
    assert litelm.supports_reasoning(model=None) is False


def test_get_supported_openai_params_known():
    params = litelm.get_supported_openai_params(model="openai/gpt-4o")
    assert "temperature" in params
    assert "max_tokens" in params
    assert "tools" in params
    assert "stream" in params


def test_get_supported_openai_params_unknown():
    params = litelm.get_supported_openai_params(model="unknown_provider/model", custom_llm_provider="unknown_provider")
    assert params is None


def test_text_completion_exported():
    """text_completion is implemented and exported."""
    assert callable(litelm.text_completion)
    assert callable(litelm.atext_completion)
