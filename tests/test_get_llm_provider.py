"""Tests for get_llm_provider() public API."""

from litelm import get_llm_provider


def test_get_llm_provider_openai():
    model, provider, key, base = get_llm_provider("openai/gpt-4o")
    assert provider == "openai"
    assert model == "gpt-4o"


def test_get_llm_provider_anthropic():
    model, provider, key, base = get_llm_provider("anthropic/claude-3-haiku")
    assert provider == "anthropic"
    assert model == "claude-3-haiku"


def test_get_llm_provider_bare_model():
    model, provider, key, base = get_llm_provider("gpt-4o")
    assert provider == "openai"
    assert model == "gpt-4o"


def test_get_llm_provider_custom_provider_override():
    model, provider, key, base = get_llm_provider("gpt-4o", custom_llm_provider="azure")
    assert provider == "azure"


def test_get_llm_provider_api_base_passthrough():
    model, provider, key, base = get_llm_provider("openai/gpt-4o", api_base="http://custom:8000/v1")
    assert base == "http://custom:8000/v1"
