"""Tests for provider registry and model string parsing."""

import os
from unittest import mock

import pytest

from litelm._providers import PROVIDERS, parse_model


def test_known_providers_exist():
    expected = [
        "openai",
        "azure",
        "openrouter",
        "groq",
        "together_ai",
        "together",
        "fireworks_ai",
        "fireworks",
        "mistral",
        "deepseek",
        "perplexity",
        "xai",
        "deepinfra",
        "gemini",
        "google",
        "cohere",
        "anthropic",
        "bedrock",
        "cloudflare",
        "ollama",
        "vllm",
        "lm_studio",
        "lmstudio",
    ]
    for p in expected:
        assert p in PROVIDERS, f"Missing provider: {p}"


def test_parse_openai_prefixed():
    provider, model, base_url, api_key, api_version = parse_model("openai/gpt-4o", api_key="sk-test")
    assert provider == "openai"
    assert model == "gpt-4o"
    assert base_url is None  # OpenAI uses default
    assert api_key == "sk-test"
    assert api_version is None


def test_parse_bare_model_defaults_to_openai():
    provider, model, *_ = parse_model("gpt-4o", api_key="sk-test")
    assert provider == "openai"
    assert model == "gpt-4o"


def test_parse_groq():
    with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk-test"}):
        provider, model, base_url, api_key, _ = parse_model("groq/llama-3.1-70b")
    assert provider == "groq"
    assert model == "llama-3.1-70b"
    assert "groq.com" in base_url
    assert api_key == "gsk-test"


def test_parse_azure():
    with mock.patch.dict(
        os.environ,
        {
            "AZURE_API_BASE": "https://my-resource.openai.azure.com",
            "AZURE_API_KEY": "az-key",
        },
    ):
        provider, model, base_url, api_key, api_version = parse_model("azure/gpt-4o")
    assert provider == "azure"
    assert model == "gpt-4o"
    assert "azure" in base_url
    assert api_key == "az-key"
    assert api_version is not None


def test_parse_unknown_provider():
    provider, model, base_url, api_key, _ = parse_model(
        "custom/my-model", api_base="https://my.api.com/v1", api_key="key"
    )
    assert provider == "custom"
    assert model == "my-model"
    assert base_url == "https://my.api.com/v1"


def test_openrouter_preserves_model_namespace():
    with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "router-key"}, clear=True):
        assert parse_model("openrouter/z-ai/glm-5.3-flash") == (
            "openrouter",
            "z-ai/glm-5.3-flash",
            "https://openrouter.ai/api/v1",
            "router-key",
            None,
        )


@pytest.mark.parametrize("provider", ["ollama", "vllm", "lm_studio", "lmstudio"])
def test_local_provider_without_credentials(provider):
    with mock.patch.dict(os.environ, {}, clear=True):
        assert parse_model(f"{provider}/model")[3] == "local"


@pytest.mark.parametrize("provider", ["ollama", "vllm", "lm_studio", "lmstudio"])
def test_local_provider_respects_credentials(provider):
    key_env = PROVIDERS[provider][1]
    with mock.patch.dict(os.environ, {key_env: "env-key"}, clear=True):
        assert parse_model(f"{provider}/model")[3] == "env-key"
        assert parse_model(f"{provider}/model", api_key="explicit-key")[3] == "explicit-key"


def test_bedrock_does_not_look_up_a_none_env_var():
    assert parse_model("bedrock/model") == ("bedrock", "model", None, None, None)


def test_unknown_provider_accepts_base_url_alias():
    assert parse_model("custom/model", base_url="https://example.test/v1", api_key="key") == (
        "custom",
        "model",
        "https://example.test/v1",
        "key",
        None,
    )


def test_api_base_kwarg_overrides_registry():
    provider, model, base_url, api_key, _ = parse_model(
        "groq/llama-3.1-70b", api_base="https://custom.endpoint/v1", api_key="k"
    )
    assert base_url == "https://custom.endpoint/v1"
