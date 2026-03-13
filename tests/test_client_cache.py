"""Tests for the client cache module."""

import asyncio
import threading
from unittest import mock

import pytest

from litelm._client_cache import (
    _async_clients,
    _lock,
    _sync_clients,
    close_async_clients,
    get_async_client,
    get_sync_client,
)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear client caches before and after each test."""
    _sync_clients.clear()
    _async_clients.clear()
    yield
    _sync_clients.clear()
    _async_clients.clear()


def test_sync_client_creation():
    """get_sync_client returns an OpenAI client."""
    client = get_sync_client("openai", None, "sk-test")
    assert client is not None


def test_sync_client_cache_hit():
    """Same params return the same cached client."""
    c1 = get_sync_client("openai", None, "sk-test")
    c2 = get_sync_client("openai", None, "sk-test")
    assert c1 is c2


def test_sync_client_cache_miss_different_key():
    """Different api_key returns a different client."""
    c1 = get_sync_client("openai", None, "sk-a")
    c2 = get_sync_client("openai", None, "sk-b")
    assert c1 is not c2


def test_sync_client_cache_miss_different_base_url():
    """Different base_url returns a different client."""
    c1 = get_sync_client("groq", "https://api.groq.com/openai/v1", "sk-test")
    c2 = get_sync_client("groq", "https://other.groq.com/v1", "sk-test")
    assert c1 is not c2


def test_sync_client_cache_miss_different_retries():
    """Different max_retries returns a different client."""
    c1 = get_sync_client("openai", None, "sk-test", max_retries=0)
    c2 = get_sync_client("openai", None, "sk-test", max_retries=3)
    assert c1 is not c2


def test_async_client_creation():
    """get_async_client returns an AsyncOpenAI client."""
    client = get_async_client("openai", None, "sk-test")
    assert client is not None


def test_async_client_cache_hit():
    """Same params return the same cached async client."""
    c1 = get_async_client("openai", None, "sk-test")
    c2 = get_async_client("openai", None, "sk-test")
    assert c1 is c2


def test_azure_sync_client():
    """Azure provider creates AzureOpenAI client."""
    import openai
    client = get_sync_client("azure", "https://my.azure.com", "sk-az", api_version="2024-02-01")
    assert isinstance(client, openai.AzureOpenAI)


def test_azure_async_client():
    """Azure provider creates AsyncAzureOpenAI client."""
    import openai
    client = get_async_client("azure", "https://my.azure.com", "sk-az", api_version="2024-02-01")
    assert isinstance(client, openai.AsyncAzureOpenAI)


def test_azure_sync_different_api_version():
    """Different api_version returns different client (Bug 1 regression)."""
    c1 = get_sync_client("azure", "https://test.openai.azure.com", "sk-az", api_version="2024-02-01")
    c2 = get_sync_client("azure", "https://test.openai.azure.com", "sk-az", api_version="2025-01-01")
    assert c1 is not c2


def test_azure_async_different_api_version():
    """Different api_version returns different async client (Bug 1 regression)."""
    c1 = get_async_client("azure", "https://test.openai.azure.com", "sk-az", api_version="2024-02-01")
    c2 = get_async_client("azure", "https://test.openai.azure.com", "sk-az", api_version="2025-01-01")
    assert c1 is not c2


def test_azure_same_api_version_cached():
    """Same api_version returns same cached client."""
    c1 = get_sync_client("azure", "https://test.openai.azure.com", "sk-az", api_version="2024-02-01")
    c2 = get_sync_client("azure", "https://test.openai.azure.com", "sk-az", api_version="2024-02-01")
    assert c1 is c2


@pytest.mark.asyncio
async def test_close_async_clients():
    """close_async_clients clears the async cache."""
    get_async_client("openai", None, "sk-test")
    assert len(_async_clients) == 1
    await close_async_clients()
    assert len(_async_clients) == 0
