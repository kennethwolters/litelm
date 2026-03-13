"""Tests for embedding() and aembedding() — mocked at the client level."""

import asyncio
from unittest import mock

from litelm._embedding import embedding, aembedding


@mock.patch("litelm._embedding.get_sync_client")
def test_embedding_sync(mock_get_client):
    mock_client = mock.MagicMock()
    mock_client.embeddings.create.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
    mock_get_client.return_value = mock_client

    result = embedding("openai/text-embedding-3-small", input=["hello"], api_key="sk-test")
    mock_client.embeddings.create.assert_called_once_with(model="text-embedding-3-small", input=["hello"])
    assert result.data[0]["embedding"] == [0.1, 0.2]


@mock.patch("litelm._embedding.get_async_client")
def test_aembedding_async(mock_get_client):
    mock_client = mock.MagicMock()
    mock_client.embeddings.create = mock.AsyncMock(return_value={"data": [{"embedding": [0.3]}]})
    mock_get_client.return_value = mock_client

    result = asyncio.run(aembedding("openai/text-embedding-3-small", input=["hi"], api_key="sk-test"))
    mock_client.embeddings.create.assert_called_once()
    assert result.data[0]["embedding"] == [0.3]


@mock.patch("litelm._embedding.get_sync_client")
def test_embedding_strips_kwargs(mock_get_client):
    mock_client = mock.MagicMock()
    mock_client.embeddings.create.return_value = {}
    mock_get_client.return_value = mock_client

    embedding("openai/text-embedding-3-small", input=["x"], cache=True, caching=True, api_key="sk-test")
    # cache/caching should not reach the client
    call_kwargs = mock_client.embeddings.create.call_args
    assert "cache" not in call_kwargs.kwargs
    assert "caching" not in call_kwargs.kwargs


@mock.patch("litelm._embedding.get_handler")
def test_embedding_routes_to_handler(mock_get_handler):
    handler = mock.MagicMock()
    handler.embedding.return_value = "handler_result"
    mock_get_handler.return_value = handler

    result = embedding("anthropic/embed-model", input=["test"], api_key="sk-test")
    assert result == "handler_result"
    handler.embedding.assert_called_once()
