"""Tests for responses() and aresponses() — mocked at the client level."""

import asyncio
from unittest import mock

from litelm._responses import responses, aresponses


@mock.patch("litelm._responses.get_sync_client")
def test_responses_sync(mock_get_client):
    mock_client = mock.MagicMock()
    mock_client.responses.create.return_value = {"id": "resp-1", "output": "hello"}
    mock_get_client.return_value = mock_client

    result = responses("openai/gpt-4o", input="hi", api_key="sk-test")
    mock_client.responses.create.assert_called_once()
    assert result == {"id": "resp-1", "output": "hello"}


@mock.patch("litelm._responses.get_async_client")
def test_aresponses_async(mock_get_client):
    mock_client = mock.MagicMock()
    mock_client.responses.create = mock.AsyncMock(return_value={"id": "resp-2"})
    mock_get_client.return_value = mock_client

    result = asyncio.run(aresponses("openai/gpt-4o", input="hi", api_key="sk-test"))
    mock_client.responses.create.assert_called_once()
    assert result == {"id": "resp-2"}


@mock.patch("litelm._responses.get_sync_client")
def test_responses_num_retries(mock_get_client):
    mock_client = mock.MagicMock()
    mock_client.responses.create.return_value = {}
    mock_get_client.return_value = mock_client

    responses("openai/gpt-4o", input="hi", num_retries=5, api_key="sk-test")
    mock_get_client.assert_called_once()
    assert mock_get_client.call_args.kwargs.get("max_retries") == 5 or mock_get_client.call_args[1].get("max_retries") == 5


@mock.patch("litelm._responses.get_handler")
def test_responses_routes_to_handler(mock_get_handler):
    handler = mock.MagicMock()
    handler.responses.return_value = "handler_resp"
    mock_get_handler.return_value = handler

    result = responses("anthropic/claude-sonnet-4-20250514", input="test", api_key="sk-test")
    assert result == "handler_resp"
    handler.responses.assert_called_once()


@mock.patch("litelm._responses.get_sync_client")
def test_responses_strips_caching_kwargs(mock_get_client):
    mock_client = mock.MagicMock()
    mock_client.responses.create.return_value = {}
    mock_get_client.return_value = mock_client

    responses("openai/gpt-4o", input="hi", caching=True, api_key="sk-test")
    call_kwargs = mock_client.responses.create.call_args
    assert "caching" not in (call_kwargs.kwargs if call_kwargs.kwargs else {})
