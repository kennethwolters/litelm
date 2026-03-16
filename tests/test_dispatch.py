"""Tests for the dispatch module and provider handler translation logic."""

import os
from unittest import mock

import pytest

from litelm._dispatch import CUSTOM_HANDLERS, _loaded, get_handler


def test_get_handler_returns_none_for_openai_compat():
    assert get_handler("openai") is None
    assert get_handler("groq") is None
    assert get_handler("together") is None


def test_get_handler_returns_module_for_custom():
    # Clear cache
    _loaded.clear()
    handler = get_handler("anthropic")
    assert handler is not None
    assert hasattr(handler, "completion")
    assert hasattr(handler, "acompletion")


def test_get_handler_caches():
    _loaded.clear()
    h1 = get_handler("anthropic")
    h2 = get_handler("anthropic")
    assert h1 is h2


def test_custom_handlers_registry():
    assert "anthropic" in CUSTOM_HANDLERS
    assert "bedrock" in CUSTOM_HANDLERS
    assert "cloudflare" in CUSTOM_HANDLERS


# ---------------------------------------------------------------------------
# Anthropic translation unit tests
# ---------------------------------------------------------------------------


class TestAnthropicTranslation:
    def setup_method(self):
        from litelm.providers import _anthropic

        self.mod = _anthropic

    def test_extract_system_single(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        system, conv = self.mod._extract_system(msgs)
        assert system == [{"type": "text", "text": "You are helpful."}]
        assert len(conv) == 1
        assert conv[0]["role"] == "user"

    def test_extract_system_multiple(self):
        msgs = [
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
            {"role": "user", "content": "Hi"},
        ]
        system, conv = self.mod._extract_system(msgs)
        assert len(system) == 2
        assert system[0]["text"] == "Rule 1"
        assert system[1]["text"] == "Rule 2"

    def test_extract_system_none(self):
        msgs = [{"role": "user", "content": "Hi"}]
        system, conv = self.mod._extract_system(msgs)
        assert system is None
        assert len(conv) == 1

    def test_translate_content_string(self):
        result = self.mod._translate_content("hello")
        assert result == [{"type": "text", "text": "hello"}]

    def test_translate_content_list_text(self):
        result = self.mod._translate_content([{"type": "text", "text": "hello"}])
        assert result == [{"type": "text", "text": "hello"}]

    def test_translate_content_vision_base64(self):
        result = self.mod._translate_content(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc123"},
                }
            ]
        )
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["media_type"] == "image/png"
        assert result[0]["source"]["data"] == "abc123"

    def test_translate_content_vision_url(self):
        result = self.mod._translate_content(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/img.png"},
                }
            ]
        )
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "url"

    def test_translate_messages_basic(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Bye"},
        ]
        result = self.mod._translate_messages(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_translate_messages_tool_call(self):
        msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "72°F"},
        ]
        result = self.mod._translate_messages(msgs)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["type"] == "tool_use"
        assert result[0]["content"][0]["name"] == "get_weather"
        assert result[1]["role"] == "user"
        assert result[1]["content"][0]["type"] == "tool_result"

    def test_translate_messages_merges_consecutive_user(self):
        msgs = [
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
        ]
        result = self.mod._translate_messages(msgs)
        assert len(result) == 1
        assert len(result[0]["content"]) == 2

    def test_translate_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        result = self.mod._translate_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert "input_schema" in result[0]

    def test_translate_tool_choice_auto(self):
        assert self.mod._translate_tool_choice("auto") == {"type": "auto"}

    def test_translate_tool_choice_required(self):
        assert self.mod._translate_tool_choice("required") == {"type": "any"}

    def test_translate_tool_choice_specific(self):
        result = self.mod._translate_tool_choice({"function": {"name": "foo"}})
        assert result == {"type": "tool", "name": "foo"}

    def test_build_request_kwargs_basic(self):
        msgs = [{"role": "user", "content": "Hi"}]
        req = self.mod._build_request_kwargs("claude-sonnet-4-20250514", msgs, False, None, None)
        assert req["model"] == "claude-sonnet-4-20250514"
        assert req["max_tokens"] == 4096
        assert req["stream"] is False
        assert len(req["messages"]) == 1

    def test_build_request_kwargs_with_tools(self):
        msgs = [{"role": "user", "content": "Hi"}]
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        req = self.mod._build_request_kwargs("claude-sonnet-4-20250514", msgs, False, None, None, tools=tools)
        assert "tools" in req
        assert req["tools"][0]["name"] == "f"

    def test_build_request_kwargs_strips_openai_params(self):
        msgs = [{"role": "user", "content": "Hi"}]
        req = self.mod._build_request_kwargs(
            "claude-sonnet-4-20250514",
            msgs,
            False,
            None,
            None,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            seed=42,
            n=1,
        )
        assert "frequency_penalty" not in req
        assert "presence_penalty" not in req
        assert "seed" not in req
        assert "n" not in req


# ---------------------------------------------------------------------------
# Anthropic error mapping
# ---------------------------------------------------------------------------


class TestAnthropicErrorMapping:
    def setup_method(self):
        from litelm.providers import _anthropic

        self.mod = _anthropic

    def test_context_window_prompt_too_long(self):
        from litelm._exceptions import ContextWindowExceededError

        sdk = self.mod._get_sdk()
        err = sdk.BadRequestError(
            message="prompt is too long",
            response=mock.MagicMock(status_code=400),
            body={"error": {"message": "prompt is too long"}},
        )
        with pytest.raises(ContextWindowExceededError):
            self.mod._map_error(err)

    def test_context_window_max_tokens(self):
        from litelm._exceptions import ContextWindowExceededError

        sdk = self.mod._get_sdk()
        err = sdk.BadRequestError(
            message="max token limit exceeded",
            response=mock.MagicMock(status_code=400),
            body={"error": {"message": "max token limit exceeded"}},
        )
        with pytest.raises(ContextWindowExceededError):
            self.mod._map_error(err)

    def test_bad_request_no_context_window(self):
        from litelm._exceptions import BadRequestError

        sdk = self.mod._get_sdk()
        err = sdk.BadRequestError(
            message="invalid model parameter",
            response=mock.MagicMock(status_code=400),
            body={"error": {"message": "invalid model parameter"}},
        )
        with pytest.raises(BadRequestError):
            self.mod._map_error(err)


# ---------------------------------------------------------------------------
# Cloudflare translation unit tests
# ---------------------------------------------------------------------------


class TestCloudflareTranslation:
    def setup_method(self):
        from litelm.providers import _cloudflare

        self.mod = _cloudflare

    def test_build_url(self):
        url = self.mod._build_url("abc123", "@cf/meta/llama-2-7b-chat-int8")
        assert "abc123" in url
        assert "@cf/meta/llama-2-7b-chat-int8" in url

    def test_build_request_body_basic(self):
        msgs = [{"role": "user", "content": "Hi"}]
        body, stream = self.mod._build_request_body(msgs, temperature=0.7)
        assert body["messages"] == msgs
        assert body["temperature"] == 0.7
        assert stream is False

    def test_build_request_body_stream(self):
        msgs = [{"role": "user", "content": "Hi"}]
        body, stream = self.mod._build_request_body(msgs, stream=True)
        assert body["stream"] is True
        assert stream is True

    def test_parse_response(self):
        data = {"result": {"response": "Hello there!"}}
        resp = self.mod._parse_response(data, "test-model")
        assert resp.choices[0].message.content == "Hello there!"
        assert resp.model == "test-model"

    def test_parse_stream_line_data(self):
        line = 'data: {"response": "Hi"}'
        chunk = self.mod._parse_stream_line(line, "model", "id1")
        assert chunk is not None
        assert chunk.choices[0].delta.content == "Hi"

    def test_parse_stream_line_done(self):
        line = "data: [DONE]"
        chunk = self.mod._parse_stream_line(line, "model", "id1")
        assert chunk is not None
        assert chunk.choices[0].finish_reason == "stop"

    def test_parse_stream_line_empty(self):
        assert self.mod._parse_stream_line("", "model", "id1") is None
        assert self.mod._parse_stream_line("event: ping", "model", "id1") is None

    def test_get_config_missing_account_id(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID"):
                self.mod._get_config()


# ---------------------------------------------------------------------------
# Bedrock unit tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mistral translation unit tests
# ---------------------------------------------------------------------------


class TestMistralTransforms:
    def setup_method(self):
        from litelm.providers import _mistral

        self.mod = _mistral

    def test_transform_messages_strips_name_from_user(self):
        msgs = [{"role": "user", "content": "Hi", "name": "alice"}]
        result = self.mod._transform_messages(msgs)
        assert "name" not in result[0]
        assert result[0]["content"] == "Hi"

    def test_transform_messages_keeps_name_on_tool(self):
        msgs = [{"role": "tool", "content": "result", "name": "get_weather", "tool_call_id": "tc1"}]
        result = self.mod._transform_messages(msgs)
        assert result[0]["name"] == "get_weather"

    def test_transform_messages_strips_name_from_assistant(self):
        msgs = [{"role": "assistant", "content": "Hello", "name": "bot"}]
        result = self.mod._transform_messages(msgs)
        assert "name" not in result[0]

    def test_transform_messages_none_passthrough(self):
        assert self.mod._transform_messages(None) is None
        assert self.mod._transform_messages([]) == []

    def test_fix_response_null_type(self):
        """tool_call with type=None gets fixed to 'function'."""
        resp = mock.MagicMock()
        tc = mock.MagicMock()
        tc.type = None
        resp.choices = [mock.MagicMock()]
        resp.choices[0].message.content = "hi"
        resp.choices[0].message.tool_calls = [tc]
        self.mod._fix_response(resp)
        assert tc.type == "function"

    def test_fix_response_empty_content(self):
        """Empty string content gets converted to None."""
        resp = mock.MagicMock()
        resp.choices = [mock.MagicMock()]
        resp.choices[0].message.content = ""
        resp.choices[0].message.tool_calls = None
        self.mod._fix_response(resp)
        assert resp.choices[0].message.content is None

    def test_fix_response_preserves_normal_content(self):
        """Non-empty content is left alone."""
        resp = mock.MagicMock()
        resp.choices = [mock.MagicMock()]
        resp.choices[0].message.content = "hello"
        resp.choices[0].message.tool_calls = None
        self.mod._fix_response(resp)
        assert resp.choices[0].message.content == "hello"


# ---------------------------------------------------------------------------
# Bedrock unit tests
# ---------------------------------------------------------------------------


class TestBedrockHelpers:
    def setup_method(self):
        from litelm.providers import _bedrock

        self.mod = _bedrock

    def test_get_bedrock_url_default_region(self):
        with mock.patch.dict(os.environ, {"AWS_REGION": "us-west-2"}, clear=False):
            url = self.mod._get_bedrock_url()
            assert "us-west-2" in url
            assert "bedrock-runtime" in url
            assert "/openai/v1" in url

    def test_get_bedrock_url_explicit_region(self):
        url = self.mod._get_bedrock_url("eu-west-1")
        assert "eu-west-1" in url
