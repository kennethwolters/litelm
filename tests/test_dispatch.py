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

    # -- _get_max_tokens tests --

    def test_get_max_tokens_opus_4(self):
        assert self.mod._get_max_tokens("claude-opus-4-20250514") == 32768

    def test_get_max_tokens_sonnet_4(self):
        assert self.mod._get_max_tokens("claude-sonnet-4-20250514") == 16384

    def test_get_max_tokens_3_7_sonnet(self):
        assert self.mod._get_max_tokens("claude-3-7-sonnet-20250219") == 16384

    def test_get_max_tokens_3_5_sonnet(self):
        assert self.mod._get_max_tokens("claude-3-5-sonnet-20241022") == 8192

    def test_get_max_tokens_3_5_haiku(self):
        assert self.mod._get_max_tokens("claude-3-5-haiku-20241022") == 8192

    def test_get_max_tokens_3_opus(self):
        assert self.mod._get_max_tokens("claude-3-opus-20240229") == 4096

    def test_get_max_tokens_3_haiku(self):
        assert self.mod._get_max_tokens("claude-3-haiku-20240307") == 4096

    def test_get_max_tokens_unknown(self):
        assert self.mod._get_max_tokens("claude-99-mega") == 4096

    # -- _build_request_kwargs tests --

    def test_build_request_kwargs_basic(self):
        msgs = [{"role": "user", "content": "Hi"}]
        req = self.mod._build_request_kwargs("claude-sonnet-4-20250514", msgs, False, None, None)
        assert req["model"] == "claude-sonnet-4-20250514"
        assert req["max_tokens"] == 16384
        assert req["stream"] is False
        assert len(req["messages"]) == 1

    def test_build_request_kwargs_opus4_default(self):
        msgs = [{"role": "user", "content": "Hi"}]
        req = self.mod._build_request_kwargs("claude-opus-4-20250514", msgs, False, None, None)
        assert req["max_tokens"] == 32768

    def test_build_request_kwargs_haiku3_default(self):
        msgs = [{"role": "user", "content": "Hi"}]
        req = self.mod._build_request_kwargs("claude-3-haiku-20240307", msgs, False, None, None)
        assert req["max_tokens"] == 4096

    def test_build_request_kwargs_user_max_tokens_override(self):
        msgs = [{"role": "user", "content": "Hi"}]
        req = self.mod._build_request_kwargs("claude-opus-4-20250514", msgs, False, None, None, max_tokens=1024)
        assert req["max_tokens"] == 1024

    def test_build_request_kwargs_max_completion_tokens_override(self):
        msgs = [{"role": "user", "content": "Hi"}]
        req = self.mod._build_request_kwargs("claude-opus-4-20250514", msgs, False, None, None, max_completion_tokens=2048)
        assert req["max_tokens"] == 2048

    def test_build_request_kwargs_with_tools(self):
        msgs = [{"role": "user", "content": "Hi"}]
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        req = self.mod._build_request_kwargs("claude-sonnet-4-20250514", msgs, False, None, None, tools=tools)
        assert "tools" in req
        assert req["tools"][0]["name"] == "f"

    def test_build_model_response_with_thinking_blocks(self):
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.id = "msg_123"
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 50
        mock_response.usage.cache_creation_input_tokens = None
        mock_response.usage.cache_read_input_tokens = None

        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "Let me reason..."
        thinking_block.signature = "sig_abc"

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Answer is 42."

        mock_response.content = [thinking_block, text_block]

        result = self.mod._build_model_response(mock_response)
        msg = result.choices[0].message
        assert msg.content == "Answer is 42."
        assert msg.reasoning_content == "Let me reason..."
        assert msg.thinking_blocks == [
            {"type": "thinking", "thinking": "Let me reason...", "signature": "sig_abc"},
        ]

    def test_build_model_response_cache_tokens(self):
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.id = "msg_123"
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.usage.cache_creation_input_tokens = 80
        mock_response.usage.cache_read_input_tokens = 20
        mock_response.content = [MagicMock(type="text", text="Hello")]

        result = self.mod._build_model_response(mock_response)
        assert result.usage.prompt_tokens_details == {
            "cached_tokens": 20,
            "cache_creation_tokens": 80,
        }

    def test_build_stream_chunk_thinking_block_start(self):
        from unittest.mock import MagicMock

        event = MagicMock()
        event.type = "content_block_start"
        event.index = 0
        event.content_block.type = "thinking"
        chunk = self.mod._build_stream_chunk(event, "claude-sonnet-4", "c1")
        assert chunk is not None
        assert chunk.choices[0].delta.role == "assistant"

    def test_build_stream_chunk_thinking_delta_emits_thinking_blocks(self):
        from unittest.mock import MagicMock

        event = MagicMock()
        event.type = "content_block_delta"
        event.index = 0
        event.delta.type = "thinking_delta"
        event.delta.thinking = "step 1"
        chunk = self.mod._build_stream_chunk(event, "claude-sonnet-4", "c1")
        delta = chunk.choices[0].delta
        assert delta.reasoning_content == "step 1"
        assert delta.thinking_blocks == [{"type": "thinking", "thinking": "step 1"}]

    def test_build_stream_chunk_signature_delta(self):
        from unittest.mock import MagicMock

        event = MagicMock()
        event.type = "content_block_delta"
        event.index = 0
        event.delta.type = "signature_delta"
        event.delta.signature = "sig_final"
        chunk = self.mod._build_stream_chunk(event, "claude-sonnet-4", "c1")
        delta = chunk.choices[0].delta
        assert delta.thinking_blocks == [{"type": "thinking", "signature": "sig_final"}]

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
# cache_control passthrough
# ---------------------------------------------------------------------------

    def test_translate_content_text_preserves_cache_control(self):
        result = self.mod._translate_content(
            [{"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}}]
        )
        assert result == [{"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}}]

    def test_translate_content_text_no_cache_control_unchanged(self):
        result = self.mod._translate_content([{"type": "text", "text": "hello"}])
        assert result == [{"type": "text", "text": "hello"}]

    def test_translate_content_image_url_preserves_cache_control(self):
        result = self.mod._translate_content([{
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.png"},
            "cache_control": {"type": "ephemeral"},
        }])
        assert result[0]["type"] == "image"
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_translate_content_image_base64_preserves_cache_control(self):
        result = self.mod._translate_content([{
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abc123"},
            "cache_control": {"type": "ephemeral"},
        }])
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_translate_tools_preserves_cache_control(self):
        tools = [{
            "type": "function",
            "cache_control": {"type": "ephemeral"},
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        result = self.mod._translate_tools(tools)
        assert result[0]["cache_control"] == {"type": "ephemeral"}
        assert result[0]["name"] == "get_weather"

    def test_extract_system_list_preserves_cache_control(self):
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": "Hi"},
        ]
        system, conv = self.mod._extract_system(msgs)
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_build_request_kwargs_cache_control_end_to_end(self):
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]},
        ]
        tools = [{"type": "function", "cache_control": {"type": "ephemeral"},
                  "function": {"name": "f", "parameters": {}}}]
        req = self.mod._build_request_kwargs("claude-sonnet-4-20250514", msgs, False, None, None, tools=tools)
        assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert req["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert req["tools"][0]["cache_control"] == {"type": "ephemeral"}

    # -- Gap #5: reasoning_effort → thinking/output_config --

    def test_map_reasoning_effort_none(self):
        assert self.mod._map_reasoning_effort(None, "claude-3-haiku") == (None, None)

    def test_map_reasoning_effort_none_string(self):
        assert self.mod._map_reasoning_effort("none", "claude-3-haiku") == (None, None)

    def test_map_reasoning_effort_low_standard(self):
        t, o = self.mod._map_reasoning_effort("low", "claude-sonnet-4-20250514")
        assert t == {"type": "enabled", "budget_tokens": 1024}
        assert o is None

    def test_map_reasoning_effort_high_standard(self):
        t, o = self.mod._map_reasoning_effort("high", "claude-3-7-sonnet-20250219")
        assert t == {"type": "enabled", "budget_tokens": 4096}
        assert o is None

    def test_map_reasoning_effort_opus_4_6(self):
        t, o = self.mod._map_reasoning_effort("high", "claude-opus-4-6-20250514")
        assert t == {"type": "adaptive"}
        assert o == {"effort": "high"}

    def test_map_reasoning_effort_opus_4_6_underscore(self):
        t, o = self.mod._map_reasoning_effort("medium", "claude_opus_4_6")
        assert t == {"type": "adaptive"}
        assert o == {"effort": "medium"}

    def test_build_request_kwargs_reasoning_effort(self):
        msgs = [{"role": "user", "content": "hi"}]
        req = self.mod._build_request_kwargs("claude-sonnet-4-20250514", msgs, False, None, None, reasoning_effort="medium")
        assert req["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert "reasoning_effort" not in req

    def test_build_request_kwargs_explicit_thinking_overrides_reasoning_effort(self):
        msgs = [{"role": "user", "content": "hi"}]
        req = self.mod._build_request_kwargs("claude-sonnet-4-20250514", msgs, False, None, None,
            thinking={"type": "enabled", "budget_tokens": 10000}, reasoning_effort="low")
        assert req["thinking"]["budget_tokens"] == 10000

    # -- Gap #6: empty text block filtering --

    def test_extract_system_filters_empty_string(self):
        msgs = [{"role": "system", "content": ""}, {"role": "user", "content": "hi"}]
        system, _ = self.mod._extract_system(msgs)
        assert system is None

    def test_extract_system_filters_empty_text_block(self):
        msgs = [{"role": "system", "content": [{"type": "text", "text": ""}]}, {"role": "user", "content": "hi"}]
        system, _ = self.mod._extract_system(msgs)
        assert system is None

    def test_extract_system_filters_empty_string_in_list(self):
        msgs = [{"role": "system", "content": [""]}, {"role": "user", "content": "hi"}]
        system, _ = self.mod._extract_system(msgs)
        assert system is None

    def test_translate_content_filters_empty_text(self):
        result = self.mod._translate_content([
            {"type": "text", "text": ""},
            {"type": "text", "text": "hello"},
        ])
        assert result == [{"type": "text", "text": "hello"}]

    # -- Gap #7: JSON schema filtering --

    def test_filter_schema_strips_unsupported(self):
        schema = {"type": "object", "properties": {
            "count": {"type": "integer", "minimum": 1, "maximum": 100}
        }}
        result = self.mod._filter_schema(schema)
        assert "minimum" not in result["properties"]["count"]
        assert "maximum" not in result["properties"]["count"]
        assert "minimum: 1" in result["properties"]["count"]["description"]

    def test_filter_schema_preserves_supported(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        assert self.mod._filter_schema(schema) == schema

    def test_filter_schema_recursive_items(self):
        schema = {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 2}
        result = self.mod._filter_schema(schema)
        assert "minItems" not in result
        assert "minLength" not in result["items"]

    def test_filter_schema_anyof(self):
        schema = {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "string"}]}
        result = self.mod._filter_schema(schema)
        assert "minimum" not in result["anyOf"][0]

    def test_translate_tools_filters_schema(self):
        tools = [{"type": "function", "function": {
            "name": "f", "parameters": {"type": "object", "properties": {
                "n": {"type": "integer", "minimum": 0, "maximum": 10}
            }}
        }}]
        result = self.mod._translate_tools(tools)
        assert "minimum" not in result[0]["input_schema"]["properties"]["n"]


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
