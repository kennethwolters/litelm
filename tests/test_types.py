"""Tests for ModelResponse, ModelResponseStream, and compatibility constructors."""

from litelm._types import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    Choices,
    Delta,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
    _to_dict,
)


def test_model_response_from_kwargs():
    r = ModelResponse(
        choices=[{"message": {"content": "Hello!", "role": "assistant"}}],
        model="gpt-4o",
    )
    assert r.choices[0].message.content == "Hello!"
    assert r.model == "gpt-4o"


def test_model_response_dict_access():
    r = ModelResponse(
        choices=[{"message": {"content": "Hi"}}],
    )
    assert r["choices"][0].message.content == "Hi"


def test_model_response_json():
    r = ModelResponse(choices=[{"message": {"content": "x"}}])
    j = r.json()
    assert "x" in j


def test_model_response_cache_hit():
    r = ModelResponse(choices=[], cache_hit=True)
    assert r.cache_hit is True


def test_model_response_stream_from_kwargs():
    s = ModelResponseStream(choices=[])
    assert s.choices == []
    assert s.model == "mock"


def test_message_defaults():
    m = Message(content="Hi!")
    assert m.role == "assistant"
    assert m.content == "Hi!"


def test_choices_defaults():
    c = Choices(message=Message(content="Hi!"))
    assert c.finish_reason == "stop"
    assert c.index == 0


def test_usage_defaults():
    u = Usage()
    assert u.prompt_tokens == 0
    assert u.completion_tokens == 0
    assert u.total_tokens == 0


def test_usage_custom():
    u = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    assert u.total_tokens == 30


def test_streaming_choices_defaults():
    sc = StreamingChoices(delta=Delta())
    assert sc.index == 0


def test_chat_completion_message_has_slots():
    assert hasattr(ChatCompletionMessage, "__slots__")
    msg = ChatCompletionMessage(content="hi", reasoning_content="think")
    assert msg.reasoning_content == "think"


def test_chat_completion_message_reasoning_defaults_none():
    msg = ChatCompletionMessage(content="hi")
    assert msg.reasoning_content is None


def test_choice_delta_has_slots():
    assert hasattr(ChoiceDelta, "__slots__")
    d = ChoiceDelta(content="hi", reasoning_content="think")
    assert d.reasoning_content == "think"


def test_choice_delta_to_dict():
    d = ChoiceDelta(role="assistant", content="hi")
    result = _to_dict(d)
    assert result["role"] == "assistant"
    assert result["content"] == "hi"
    assert result["reasoning_content"] is None


def test_model_response_model_dump():
    r = ModelResponse(
        choices=[{"message": {"content": "Hello!", "role": "assistant"}}],
        model="gpt-4o",
    )
    d = r.model_dump()
    assert isinstance(d, dict)
    assert d["model"] == "gpt-4o"
    assert d["choices"][0]["message"]["content"] == "Hello!"


def test_chat_completion_model_dump():
    from litelm._types import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage

    c = ChatCompletion(
        id="test",
        model="gpt-4o",
        choices=[Choice(index=0, message=ChatCompletionMessage(content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    d = c.model_dump()
    assert isinstance(d, dict)
    assert d["id"] == "test"
    assert d["usage"]["total_tokens"] == 3


def test_model_response_coerces_dict_choices():
    r = ModelResponse(
        choices=[
            {"message": {"content": "a", "role": "assistant"}, "finish_reason": "stop", "index": 0},
            {"message": {"content": "b"}, "index": 1},
        ]
    )
    assert r.choices[0].message.content == "a"
    assert r.choices[1].message.content == "b"
    assert r.choices[1].finish_reason == "stop"  # default


def test_tool_call_auto_generates_id():
    tc = ChatCompletionMessageToolCall(function={"name": "f", "arguments": "{}"})
    assert tc.id.startswith("call_")
    assert len(tc.id) == 29  # "call_" (5) + 24 hex chars


def test_tool_call_preserves_explicit_id():
    tc = ChatCompletionMessageToolCall(id="custom_123", function={"name": "f", "arguments": "{}"})
    assert tc.id == "custom_123"


def test_delta_tool_call_id_defaults_to_none():
    """Delta tool calls are partial — id stays None until set by a stream chunk."""
    dtc = ChoiceDeltaToolCall()
    assert dtc.id is None


# ---------------------------------------------------------------------------
# thinking_blocks on ChatCompletionMessage / ChoiceDelta
# ---------------------------------------------------------------------------


def test_chat_completion_message_thinking_blocks():
    blocks = [{"type": "thinking", "thinking": "Let me think...", "signature": "sig123"}]
    msg = ChatCompletionMessage(content="hi", thinking_blocks=blocks)
    assert msg.thinking_blocks == blocks


def test_chat_completion_message_thinking_blocks_defaults_none():
    msg = ChatCompletionMessage(content="hi")
    assert msg.thinking_blocks is None


def test_choice_delta_thinking_blocks():
    blocks = [{"type": "thinking", "thinking": "step 1", "signature": "s1"}]
    d = ChoiceDelta(content="hi", thinking_blocks=blocks)
    assert d.thinking_blocks == blocks


def test_choice_delta_thinking_blocks_defaults_none():
    d = ChoiceDelta(content="hi")
    assert d.thinking_blocks is None


# ---------------------------------------------------------------------------
# completion_tokens_details / prompt_tokens_details on CompletionUsage
# ---------------------------------------------------------------------------


def test_usage_completion_tokens_details():
    from litelm._types import CompletionUsage

    u = CompletionUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        completion_tokens_details={"reasoning_tokens": 15},
    )
    assert u.completion_tokens_details == {"reasoning_tokens": 15}


def test_usage_prompt_tokens_details():
    from litelm._types import CompletionUsage

    u = CompletionUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_tokens_details={"cached_tokens": 5},
    )
    assert u.prompt_tokens_details == {"cached_tokens": 5}


def test_usage_token_details_default_none():
    from litelm._types import CompletionUsage

    u = CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    assert u.completion_tokens_details is None
    assert u.prompt_tokens_details is None


def test_usage_dict_includes_token_details():
    from litelm._types import CompletionUsage

    u = CompletionUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        completion_tokens_details={"reasoning_tokens": 15},
    )
    d = dict(u)
    assert d["completion_tokens_details"] == {"reasoning_tokens": 15}
    assert d["prompt_tokens_details"] is None


# --- provider_specific_fields ---


def test_message_provider_specific_fields_default_none():
    msg = ChatCompletionMessage(role="assistant", content="hi")
    assert msg.provider_specific_fields is None


def test_message_provider_specific_fields_set():
    citations = [{"type": "char_location", "cited_text": "hello"}]
    msg = ChatCompletionMessage(
        role="assistant",
        content="hi",
        provider_specific_fields={"citations": citations},
    )
    assert msg.provider_specific_fields["citations"] == citations


def test_delta_provider_specific_fields_default_none():
    delta = ChoiceDelta(content="hi")
    assert delta.provider_specific_fields is None


def test_delta_provider_specific_fields_set():
    delta = ChoiceDelta(
        content="hi",
        provider_specific_fields={"citations": [{"type": "char_location"}]},
    )
    assert delta.provider_specific_fields["citations"][0]["type"] == "char_location"


def test_message_provider_specific_fields_in_dict():
    from litelm._types import _to_dict

    msg = ChatCompletionMessage(
        role="assistant",
        content="hi",
        provider_specific_fields={"citations": [{"type": "char_location"}]},
    )
    d = _to_dict(msg)
    assert d["provider_specific_fields"]["citations"][0]["type"] == "char_location"
