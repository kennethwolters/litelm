"""Tests for ModelResponse, ModelResponseStream, and compatibility constructors."""

from litelm._types import (
    ChatCompletionMessage,
    ChoiceDelta,
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
