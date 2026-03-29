"""Live integration tests — one per provider, verifying DSPy response contract."""

import json
import os

import pytest

import litelm


def assert_dspy_response_contract(response):
    """Every assertion here mirrors an actual access pattern in DSPy's lm.py / adapters/."""
    # choices iteration
    assert len(response.choices) >= 1
    c = response.choices[0]

    # message content (DSPy: c.message.content)
    assert isinstance(c.message.content, str)
    assert len(c.message.content) > 0

    # finish reason (DSPy: c.finish_reason)
    assert c.finish_reason in ("stop", "end_turn", "length")

    # usage (DSPy: response.usage.prompt_tokens, dict(response.usage))
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens > 0
    usage_dict = dict(response.usage)
    assert "prompt_tokens" in usage_dict
    assert "completion_tokens" in usage_dict

    # cache_hit (DSPy: getattr(response, "cache_hit", False))
    cache_hit = getattr(response, "cache_hit", False)
    assert cache_hit is False or cache_hit is True


MESSAGES = [{"role": "user", "content": "Say 'hello' and nothing else."}]


def _chunks_with_usage(chunks):
    """Return [(index, prompt_tokens, completion_tokens)] for chunks with non-None usage."""
    result = []
    for i, chunk in enumerate(chunks):
        raw = chunk._chunk if hasattr(chunk, "_chunk") else chunk
        if hasattr(raw, "usage") and raw.usage is not None:
            result.append((i, raw.usage.prompt_tokens, raw.usage.completion_tokens))
    return result


def assert_stream_contract(chunks):
    """Verify chunk-level invariants across all providers."""
    assert len(chunks) > 0, "stream yielded no chunks"

    for chunk in chunks:
        # DSPy: chunk.predict_id (assignable)
        chunk.predict_id = "test-id"
        assert chunk.predict_id == "test-id"

        # must have choices attr
        assert hasattr(chunk, "choices")

    # at least one chunk should have text content
    has_content = any(chunk.choices and chunk.choices[0].delta.content for chunk in chunks)
    assert has_content, "no chunk contained text content"


def assert_stream_reassembly(chunks, require_usage=False):
    """Collect chunks → stream_chunk_builder → verify DSPy response contract.

    Most providers don't send usage in stream without explicit opt-in,
    so usage assertion is optional unless require_usage=True.
    """
    response = litelm.stream_chunk_builder(chunks)
    assert len(response.choices) >= 1
    c = response.choices[0]
    assert isinstance(c.message.content, str)
    assert len(c.message.content) > 0
    assert c.finish_reason in ("stop", "end_turn", "length")
    # usage: always present, but may be zeros without include_usage
    assert hasattr(response, "usage")
    if require_usage:
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
    return response


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
def test_openai_basic_completion():
    response = litelm.completion("openai/gpt-4o-mini", messages=MESSAGES)
    print(response)
    assert_dspy_response_contract(response)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
def test_anthropic_basic_completion():
    response = litelm.completion("anthropic/claude-3-haiku-20240307", messages=MESSAGES)
    print(response)
    assert_dspy_response_contract(response)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="no GROQ_API_KEY")
def test_groq_basic_completion():
    response = litelm.completion("groq/llama-3.1-8b-instant", messages=MESSAGES)
    print(response)
    assert_dspy_response_contract(response)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("MISTRAL_API_KEY"), reason="no MISTRAL_API_KEY")
def test_mistral_basic_completion():
    response = litelm.completion("mistral/mistral-small-latest", messages=MESSAGES)
    print(response)
    assert_dspy_response_contract(response)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("XAI_API_KEY"), reason="no XAI_API_KEY")
def test_xai_basic_completion():
    response = litelm.completion("xai/grok-3-mini-fast", messages=MESSAGES)
    print(response)
    assert_dspy_response_contract(response)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="no OPENROUTER_API_KEY")
def test_openrouter_basic_completion():
    response = litelm.completion("openrouter/meta-llama/llama-3.1-8b-instruct", messages=MESSAGES)
    print(response)
    assert_dspy_response_contract(response)


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
def test_openai_streaming():
    chunks = list(litelm.completion("openai/gpt-4o-mini", messages=MESSAGES, stream=True))
    assert_stream_contract(chunks)
    assert_stream_reassembly(chunks)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
def test_openai_streaming_with_usage():
    chunks = list(
        litelm.completion(
            "openai/gpt-4o-mini",
            messages=MESSAGES,
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assert_stream_contract(chunks)
    response = assert_stream_reassembly(chunks, require_usage=True)
    # total_tokens consistency: accumulation must not double-count
    assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="no GROQ_API_KEY")
def test_groq_streaming_with_usage():
    chunks = list(
        litelm.completion(
            "groq/llama-3.1-8b-instant",
            messages=MESSAGES,
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assert_stream_contract(chunks)
    response = assert_stream_reassembly(chunks, require_usage=True)
    assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("MISTRAL_API_KEY"), reason="no MISTRAL_API_KEY")
def test_mistral_streaming_with_usage():
    chunks = list(
        litelm.completion(
            "mistral/mistral-small-latest",
            messages=MESSAGES,
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assert_stream_contract(chunks)
    response = assert_stream_reassembly(chunks, require_usage=True)
    assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("XAI_API_KEY"), reason="no XAI_API_KEY")
def test_xai_streaming_with_usage():
    chunks = list(
        litelm.completion(
            "xai/grok-3-mini-fast",
            messages=MESSAGES,
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assert_stream_contract(chunks)
    response = assert_stream_reassembly(chunks, require_usage=True)
    # xAI reasoning models: total_tokens includes reasoning tokens not in prompt+completion
    assert response.usage.total_tokens >= response.usage.prompt_tokens + response.usage.completion_tokens


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="no OPENROUTER_API_KEY")
def test_openrouter_streaming_with_usage():
    chunks = list(
        litelm.completion(
            "openrouter/meta-llama/llama-3.1-8b-instruct",
            messages=MESSAGES,
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assert_stream_contract(chunks)
    response = assert_stream_reassembly(chunks, require_usage=True)
    assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
def test_anthropic_streaming():
    chunks = list(litelm.completion("anthropic/claude-3-haiku-20240307", messages=MESSAGES, stream=True))
    assert_stream_contract(chunks)
    # Anthropic sends usage in stream natively (message_start + message_delta)
    assert_stream_reassembly(chunks, require_usage=True)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="no GROQ_API_KEY")
def test_groq_streaming():
    chunks = list(litelm.completion("groq/llama-3.1-8b-instant", messages=MESSAGES, stream=True))
    assert_stream_contract(chunks)
    assert_stream_reassembly(chunks)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("MISTRAL_API_KEY"), reason="no MISTRAL_API_KEY")
def test_mistral_streaming():
    chunks = list(litelm.completion("mistral/mistral-small-latest", messages=MESSAGES, stream=True))
    assert_stream_contract(chunks)
    assert_stream_reassembly(chunks)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("XAI_API_KEY"), reason="no XAI_API_KEY")
def test_xai_streaming():
    chunks = list(litelm.completion("xai/grok-3-mini-fast", messages=MESSAGES, stream=True))
    assert_stream_contract(chunks)
    assert_stream_reassembly(chunks)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="no OPENROUTER_API_KEY")
def test_openrouter_streaming():
    chunks = list(litelm.completion("openrouter/meta-llama/llama-3.1-8b-instruct", messages=MESSAGES, stream=True))
    assert_stream_contract(chunks)
    assert_stream_reassembly(chunks)


# ---------------------------------------------------------------------------
# Chunk-level usage verification
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
def test_openai_stream_chunks_no_usage_without_opt_in():
    """Without stream_options=include_usage, raw chunks carry no usage."""
    chunks = list(litelm.completion("openai/gpt-4o-mini", messages=MESSAGES, stream=True))
    usage_chunks = _chunks_with_usage(chunks)
    print(f"usage_chunks (should be empty): {usage_chunks}")
    assert usage_chunks == [], f"expected no usage in chunks without opt-in, got {usage_chunks}"


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
def test_anthropic_stream_chunks_split_usage():
    """Anthropic sends usage split across message_start (prompt) and message_delta (completion)."""
    chunks = list(litelm.completion("anthropic/claude-3-haiku-20240307", messages=MESSAGES, stream=True))
    usage_chunks = _chunks_with_usage(chunks)
    print(f"usage_chunks: {usage_chunks}")

    # At least 2 chunks carry usage (message_start + message_delta)
    assert len(usage_chunks) >= 2, f"expected >=2 usage chunks, got {len(usage_chunks)}: {usage_chunks}"

    # First usage chunk: prompt_tokens > 0, completion_tokens == 0 (message_start)
    _, first_prompt, first_completion = usage_chunks[0]
    assert first_prompt > 0, f"first usage chunk should have prompt_tokens > 0, got {first_prompt}"
    assert first_completion == 0, f"first usage chunk should have completion_tokens == 0, got {first_completion}"

    # Last usage chunk: prompt_tokens == 0, completion_tokens > 0 (message_delta)
    _, last_prompt, last_completion = usage_chunks[-1]
    assert last_prompt == 0, f"last usage chunk should have prompt_tokens == 0, got {last_prompt}"
    assert last_completion > 0, f"last usage chunk should have completion_tokens > 0, got {last_completion}"

    # Accumulation via stream_chunk_builder must combine correctly
    response = litelm.stream_chunk_builder(chunks)
    assert response.usage.prompt_tokens == first_prompt
    assert response.usage.completion_tokens == last_completion
    assert response.usage.total_tokens == first_prompt + last_completion


# ---------------------------------------------------------------------------
# Tool call tests
# ---------------------------------------------------------------------------

TOOL_MESSAGES = [{"role": "user", "content": "What is the weather in Paris?"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]


def assert_tool_call_contract(response):
    """Verify the exact shape DSPy reads from tool call responses."""
    c = response.choices[0]
    assert c.finish_reason == "tool_calls"
    assert c.message.tool_calls is not None
    assert len(c.message.tool_calls) >= 1

    tc = c.message.tool_calls[0]
    assert tc.type == "function"
    assert isinstance(tc.id, str) and len(tc.id) > 0
    assert isinstance(tc.function.name, str) and tc.function.name == "get_weather"
    assert isinstance(tc.function.arguments, str)
    args = json.loads(tc.function.arguments)
    assert isinstance(args, dict)
    assert "city" in args

    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0


def assert_stream_tool_call_contract(chunks):
    """Verify streaming tool calls reassemble correctly via stream_chunk_builder."""
    response = litelm.stream_chunk_builder(chunks)
    # Same as assert_tool_call_contract but skip usage (not sent without stream_options)
    c = response.choices[0]
    assert c.finish_reason == "tool_calls"
    assert c.message.tool_calls is not None
    assert len(c.message.tool_calls) >= 1
    tc = c.message.tool_calls[0]
    assert tc.type == "function"
    assert isinstance(tc.id, str) and len(tc.id) > 0
    assert tc.function.name == "get_weather"
    assert isinstance(tc.function.arguments, str)
    args = json.loads(tc.function.arguments)
    assert isinstance(args, dict) and "city" in args
    return response


_TOOL_KW = dict(messages=TOOL_MESSAGES, tools=TOOLS, tool_choice="required")

# --- Non-streaming tool calls ---


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
def test_openai_tool_call():
    response = litelm.completion("openai/gpt-4o-mini", **_TOOL_KW)
    print(response)
    assert_tool_call_contract(response)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
def test_anthropic_tool_call():
    response = litelm.completion("anthropic/claude-3-haiku-20240307", **_TOOL_KW)
    print(response)
    assert_tool_call_contract(response)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="no GROQ_API_KEY")
def test_groq_tool_call():
    response = litelm.completion("groq/llama-3.3-70b-versatile", **_TOOL_KW)
    print(response)
    assert_tool_call_contract(response)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("MISTRAL_API_KEY"), reason="no MISTRAL_API_KEY")
def test_mistral_tool_call():
    response = litelm.completion("mistral/mistral-small-latest", **_TOOL_KW)
    print(response)
    assert_tool_call_contract(response)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("XAI_API_KEY"), reason="no XAI_API_KEY")
def test_xai_tool_call():
    response = litelm.completion("xai/grok-3-mini-fast", **_TOOL_KW)
    print(response)
    assert_tool_call_contract(response)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="no OPENROUTER_API_KEY")
def test_openrouter_tool_call():
    response = litelm.completion("openrouter/openai/gpt-4o-mini", **_TOOL_KW)
    print(response)
    assert_tool_call_contract(response)


# --- Streaming tool calls ---


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
def test_openai_streaming_tool_call():
    chunks = list(litelm.completion("openai/gpt-4o-mini", **_TOOL_KW, stream=True))
    has_tc = any(c.choices and c.choices[0].delta.tool_calls for c in chunks if c.choices)
    assert has_tc, "no chunk contained tool_calls delta"
    assert_stream_tool_call_contract(chunks)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
def test_anthropic_streaming_tool_call():
    chunks = list(litelm.completion("anthropic/claude-3-haiku-20240307", **_TOOL_KW, stream=True))
    has_tc = any(c.choices and c.choices[0].delta.tool_calls for c in chunks if c.choices)
    assert has_tc, "no chunk contained tool_calls delta"
    assert_stream_tool_call_contract(chunks)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="no GROQ_API_KEY")
def test_groq_streaming_tool_call():
    chunks = list(litelm.completion("groq/llama-3.3-70b-versatile", **_TOOL_KW, stream=True))
    has_tc = any(c.choices and c.choices[0].delta.tool_calls for c in chunks if c.choices)
    assert has_tc, "no chunk contained tool_calls delta"
    assert_stream_tool_call_contract(chunks)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("MISTRAL_API_KEY"), reason="no MISTRAL_API_KEY")
def test_mistral_streaming_tool_call():
    chunks = list(litelm.completion("mistral/mistral-small-latest", **_TOOL_KW, stream=True))
    has_tc = any(c.choices and c.choices[0].delta.tool_calls for c in chunks if c.choices)
    assert has_tc, "no chunk contained tool_calls delta"
    assert_stream_tool_call_contract(chunks)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("XAI_API_KEY"), reason="no XAI_API_KEY")
def test_xai_streaming_tool_call():
    chunks = list(litelm.completion("xai/grok-3-mini-fast", **_TOOL_KW, stream=True))
    has_tc = any(c.choices and c.choices[0].delta.tool_calls for c in chunks if c.choices)
    assert has_tc, "no chunk contained tool_calls delta"
    assert_stream_tool_call_contract(chunks)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="no OPENROUTER_API_KEY")
def test_openrouter_streaming_tool_call():
    chunks = list(litelm.completion("openrouter/openai/gpt-4o-mini", **_TOOL_KW, stream=True))
    has_tc = any(c.choices and c.choices[0].delta.tool_calls for c in chunks if c.choices)
    assert has_tc, "no chunk contained tool_calls delta"
    assert_stream_tool_call_contract(chunks)


# --- Anthropic chunk-level tool call inspection ---


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
def test_anthropic_stream_tool_call_chunks():
    """Inspect Anthropic streaming tool call chunks: content_block_start has id+name, deltas have args."""
    chunks = list(litelm.completion("anthropic/claude-3-haiku-20240307", **_TOOL_KW, stream=True))

    tc_chunks = [(i, c) for i, c in enumerate(chunks) if c.choices and c.choices[0].delta.tool_calls]
    print(f"tool_call chunks: {len(tc_chunks)} at indices {[i for i, _ in tc_chunks]}")
    assert len(tc_chunks) >= 2, f"need at least start + delta chunks, got {len(tc_chunks)}"

    # First tool_call chunk: has id and name (content_block_start)
    first_tc = tc_chunks[0][1].choices[0].delta.tool_calls[0]
    assert first_tc.id is not None and len(first_tc.id) > 0
    assert first_tc.function.name == "get_weather"

    # Last tool_call chunk: has argument fragment
    last_tc = tc_chunks[-1][1].choices[0].delta.tool_calls[0]
    assert last_tc.function.arguments  # non-empty


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------


def assert_embedding_contract(response):
    """Verify the exact shape DSPy reads from embedding responses."""
    assert len(response.data) >= 1
    emb = response.data[0]["embedding"]
    assert isinstance(emb, list)
    assert len(emb) > 0
    assert all(isinstance(x, float) for x in emb)
    assert response.usage.prompt_tokens > 0
    assert response.usage.total_tokens > 0


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
def test_openai_embedding():
    response = litelm.embedding("openai/text-embedding-3-small", input="hello world")
    print(response)
    assert_embedding_contract(response)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("MISTRAL_API_KEY"), reason="no MISTRAL_API_KEY")
def test_mistral_embedding():
    response = litelm.embedding("mistral/mistral-embed", input="hello world")
    print(response)
    assert_embedding_contract(response)


# ---------------------------------------------------------------------------
# ContextWindowExceededError tests
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
def test_openai_context_window_error():
    """Exceed gpt-4o-mini's 128k context → ContextWindowExceededError."""
    huge = "word " * 200000  # ~200k tokens
    with pytest.raises(litelm.ContextWindowExceededError):
        litelm.completion("openai/gpt-4o-mini", messages=[{"role": "user", "content": huge}])


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
def test_anthropic_context_window_error():
    """Oversized max_tokens triggers ContextWindowExceededError via keyword match on 'token'.

    Cannot trigger a true context-window error for Anthropic: haiku has 200k context but
    100k/min rate limit — rate limit fires first. Instead we send max_tokens=20000 (haiku
    max output is 4096). Anthropic rejects with "max_tokens: 20000 > 4096" which contains
    "token" → _map_error maps to ContextWindowExceededError. Semantically this is a
    max-output-tokens error, not context-window, but litellm uses the same keyword matching
    and DSPy only catches this exception to retry with shorter prompts.
    """
    try:
        litelm.completion(
            "anthropic/claude-3-haiku-20240307",
            messages=MESSAGES,
            max_tokens=20_000,
        )
        # If API accepted it (silently capped), test is inconclusive — skip
        pytest.skip("Anthropic accepted oversized max_tokens without error")
    except litelm.ContextWindowExceededError:
        pass  # Expected


# ---------------------------------------------------------------------------
# Text completion tests
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
def test_openai_text_completion():
    """Legacy completions API via text-completion-openai/ prefix."""
    response = litelm.text_completion(
        "text-completion-openai/gpt-3.5-turbo-instruct",
        prompt="The capital of France is",
        max_tokens=10,
    )
    print(response)
    assert len(response.choices) >= 1
    assert isinstance(response.choices[0].text, str)
    assert len(response.choices[0].text) > 0
    assert response.choices[0].finish_reason in ("stop", "length")
    assert response.usage.prompt_tokens > 0
    assert response.usage.total_tokens > 0


# --- Pydantic response_format ---

from pydantic import BaseModel


class StructuredAnswer(BaseModel):
    answer: str
    reasoning: str


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
def test_openai_pydantic_response_format():
    """Pydantic BaseModel as response_format — exercises _normalize_response_format."""
    response = litelm.completion(
        "openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 2+2? Explain briefly."}],
        response_format=StructuredAnswer,
    )
    assert_dspy_response_contract(response)
    content = response.choices[0].message.content
    parsed = json.loads(content)
    assert "answer" in parsed
    assert "reasoning" in parsed


# --- Azure OpenAI ---


def _azure_env():
    """Extract Azure base endpoint and api_version from AZURE_OPENAI_URL."""
    from urllib.parse import urlparse, parse_qs

    raw = os.environ.get("AZURE_OPENAI_URL", "")
    parsed = urlparse(raw)
    base = f"{parsed.scheme}://{parsed.netloc}"
    qs = parse_qs(parsed.query)
    version = qs.get("api-version", [None])[0]
    return base, version


@pytest.mark.live
@pytest.mark.skipif(
    not all(os.environ.get(k) for k in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_URL", "AZURE_OPENAI_MODEL")),
    reason="Azure credentials not set",
)
def test_azure_openai():
    """Basic Azure OpenAI completion."""
    base, version = _azure_env()
    kwargs = {"api_key": os.environ["AZURE_OPENAI_API_KEY"], "api_base": base}
    if version:
        kwargs["api_version"] = version
    resp = litelm.completion(
        model=f"azure/{os.environ['AZURE_OPENAI_MODEL']}",
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        **kwargs,
    )
    assert_dspy_response_contract(resp)


@pytest.mark.live
@pytest.mark.skipif(
    not all(os.environ.get(k) for k in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_URL", "AZURE_OPENAI_MODEL")),
    reason="Azure credentials not set",
)
def test_azure_openai_streaming():
    """Azure OpenAI streaming completion."""
    base, version = _azure_env()
    kwargs = {"api_key": os.environ["AZURE_OPENAI_API_KEY"], "api_base": base}
    if version:
        kwargs["api_version"] = version
    chunks = list(
        litelm.completion(
            model=f"azure/{os.environ['AZURE_OPENAI_MODEL']}",
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )
    )
    assert_stream_contract(chunks)
    response = litelm.stream_chunk_builder(chunks)
    assert_dspy_response_contract(response)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
def test_openai_pydantic_response_format_streaming():
    """Streaming + pydantic response_format — verify JSON reassembles."""
    chunks = list(
        litelm.completion(
            "openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 2+2? Explain briefly."}],
            response_format=StructuredAnswer,
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assert_stream_contract(chunks)
    response = litelm.stream_chunk_builder(chunks)
    content = response.choices[0].message.content
    parsed = json.loads(content)
    assert "answer" in parsed
    assert "reasoning" in parsed
