"""Stream chunk builder tests — ported 1:1 from litellm/tests/local_testing/test_stream_chunk_builder.py

Tests that litelm.stream_chunk_builder produces identical results to litellm.stream_chunk_builder
given the same input chunks.
"""

from litelm import ModelResponseStream, stream_chunk_builder
from tests import stream_chunk_testdata


def test_stream_chunk_builder_litellm_mixed_calls():
    """Ported from litellm: text content + tool call reassembly from Claude 3.5 Sonnet chunks."""
    response = stream_chunk_builder(stream_chunk_testdata.chunks)
    expected = (
        "To answer your question about how many rows are in the 'users' table,"
        " I'll need to run a SQL query. Let me do that for you."
    )
    assert response.choices[0].message.content == expected

    assert len(response.choices[0].message.tool_calls) == 1
    tc = response.choices[0].message.tool_calls[0]
    assert tc.function.name == "sql_query"
    assert tc.function.arguments == '{"query": "SELECT COUNT(*) FROM users;"}'
    assert tc.id == "toolu_01H3AjkLpRtGQrof13CBnWfK"
    assert tc.type == "function"


def test_stream_chunk_builder_litellm_empty_chunks():
    """Ported from litellm: empty chunk list returns empty ModelResponse."""
    response = stream_chunk_builder(chunks=[])
    assert response.choices == []


def test_stream_chunk_builder_multiple_tool_calls():
    """Ported from litellm: multiple parallel tool calls (exponentiate + add) from GPT-4o chunks."""
    init_chunks = [
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_X9P9B6STj7ze8OsJCGkfoN94",
                                "function": {"arguments": "", "name": "exponentiate"},
                                "type": "function",
                                "index": 0,
                            }
                        ],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{"function": {"arguments": '{"ba'}, "type": "function", "index": 0}],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{"function": {"arguments": 'se": '}, "type": "function", "index": 0}],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{"function": {"arguments": '3, "ex'}, "type": "function", "index": 0}],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{"function": {"arguments": "pone"}, "type": "function", "index": 0}],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{"function": {"arguments": 'nt": '}, "type": "function", "index": 0}],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{"function": {"arguments": "5}"}, "type": "function", "index": 0}],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
        },
        # Second tool call starts
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_Qq8yDeRx7v276abRcLrYORdW",
                                "function": {"arguments": "", "name": "add"},
                                "type": "function",
                                "index": 1,
                            }
                        ],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{"function": {"arguments": '{"fi'}, "type": "function", "index": 1}],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{"function": {"arguments": "rst_i"}, "type": "function", "index": 1}],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{"function": {"arguments": 'nt": 1'}, "type": "function", "index": 1}],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{"function": {"arguments": '2, "'}, "type": "function", "index": 1}],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{"function": {"arguments": "secon"}, "type": "function", "index": 1}],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{"function": {"arguments": 'd_int"'}, "type": "function", "index": 1}],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{"function": {"arguments": ": 3}"}, "type": "function", "index": 1}],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
        },
        # Final chunk
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [{"finish_reason": "tool_calls", "index": 0, "delta": {}}],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
        },
    ]

    chunks = [ModelResponseStream(**chunk) for chunk in init_chunks]
    response = stream_chunk_builder(chunks=chunks)

    # Verify two tool calls were assembled correctly
    assert len(response.choices[0].message.tool_calls) == 2

    tc0 = response.choices[0].message.tool_calls[0]
    assert tc0.function.name == "exponentiate"
    assert tc0.function.arguments == '{"base": 3, "exponent": 5}'
    assert tc0.id == "call_X9P9B6STj7ze8OsJCGkfoN94"

    tc1 = response.choices[0].message.tool_calls[1]
    assert tc1.function.name == "add"
    assert tc1.function.arguments == '{"first_int": 12, "second_int": 3}'
    assert tc1.id == "call_Qq8yDeRx7v276abRcLrYORdW"

    assert response.choices[0].finish_reason == "tool_calls"


def test_stream_chunk_builder_thinking_blocks():
    chunks = [
        ModelResponseStream(id="c1", model="m", choices=[{"delta": {"role": "assistant", "thinking_blocks": [{"type": "thinking", "thinking": "step 1"}]}}]),
        ModelResponseStream(id="c1", model="m", choices=[{"delta": {"thinking_blocks": [{"type": "thinking", "thinking": "step 2"}]}}]),
        ModelResponseStream(id="c1", model="m", choices=[{"delta": {"thinking_blocks": [{"type": "thinking", "signature": "sig_abc"}]}}]),
        ModelResponseStream(id="c1", model="m", choices=[{"delta": {"content": "Answer."}}]),
        ModelResponseStream(id="c1", model="m", choices=[{"finish_reason": "stop", "delta": {}}]),
    ]
    response = stream_chunk_builder(chunks)
    msg = response.choices[0].message
    assert msg.content == "Answer."
    assert msg.thinking_blocks is not None
    assert len(msg.thinking_blocks) == 1
    assert msg.thinking_blocks[0]["type"] == "thinking"
    assert msg.thinking_blocks[0]["thinking"] == "step 1step 2"
    assert msg.thinking_blocks[0]["signature"] == "sig_abc"


def test_stream_chunk_builder_token_details():
    from litelm._types import CompletionUsage

    chunks = [
        ModelResponseStream(id="c1", model="m", choices=[{"delta": {"role": "assistant", "content": "hi"}}]),
        ModelResponseStream(
            id="c1", model="m",
            choices=[{"finish_reason": "stop", "delta": {}}],
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=20, total_tokens=30,
                completion_tokens_details={"reasoning_tokens": 5},
                prompt_tokens_details={"cached_tokens": 3},
            ),
        ),
    ]
    response = stream_chunk_builder(chunks)
    assert response.usage.completion_tokens_details == {"reasoning_tokens": 5}
    assert response.usage.prompt_tokens_details == {"cached_tokens": 3}
