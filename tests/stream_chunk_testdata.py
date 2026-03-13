"""Test data ported 1:1 from litellm/tests/local_testing/stream_chunk_testdata.py

Streaming chunks from Claude 3.5 Sonnet: text content + tool call (sql_query).
"""

from litelm._types import (
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChunkChoice,
    ChatCompletionChunk,
)
from litelm import ModelResponseStream


def _chunk(content="", role=None, tool_calls=None, finish_reason=None, id="chatcmpl-634a6ad3-483a-44a1-8cdd-3befbeb4ac2f", created=1722656356):
    delta = ChoiceDelta(content=content, role=role, tool_calls=tool_calls)
    choice = ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)
    raw = ChatCompletionChunk(id=id, choices=[choice], created=created, model="claude-3-5-sonnet-20240620", object="chat.completion.chunk")
    return ModelResponseStream(raw)


def _tc(index, id=None, name=None, arguments=""):
    return ChoiceDeltaToolCall(
        index=index,
        id=id,
        function=ChoiceDeltaToolCallFunction(arguments=arguments, name=name),
        type="function",
    )


chunks = [
    _chunk(content="To answer", role="assistant"),
    _chunk(content=" your"),
    _chunk(content=" question about"),
    _chunk(content=" how"),
    _chunk(content=" many rows are in the "),
    _chunk(content="'users' table, I"),
    _chunk(content="'ll"),
    _chunk(content=" need to"),
    _chunk(content=" run"),
    _chunk(content=" a SQL query."),
    _chunk(content=" Let"),
    _chunk(content=" me"),
    _chunk(content=" "),
    _chunk(content="do that for"),
    _chunk(content=" you."),
    # Tool call start
    _chunk(content="", tool_calls=[_tc(index=1, id="toolu_01H3AjkLpRtGQrof13CBnWfK", name="sql_query")]),
    _chunk(content="", tool_calls=[_tc(index=1)]),
    _chunk(content="", tool_calls=[_tc(index=1, arguments='{"')], created=1722656357),
    _chunk(content="", tool_calls=[_tc(index=1, arguments='query": ')], created=1722656357),
    _chunk(content="", tool_calls=[_tc(index=1, arguments='"SELECT C')], created=1722656357),
    _chunk(content="", tool_calls=[_tc(index=1, arguments="OUNT(*")], created=1722656357),
    _chunk(content="", tool_calls=[_tc(index=1, arguments=") ")], created=1722656357),
    _chunk(content="", tool_calls=[_tc(index=1, arguments="FROM use")], created=1722656357),
    _chunk(content="", tool_calls=[_tc(index=1, arguments='rs;"}')], created=1722656357),
    # Final chunk
    _chunk(content=None, finish_reason="tool_calls", created=1722656357),
]
