"""DSPy integration smoke tests — proves litelm can replace litellm across all 7 DSPy execution paths."""

import asyncio
import sys

import litelm

# Swap before DSPy import
sys.modules["litellm"] = litelm

import dspy  # noqa: E402
import pytest  # noqa: E402
from pydantic import BaseModel  # noqa: E402

# ---------------------------------------------------------------------------
# Path 0: Basic completion (Predict + CoT)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_dspy_predict():
    """Basic DSPy Predict with gpt-4o-mini through litelm."""
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    predict = dspy.Predict("question -> answer")
    result = predict(question="What is 2+2?")
    assert result.answer is not None
    assert len(result.answer) > 0


@pytest.mark.live
def test_dspy_cot():
    """DSPy ChainOfThought — exercises streaming + response parsing."""
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    cot = dspy.ChainOfThought("question -> answer")
    result = cot(question="What is the capital of France?")
    assert "Paris" in result.answer


# ---------------------------------------------------------------------------
# Path 1: Typed signatures / JSON adapter
# ---------------------------------------------------------------------------


class QAOutput(BaseModel):
    answer: str
    confidence: float


class TypedQA(dspy.Signature):
    """Answer with structured output."""

    question: str = dspy.InputField()
    response: QAOutput = dspy.OutputField()


@pytest.mark.live
def test_dspy_typed_signature():
    """Typed Pydantic output — exercises get_supported_openai_params, supports_response_schema, response_format."""
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    predict = dspy.Predict(TypedQA)
    result = predict(question="What is 2+2?")
    assert isinstance(result.response, QAOutput)
    assert result.response.answer is not None
    assert isinstance(result.response.confidence, float)


# ---------------------------------------------------------------------------
# Path 2: Streaming via dspy.streamify
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_dspy_streaming():
    """dspy.streamify — exercises acompletion(stream=True), isinstance(ModelResponseStream), stream_chunk_builder."""
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    predict = dspy.Predict("question -> answer")
    streaming_predict = dspy.streamify(predict)

    async def run():
        chunks = []
        async for chunk in streaming_predict(question="What is 2+2?"):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run())
    assert len(chunks) > 0
    # Last chunk should be the final Prediction
    final = chunks[-1]
    assert hasattr(final, "answer")
    assert final.answer is not None


# ---------------------------------------------------------------------------
# Path 3: Embeddings via dspy.Embedder
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_dspy_embeddings():
    """dspy.Embedder — exercises embedding(), data[i]['embedding'] access."""
    embedder = dspy.Embedder("openai/text-embedding-3-small")

    result = embedder(["hello world", "test"])
    assert len(result) == 2
    assert len(result[0]) > 0  # non-empty embedding vector


# ---------------------------------------------------------------------------
# Path 4: Tool use / ReAct
# ---------------------------------------------------------------------------


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny, 72°F in {city}"


@pytest.mark.live
def test_dspy_tool_use():
    """ReAct with tools — exercises supports_function_calling, completion(tools=...), tool call parsing."""
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    react = dspy.ReAct("question -> answer", tools=[get_weather], max_iters=3)
    result = react(question="What is the weather in Paris?")
    assert result.answer is not None
    assert len(result.answer) > 0


# ---------------------------------------------------------------------------
# Path 5: Multi-output (n>1)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_dspy_multi_output():
    """Predict with n=3 — exercises completion(n=3), multi-choice response parsing."""
    lm = dspy.LM("openai/gpt-4o-mini", n=3, temperature=1.0)
    dspy.configure(lm=lm)

    predict = dspy.Predict("question -> answer")
    result = predict(question="Name a color")
    # DSPy should have access to completions
    assert result.answer is not None
    assert len(result.completions.answer) == 3


# ---------------------------------------------------------------------------
# Path 6: Multi-provider (Anthropic, Groq)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_dspy_anthropic():
    """Anthropic through DSPy — exercises native handler dispatch."""
    lm = dspy.LM("anthropic/claude-3-haiku-20240307", max_tokens=100)
    dspy.configure(lm=lm)

    predict = dspy.Predict("question -> answer")
    result = predict(question="What is 2+2?")
    assert result.answer is not None


@pytest.mark.live
def test_dspy_groq():
    """Groq through DSPy — exercises OpenAI-compat provider dispatch."""
    lm = dspy.LM("groq/llama-3.3-70b-versatile", max_tokens=100)
    dspy.configure(lm=lm)

    predict = dspy.Predict("question -> answer")
    result = predict(question="What is 2+2?")
    assert result.answer is not None


# ---------------------------------------------------------------------------
# Path 7: Error recovery — ContextWindowExceededError
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_dspy_context_window_error():
    """ContextWindowExceededError propagates through DSPy's adapter retry logic."""
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    # Trigger context window error with massive input
    predict = dspy.Predict("text -> summary")
    huge_text = "word " * 200_000  # ~200k tokens, exceeds 128k context

    with pytest.raises(Exception) as exc_info:
        predict(text=huge_text)
    # DSPy should surface ContextWindowExceededError (or re-raise it)
    err_str = str(type(exc_info.value).__name__) + str(exc_info.value)
    assert any(kw in err_str.lower() for kw in ("context", "token", "length", "too long"))
