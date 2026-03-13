# litelm

[litellm](https://github.com/BerriAI/litellm) is bloated. 86 MB of dependencies for what is mostly a router.

litelm strips it to core routing + formatting. ~2,000 lines. Two deps (`openai`, `httpx`).

## Strategy

litellm has become infrastructure — dozens of frameworks depend on it. We validate litelm as a drop-in replacement one consumer at a time, starting with the most popular.

**Current target: [DSPy](https://github.com/stanfordnlp/dspy)** — full drop-in verified. All 7 execution paths proven live (Predict, CoT, typed signatures, streaming, embeddings, tool use, multi-output). 6 providers tested.

**Next targets:** [LangChain](https://github.com/langchain-ai/langchain), [CrewAI](https://github.com/crewAIInc/crewAI), [AutoGen](https://github.com/microsoft/autogen) — litellm's most depended-on consumers. Each one we pass is proof the API surface is correct.

The litellm test suite itself runs against litelm unmodified via `sys.modules` shimming — 206 tests passing and climbing.

---

```python
import litelm

response = litelm.completion("openai/gpt-4o", messages=[{"role": "user", "content": "Hello!"}])
response = litelm.completion("groq/llama-3.1-70b-versatile", messages=[...], stream=True)
response = litelm.embedding("openai/text-embedding-3-small", input=["hello world"])
```

Every function has an async variant: `acompletion`, `aembedding`, `aresponses`, `atext_completion`.

### Drop-in usage (DSPy example)

```python
import sys, litelm
sys.modules["litellm"] = litelm

import dspy  # now uses litelm for all LLM calls

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)
dspy.Predict("question -> answer")(question="What is 2+2?")
```

The `sys.modules` swap must happen **before** the consumer import.

---

## Install

```bash
pip install litelm                # openai + httpx
pip install litelm[anthropic]     # + anthropic SDK
pip install litelm[bedrock]       # + boto3
pip install litelm[all]           # everything
```

## Providers

`"provider/model-name"` → routes to the right base URL and API key.

OpenAI, Azure, Groq, Together, Fireworks, Mistral, DeepSeek, OpenRouter, Anthropic, Perplexity, xAI, Gemini, Cohere, Cloudflare, Bedrock, Ollama — and any OpenAI-compatible endpoint via `api_base`.

Custom handlers (non-OpenAI-compat): Anthropic, Bedrock, Cloudflare, Mistral.

## What's Implemented

| Function | Status |
|----------|--------|
| `completion` / `acompletion` | Implemented |
| `embedding` / `aembedding` | Implemented |
| `responses` / `aresponses` | Implemented |
| `text_completion` / `atext_completion` | Implemented |
| `stream_chunk_builder` | Implemented |
| `mock_response` | Implemented |
| `supports_function_calling` | Stub (always True) |
| `supports_response_schema` | Stub (always True) |
| `supports_reasoning` | Implemented (pattern match) |
| `get_supported_openai_params` | Implemented |

## What's Not Implemented (by design)

Proxy, router, caching, budgeting, agents, guardrails, image gen, audio, OCR, fine-tuning, batches, assistants, scheduler, token counting, cost tracking.

## Tests

```bash
uv run pytest tests/ -x --ignore=tests/ported        # 92 unit tests
uv run pytest tests/test_live.py -m live --timeout=30 # 37 live provider tests
uv run pytest tests/test_dspy_smoke.py -m live --timeout=60  # 10 DSPy integration tests
```

Live tests require API keys in `.env.test`. Skipped by default; run with `-m live`.
