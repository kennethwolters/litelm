# litelm

[![PyPI](https://img.shields.io/pypi/v/litelm)](https://pypi.org/project/litelm/)
[![Python](https://img.shields.io/pypi/pyversions/litelm)](https://pypi.org/project/litelm/)
[![Tests](https://github.com/kennethwolters/litelm/actions/workflows/test.yml/badge.svg)](https://github.com/kennethwolters/litelm/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

litellm's routing + translation in ~2,300 lines and 2 dependencies (`openai`, `httpx`).

litellm routes LLM calls across providers and translates between message formats. That core is buried under 100k+ LOC of proxy servers, caching layers, cost tracking, and dozens of features most users never touch. litelm extracts just the call path — model routing, message translation, streaming, tool use, embeddings — and nothing else. No Router class, no proxy, no caching.

## Install

```bash
pip install litelm                # openai + httpx
pip install litelm[anthropic]     # + anthropic SDK
pip install litelm[bedrock]       # + boto3
pip install litelm[all]           # everything
```

## Usage

```python
import litelm

# Basic completion
response = litelm.completion("openai/gpt-4o", messages=[{"role": "user", "content": "Hello!"}])
print(response.choices[0].message.content)

# Streaming
for chunk in litelm.completion("groq/llama-3.1-70b-versatile", messages=[...], stream=True):
    print(chunk.choices[0].delta.content or "", end="")

# Embeddings
response = litelm.embedding("openai/text-embedding-3-small", input=["hello world"])
```

Every function has an async variant: `acompletion`, `aembedding`, `aresponses`, `atext_completion`.

The API mirrors litellm — same function names, same arguments, same response types. If you're using litellm today, switching is `s/litellm/litelm/` in your imports.

## What's in / what's out

| | litellm | litelm |
|---|---|---|
| Model routing (`provider/model` → right endpoint) | ✓ | ✓ |
| Message translation (Anthropic, Bedrock, Cloudflare, Mistral) | ✓ | ✓ |
| Streaming + `stream_chunk_builder` | ✓ | ✓ |
| Tool use (function calling) | ✓ | ✓ |
| Embeddings | ✓ | ✓ |
| Text completions | ✓ | ✓ |
| OpenAI Responses API | ✓ | ✓ |
| Mock responses | ✓ | ✓ |
| Router (load balancing, fallbacks) | ✓ | ✗ |
| Proxy server | ✓ | ✗ |
| Caching / budgeting / cost tracking | ✓ | ✗ |
| Token counting | ✓ | ✗ |
| Image gen, audio, OCR, fine-tuning | ✓ | ✗ |
| Agents, guardrails, scheduler | ✓ | ✗ |

## Providers

Routes to 19 providers via `"provider/model-name"` syntax. Any OpenAI-compatible endpoint works via `api_base`.

| Provider | Env Var | Handler | Verified |
|----------|---------|---------|:--------:|
| OpenAI | `OPENAI_API_KEY` | OpenAI SDK | Yes |
| Anthropic | `ANTHROPIC_API_KEY` | Custom | Yes |
| Groq | `GROQ_API_KEY` | OpenAI-compat | Yes |
| Mistral | `MISTRAL_API_KEY` | Custom | Yes |
| xAI | `XAI_API_KEY` | OpenAI-compat | Yes |
| OpenRouter | `OPENROUTER_API_KEY` | OpenAI-compat | Yes |
| Azure | `AZURE_API_KEY` | OpenAI SDK (Azure) | No |
| Bedrock | `AWS_ACCESS_KEY_ID` | Custom | No |
| Cloudflare | `CLOUDFLARE_API_TOKEN` | Custom | No |
| Together | `TOGETHERAI_API_KEY` | OpenAI-compat | No |
| Fireworks | `FIREWORKS_API_KEY` | OpenAI-compat | No |
| DeepSeek | `DEEPSEEK_API_KEY` | OpenAI-compat | No |
| Perplexity | `PERPLEXITYAI_API_KEY` | OpenAI-compat | No |
| DeepInfra | `DEEPINFRA_API_TOKEN` | OpenAI-compat | No |
| Gemini | `GEMINI_API_KEY` | OpenAI-compat | No |
| Cohere | `COHERE_API_KEY` | OpenAI-compat | No |
| Ollama | — | OpenAI-compat | No |
| vLLM | — | OpenAI-compat | No |
| LM Studio | — | OpenAI-compat | No |

## API Keys

Set the environment variable for your provider:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

Or pass directly:

```python
litelm.completion("openai/gpt-4o", messages=[...], api_key="sk-...")
litelm.completion("openai/gpt-4o", messages=[...], api_base="http://localhost:8000/v1")
```

## Error Handling

All provider errors are mapped to litelm's exception hierarchy:

```python
from litelm import ContextWindowExceededError, RateLimitError, AuthenticationError

try:
    response = litelm.completion("openai/gpt-4o", messages=messages)
except ContextWindowExceededError:
    # prompt too long — truncate and retry
    pass
except RateLimitError:
    # back off
    pass
except AuthenticationError:
    # bad API key
    pass
```

## Tool Calling

```python
tools = [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}}]

response = litelm.completion(
    "openai/gpt-4o", messages=[{"role": "user", "content": "Weather in Paris?"}],
    tools=tools, tool_choice="required",
)
tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name, tool_call.function.arguments)
```

## Custom / Local Providers

Any OpenAI-compatible server works via `api_base`:

```python
# vLLM
litelm.completion("openai/my-model", messages=[...], api_base="http://localhost:8000/v1")

# Ollama
litelm.completion("ollama/llama3", messages=[...], api_base="http://localhost:11434/v1")

# LM Studio
litelm.completion("openai/local-model", messages=[...], api_base="http://localhost:1234/v1")
```

## Status

**Alpha.** 129 own tests, 56 ported litellm tests passing unmodified via `sys.modules` shimming.

[DSPy](https://github.com/stanfordnlp/dspy) drop-in verified — all 7 execution paths proven live (Predict, CoT, typed signatures, streaming, embeddings, tool use, multi-output).

## Tests

```bash
uv run pytest tests/ -x --ignore=tests/ported        # 129 unit tests
uv run pytest tests/test_live.py -m live --timeout=30 # 37 live provider tests
uv run pytest tests/test_dspy_smoke.py -m live --timeout=60  # 10 DSPy integration tests
```

Live tests require API keys in `.env.test`. Skipped by default; run with `-m live`.
