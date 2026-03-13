# litelm

litellm's routing + translation in ~2,200 lines and 2 dependencies (`openai`, `httpx`).

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

## Status

**Alpha.** Proven against litellm's own test suite — 206 ported tests passing unmodified via `sys.modules` shimming. 92 own tests.

[DSPy](https://github.com/stanfordnlp/dspy) drop-in verified — all 7 execution paths proven live (Predict, CoT, typed signatures, streaming, embeddings, tool use, multi-output).

## Roadmap

Validating as a drop-in replacement one litellm consumer at a time — DSPy is done. More consumers, providers, and end-to-end verifications will follow.

## Tests

```bash
uv run pytest tests/ -x --ignore=tests/ported        # 92 unit tests
uv run pytest tests/test_live.py -m live --timeout=30 # 37 live provider tests
uv run pytest tests/test_dspy_smoke.py -m live --timeout=60  # 10 DSPy integration tests
```

Live tests require API keys in `.env.test`. Skipped by default; run with `-m live`.
