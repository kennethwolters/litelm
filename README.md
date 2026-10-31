# litelm

[![PyPI](https://img.shields.io/pypi/v/litelm)](https://pypi.org/project/litelm/)
[![Python](https://img.shields.io/pypi/pyversions/litelm)](https://pypi.org/project/litelm/)
[![Tests](https://github.com/kennethwolters/litelm/actions/workflows/test.yml/badge.svg)](https://github.com/kennethwolters/litelm/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

litelm is a Python library for calling language models through a common, LiteLLM-style API. It handles provider selection, message translation, streaming, tool calls, and embeddings.

It does not include LiteLLM's Router, proxy server, response caching, or cost tracking. If you need those features, litelm is not a replacement for LiteLLM.

## Install

```bash
pip install litelm                # openai + httpx2
pip install litelm[anthropic]     # + anthropic SDK
pip install litelm[bedrock]       # + boto3
pip install litelm[all]           # everything
```

The direct runtime dependencies are `openai[httpx2]>=2.54.0,<3`, `httpx2>=2.12.0,<3`, and `pydantic>=2.11,<3` (already required transitively by OpenAI). Requests use HTTPX2; OpenAI still installs HTTPX as a transitive dependency. The Anthropic extra requires `anthropic>=1.5.0`, which also uses HTTPX2.

## Usage

```python
import litelm

# Basic completion
response = litelm.completion("openai/gpt-4o", messages=[{"role": "user", "content": "Hello!"}])
print(response.choices[0].message.content)

# Streaming
for chunk in litelm.completion("groq/openai/gpt-oss-20b", messages=[...], stream=True):
    print(chunk.choices[0].delta.content or "", end="")

# Embeddings
response = litelm.embedding("openai/text-embedding-3-small", input=["hello world"])
```

Every function has an async variant: `acompletion`, `aembedding`, `aresponses`, `atext_completion`.

The supported functions use LiteLLM-style arguments and response access patterns. This is a scoped compatibility layer, not a full drop-in replacement: check the feature table before changing imports. Non-empty `fallbacks=` raises `ValueError`; litelm does not execute failover. Empty or `None` fallbacks are accepted.

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

Routes to 19 providers via `"provider/model-name"` syntax. Use `api_base` for a custom OpenAI-compatible endpoint.

The first segment selects the provider; the remainder is the model identifier. For example, OpenRouter model `z-ai/glm-5.3-flash` becomes `openrouter/z-ai/glm-5.3-flash`. Unknown prefixes require an explicit `api_base` for real calls rather than silently sending the request to OpenAI. Bare names without a slash still select OpenAI.

| Provider | Env Var | Handler | Verified |
|----------|---------|---------|:--------:|
| OpenAI | `OPENAI_API_KEY` | OpenAI SDK | Yes |
| Anthropic | `ANTHROPIC_API_KEY` | Custom | Yes |
| Groq | `GROQ_API_KEY` | OpenAI-compat | Yes |
| Mistral | `MISTRAL_API_KEY` | Custom | Yes |
| xAI | `XAI_API_KEY` | OpenAI-compat | Yes |
| OpenRouter | `OPENROUTER_API_KEY` | OpenAI-compat | Yes |
| Azure | `AZURE_API_KEY` | OpenAI SDK (Azure) | Yes |
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

## Native decisions

`systemone` / `asystemone` call native decision APIs, not chat completions:

```python
response = litelm.systemone(
    "typesafe/jev-latest",
    state={"text": "Die Rechnung wurde doppelt bezahlt."},
    questions={
        "billing": {
            "type": "noul",
            "instructions": "Betrifft der Text eine Zahlungsfrage?",
        },
    },
)
print(response.answers["billing"].noul)  # True-probability, not generated confidence
```

- **TypeSafe:** `TYPESAFE_API_KEY`, optional `TYPESAFE_API_BASE` (default `https://api.typesafe.ai`).
- **Microsoft Foundry:** `azure_ai/<deployment-name>`, `AZURE_AI_API_KEY` and `AZURE_AI_API_BASE` (resource root). API keys use `api-key`; alternatively pass `azure_ad_token_provider=callback`. Async calls accept synchronous or asynchronous callbacks, refreshed on every attempt.
- Both accept text/JSON `state` and named `noul`, `choice` or `score` questions. Typed `NoulQuestion`, `ChoiceQuestion` and `ScoreQuestion` are also accepted. Choice criteria map 2–255 string labels to JSON descriptions; score criteria contain 2–10 ordered levels. At most 128 questions are supported.
- Results are `SystemOneResponse` objects with typed answers, provider distributions and optional usage. `model_dump_json()` / `model_validate_json()` preserve evidence. Missing usage remains `None`; no token counts or costs are invented. Thresholds, abstention and downstream actions belong to the application and require task-specific evaluation. Provider confidence and native probabilities are different signals.
- Unknown, missing or mismatched answers, nonfinite/out-of-range probabilities, incomplete distributions and distributions not summing to one within 0.001 raise `APIResponseValidationError`. These successful-but-invalid responses are never retried.
- Default timeout is 30 seconds **per attempt**, with zero retries. Explicit `num_retries` (0–10) retries only transport errors and HTTP 408/429/500/502/503/504, using bounded backoff. Repeated attempts can incur additional charges.
- No generation, tools, streaming, media, fallback, custom headers or completion callbacks. Unsupported arguments fail rather than being ignored. Select this explicit operation for decision deployments; provider prefixes alone do not identify a deployment's capabilities, and existing generic routing remains unchanged.
- Requests/responses are bounded to 2 MB/1 MB. HTTPS is required except for loopback development; redirects and compressed responses are refused. Requests specify `Accept-Encoding: identity` to enforce the response byte bound without unbounded decompression. API bases cannot contain credentials, queries or fragments. Exceptions omit provider bodies, requests and credentials.
- Optional `http_client=` borrows a sync/async HTTPX2 client without closing it. Otherwise each call owns and closes its client. Applications should reuse clients for sustained throughput.

These are protocol-level guarantees, not evidence of model accuracy or data residency. Verify the deployment's processing zone and applicable data-protection terms before sending personal data; a resource's geographic location alone is insufficient.

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

Catch provider API errors using litelm's exception hierarchy:

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

## Retries and Streaming Errors

`num_retries` configures provider SDK retries for chat completion calls (default: `0`). `max_retries` is an alias; `num_retries` takes precedence when both are supplied. This applies to OpenAI-compatible providers and the Anthropic, Bedrock, Cloudflare, and Mistral handlers. Each SDK controls which errors it retries and its backoff behavior.

Once a streaming response is open, iterator failures are surfaced without replaying the request. Transport timeouts become `Timeout`; other HTTPX2 request errors become `APIConnectionError`. Already-yielded output is not replayed. Close partially consumed streams explicitly with `close()` or `await aclose()` to release the response.

`ContextWindowExceededError` identifies input/context overflow. Invalid token credentials, parameter lengths, and output-only `max_tokens` limits are not classified as context overflow.

## Completion Callbacks

LiteLLM-compatible callbacks receive `(kwargs, response, start_time, end_time)`:

```python
def on_failure(kwargs, response, start_time, end_time):
    error = kwargs["exception"]
    print(kwargs["model"], error)

litelm.failure_callback.append(on_failure)
```

`success_callback` and `failure_callback` cover non-streaming `completion` and `acompletion` calls. Callback errors are logged without replacing the provider result or exception. Callback `kwargs` can contain messages and request metadata; observers should explicitly redact exported telemetry. Streaming lifecycle callbacks are not yet supported.

The older `success_callbacks` one-event-dict registry remains available for litelm 0.5.0 compatibility.

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

Use `api_base` to override an OpenAI-compatible server's endpoint. The `ollama`, `vllm`, `lm_studio`, and `lmstudio` prefixes work without an API key; an explicit key or the provider's environment variable takes precedence over the SDK placeholder.

```python
# vLLM
litelm.completion("vllm/my-model", messages=[...], api_base="http://localhost:8000/v1")

# Ollama
litelm.completion("ollama/llama3", messages=[...], api_base="http://localhost:11434/v1")

# LM Studio
litelm.completion("lm_studio/local-model", messages=[...], api_base="http://localhost:1234/v1")
```

## Development transparency

litelm is human-directed, AI-assisted software. Much of the code was written with Claude Code using Claude Opus 4.6/4.7. Code written from 2026-05-14 onward is written through Pi using GPT-5.5. Compatibility claims are based on tests and maintainer review, not AI authorship.

## Upstream attestation

Maintainer attestation, 2026-09-11: LiteLLM's routing/formatting changes were reviewed from `649eb2d` through `9a715df2`. The audit triaged 360 core-path commits, inspected upstream tests for potentially relevant behavior, and fixed the resulting compatibility gaps test-first. Local scoped tests: `262 passed, 55 skipped`; all 45 available-provider live tests and all 10 DSPy smoke tests also passed with the current dependency lock.

This attests litelm's declared routing/formatting/DSPy surface only, not full litellm compatibility.

## Status

**Alpha.** 580 own tests passing. The current scoped LiteLLM `9a715df2` baseline has 75 passing ported tests and no remaining actionable assertion/runtime failures.

[DSPy](https://github.com/stanfordnlp/dspy) drop-in verified — all 7 execution paths proven live (Predict, CoT, typed signatures, streaming, embeddings, tool use, multi-output).

Resilience validation, 2026-10-01: 580 local tests passed on Python 3.10 and 3.14; 49 upstream contract tests passed. All 52 selected live provider/DSPy tests passed, including the corrected Anthropic output-token error classification. Two context-window tests and the retired `gpt-3.5-turbo-instruct` test were excluded. Bedrock signing/retries have sync/async mocked-transport coverage, not live AWS verification.

## Tests

```bash
uv run --extra all pytest tests/ -x --ignore=tests/ported --timeout=10  # 580 non-live tests
bash scripts/ported_contract.sh                                        # 49 fast upstream contract tests
uv run --extra all pytest tests/test_live.py -m live --timeout=30       # 45 live provider tests
uv run pytest tests/test_dspy_smoke.py -m live --timeout=60             # 10 DSPy integration tests
```

Live tests require API keys in `.env.test`. Skipped by default; run with `-m live`.
