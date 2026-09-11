# Changelog

## 0.5.2 (2026-09-11)

- **Current upstream attestation** — audited LiteLLM core-path changes through `9a715df2`, refreshed the scoped ported baseline to 75 passing tests with no actionable failures, and made relevance decisions explicit and regression-tested.
- **Anthropic compatibility** — support future adaptive/always-on thinking models, fit reasoning budgets, request summarized reasoning, consume provider thinking-token usage, complete schema filtering/reference handling, and forward native structured outputs on supported Claude families.
- **Retry and error compatibility** — configure chat/embedding retries on cached clients, strip remaining LiteLLM-only control kwargs, and expose provider response headers on mapped exceptions.
- **Dependency refresh** — update Anthropic to 1.5.0, boto3/botocore to 1.43.92, DSPy to 3.3.1, LiteLLM to 1.100.1, and OpenAI to the latest DSPy-compatible 2.x release (2.54.0). Keep OpenAI 3.x valid for runtime users but constrained in the development group until DSPy's LiteLLM dependency supports it; a separate OpenAI 3.13.0 environment passed all 256 unit and 45 provider-live tests.
- **Live evidence** — 45 provider tests and 10 DSPy smoke tests pass with the final dependency lock.

## 0.5.1 (2026-06-24)

- **Cloudflare compatibility** — route `cloudflare/...` calls through Cloudflare's OpenAI-compatible `/ai/v1` endpoint, migrate legacy `/ai/run` bases, and normalize full `/chat/completions` URLs before handing them to the OpenAI SDK.
- **Replay-safe provider transforms** — sanitize Anthropic tool-use IDs before native API calls, and strip Mistral output-only assistant fields (`reasoning_content`, `thinking_blocks`) before replaying message history.
- **Upstream verification discipline** — add repo guidance requiring upstream behavior ports to inspect upstream tests first, then write local equivalents before implementation.

## 0.5.0 (2026-04-22)

- **Success callback registry** — append to `litelm.success_callbacks` to receive per-completion telemetry (model, provider, response, latency_ms). Fires for both `completion` and `acompletion` on every non-streaming success path (mock, custom-handler, direct-SDK). Callback exceptions are logged and swallowed so an observer can't cascade into call failure.
- Streaming is out of scope for 0.5.0 — usage data only lands in the final chunk and per-call firing semantics are ambiguous. Future work.

## 0.4.0 (2026-03-29)

- Pydantic `response_format` support with strict JSON Schema conversion
- Anthropic citation streaming, reasoning/thinking blocks, prompt caching, per-model output limits, reasoning-effort mapping, empty-block filtering, and schema filtering
- Azure AD token-provider passthrough
- Automated upstream drift watch

## 0.3.2 (2026-03-16)

- Remove `__slots__` from `TextCompletionResponse` for deepcopy and arbitrary attribute assignment compatibility

## 0.3.1 (2026-03-16)

- Add the `py.typed` marker, Ruff CI, documentation polish, and dict access for streaming delta types

## 0.3.0 (2026-03-16)

- Complete error mapping: `NotFoundError`, `PermissionDeniedError`, `UnprocessableEntityError` now correctly raised in all 4 handlers
- Export `get_llm_provider()` for litellm compat
- 129 own tests passing

## 0.2.0 (2026-03-13)

- Error wrapping across all 8 SDK call paths (`completion`, `acompletion`, `embedding`, `aembedding`, `responses`, `aresponses`, `text_completion`, `atext_completion`)
- Azure client cache key includes `api_version`
- Added `model_dump()` to `ChatCompletion`, `ChatCompletionChunk`, `ModelResponse`
- Bedrock client caching (thread-safe, no longer creates client per request)
- Mistral error wrapping
- CI installs optional SDKs

## 0.1.0 (2026-03-13)

- Initial release
- Core routing for 19 providers via `provider/model` syntax
- Custom handlers: Anthropic, Bedrock, Cloudflare, Mistral
- `completion`, `acompletion`, `embedding`, `aembedding`, `text_completion`, `atext_completion`, `responses`, `aresponses`
- Streaming + `stream_chunk_builder`
- Tool calling support
- Mock responses
- Own type system and exception hierarchy (no openai SDK dependency for types)
- DSPy drop-in verified: all 7 execution paths
- 6 providers verified live: OpenAI, Anthropic, Groq, Mistral, xAI, OpenRouter
