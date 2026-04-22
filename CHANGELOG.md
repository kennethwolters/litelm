# Changelog

## 0.5.0 (2026-04-22)

- **Success callback registry** — append to `litelm.success_callbacks` to receive per-completion telemetry (model, provider, response, latency_ms). Fires for both `completion` and `acompletion` on every non-streaming success path (mock, custom-handler, direct-SDK). Callback exceptions are logged and swallowed so an observer can't cascade into call failure.
- Streaming is out of scope for 0.5.0 — usage data only lands in the final chunk and per-call firing semantics are ambiguous. Future work.

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
