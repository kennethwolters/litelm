# Ground Truth (updated 2026-04-17)

litelm is a 2,660 LOC reimplementation of litellm's core routing+formatting. It works as a **DSPy backend for 6 providers** — that is the only verified claim. Everything beyond that is untested.

## What's Actually Proven

- **DSPy integration:** All 7 DSPy execution paths work (Predict, CoT, typed signatures, streaming, embeddings, ReAct, multi-output). 10 live smoke tests.
- **7 providers verified live:** openai, anthropic, groq, mistral, xai, openrouter, azure. 44 live tests covering basic completion, streaming, streaming+usage, tool calls, streaming tool calls, embeddings, error mapping.
- **161 own tests pass**, 47 skipped (live tests needing API keys).
- **65 of litellm's ported tests pass** out of 79 high-relevance tests (82.3%). The other 1060+ collected tests fail at import — they reference litellm internals (Router, proxy, provider-specific LLM modules) we intentionally don't implement. (Upstream re-synced 2026-03-16; test count changed due to litellm restructuring.)

## What's NOT Proven

- **10 providers with no API keys:** bedrock, cloudflare, together_ai, fireworks_ai, deepseek, perplexity, deepinfra, gemini, cohere, ollama. They route through OpenAI-compat which works for the 7 tested providers, but provider-specific quirks (like Mistral's `type=None` tool calls) can only be found with live testing.
- **Bedrock + Cloudflare handlers:** 420 LOC of custom handler code (SigV4 auth, raw httpx SSE parsing) with zero live testing. The Bedrock client cache was just added — also untested against a real endpoint.

## Honest Ported Test Breakdown (1,080 collected, synced 2026-03-16)

After fixing conftest to exclude nested dirs by basename (not just root-level glob):

| Bucket | Count | What it means |
|--------|------:|---------------|
| Passed | 56 | Working |
| Assertion failures | 12 | Our code runs but produces wrong results |
| Runtime errors | 6 | Our code runs but crashes |
| Needs API key | 29 | Would need live credentials |
| Skipped | 70 | Skipped by test logic |
| Timeout | 1 | Timed out |
| Import errors | 906 | Tests importing litellm internals we don't implement — **will never pass** |

Of the 14 high-relevance failures, **none are actionable** — all are out of scope (compactifai 7, fallbacks/Router 3, gemini no_api_key 2, poetry deps 1, Anthropic system message format 1). 0 regressions vs previous baseline; 9 fewer passes are tests removed upstream.

## Project Plan

### Phase 1: Harden Core (complete) — litelm as litellm drop-in for DSPy

- Core routing, completion, streaming, embedding, text_completion, responses API
- 4 custom handlers (anthropic, bedrock, cloudflare, mistral)
- Own type system, exception hierarchy, error wrapping across all SDK paths
- DSPy contract fully satisfied for 6 providers
- Client caching (thread-safe, Azure api_version-aware, Bedrock cached)
- All actionable ported test failures fixed: exception kwargs, `__getitem__`, images, kwarg stripping, mock streaming, mock_completion, n support
- No remaining actionable ported test failures (14 high-relevance failures all out of scope)
- Error mapping complete: NotFoundError, PermissionDeniedError, UnprocessableEntityError mapped in all 4 handlers
- `get_llm_provider()` exported for litellm compat

### Phase 2: dspy-lite — the actual goal

Fork DSPy, replace `import litellm` with `import litelm`, remove litellm dependency entirely. Strip proxy/router/caching/budgeting/etc. This is where the value is — litelm is just the enabler.

### Phase 3: More providers (as needed)

Only worth doing when a specific use case demands it. Each new provider key can uncover quirks like Mistral's `type=None`. Bedrock and Cloudflare handlers need live testing before they can be called proven.

### What's explicitly out of scope (and stays out)

Router, proxy, caching, budgeting, agents, guardrails, image gen, audio, OCR, fine-tuning, batches, assistants, scheduler, callbacks/integrations (opik, mlflow, etc), provider config registry, a2a protocol, compactifai.

## Architecture Comparison: litellm vs litelm (audited 2026-03-16)

### What litellm actually is

litellm is ~40K LOC across 1,667 Python files. Its public namespace exports **1,323 attributes** — ours exports 44. The difference is almost entirely feature sprawl that doesn't touch the core routing path.

**litellm's layers:**

| Layer | LOC | What it does | In litelm? |
|-------|-----|-------------|:---:|
| `main.py` — core completion engine | 7,601 | `completion()`, `acompletion()`, `embedding()`, streaming, error handling | **Yes** (300 LOC) |
| `utils.py` — helpers | 9,313 | Token counters, param validation, model info lookups, `get_optional_params()` | **No** |
| `router.py` — load balancer | 9,611 | Fallback routing, round-robin, cost-based, TPM/RPM strategies | **No** |
| `cost_calculator.py` — pricing | 2,253 | `completion_cost()` from 500KB model pricing JSON | **No** |
| `llms/` — 90+ provider handlers | ~15K | Per-provider translation classes | **4 handlers** (1,135 LOC) |
| `types/` — 156 type files | ~8K | Response types, provider-specific types, proxy types | **1 file** (361 LOC) |
| `exceptions.py` | 1,032 | Exception hierarchy inheriting from openai SDK | **Yes** (own hierarchy, no openai dep) |
| `proxy/` — API server | ~20K | FastAPI server, auth, user/team/key management | **No** |
| `caching/` — cache backends | ~2K | In-memory, Redis, dual-cache, semantic cache | **No** |
| `integrations/` — 50+ callbacks | ~10K | Langfuse, Datadog, MLflow, OpenTelemetry hooks | **No** |
| 117 model name registries | — | 2,300+ model names for bare-name → provider routing | **No** |

### What we deliberately don't replicate (and why)

**`get_optional_params()` (1,000+ LOC)** — Whitelists valid kwargs per provider. Our approach: pass kwargs through, let the SDK reject bad ones. The SDK error message is clear. The Anthropic handler already handles its own params. Maintaining param lists for 80+ providers is a maintenance burden with near-zero value for the `provider/model` prefix pattern.

**`completion_cost()` / `get_model_info()` / model pricing JSON** — Lookup table that goes stale. Not routing or formatting. If dspy-lite needs cost tracking, use the provider's billing dashboard or a separate package.

**`token_counter()`** — Token counting via tiktoken. Useful but not core. DSPy doesn't call it through litellm. Users can call tiktoken directly.

**Bare model name routing** — litellm routes `claude-3-haiku-20240307` (no prefix) → `anthropic` via 117 model name lists (2,300+ entries). We require `anthropic/claude-3-haiku-20240307`. This is intentional — explicit prefixes are unambiguous and don't require a stale registry. DSPy always uses `provider/model` format.

**Exception inheritance from openai SDK** — litellm exceptions inherit from `openai.NotFoundError`, `openai.RateLimitError`, etc. Code that catches openai SDK exceptions catches litellm's too. **Ours don't — intentionally.** Inheriting from openai would make it a hard dependency; currently it's optional. DSPy only catches `ContextWindowExceededError` by our class name, so this doesn't matter for the mission.

**Router / fallbacks** — `completion(fallbacks=["model_b"])`. This is orchestration, not routing. dspy-lite can implement its own retry logic in 20 lines.

**Callbacks / logging integrations** — 50+ integration hooks (langfuse, datadog, etc). Out of scope. dspy-lite doesn't need observability hooks in the LLM layer.

**`reasoning_items` on Message/Delta** — ZDR round-trip of OpenAI Responses-API reasoning items through chat-completion shapes. Populated only by `completion_extras/litellm_responses_transformation/transformation.py` (the /chat/completions↔Responses adapter we don't implement). Audited 2026-04-17: 0 access via litellm across DSPy, OpenHands, MLflow, OpenInterpreter, Arize Phoenix, Instructor, Agno, CrewAI, AutoGen. CrewAI/Instructor use reasoning items but via the openai SDK directly. Dismissed.

**Anthropic `text_tokens` in PromptTokensDetails** — upstream added `text_tokens=raw_input_tokens` for cost calculators to distinguish pre- vs post-cache totals. Cost calculation is out of scope; no consumer reads it. Dismissed.

**Anthropic invalid-thinking-signature retry** — upstream retries `/v1/messages` once after stripping thinking blocks on HTTP 400 signature errors. Retry/orchestration layer (same reason we don't implement Router/fallbacks). Dismissed.

### What we ARE missing (actionable) — audited 2026-03-17

Deep comparative audit against litellm's actual source revealed these behavioral gaps within our scope. All are in the core routing+formatting path that a litellm drop-in user would hit.

**1. Tool call ID generation** — litellm auto-generates UUID for `ChatCompletionMessageToolCall.id` when not provided. litelm defaults to `""`. Code checking `if tool_call.id:` behaves differently. Fix in `_types.py`.

**2. Anthropic max_tokens per model** — litellm looks up per-model max output tokens. litelm hardcodes `max_tokens=4096` in `_anthropic.py:_build_request_kwargs()`. Wrong for Opus (32k output), Sonnet (8192). Causes silent truncation or API errors.

**3. ~~thinking_blocks + reasoning_tokens~~ (DONE 2026-03-19)** — `ChatCompletionMessage.thinking_blocks` and `ChoiceDelta.thinking_blocks` populated from Anthropic thinking blocks. `CompletionUsage.completion_tokens_details` / `.prompt_tokens_details` added (plain dicts). Anthropic cache tokens (`cache_creation_input_tokens`, `cache_read_input_tokens`) → `prompt_tokens_details`. Streaming: `thinking_delta` and `signature_delta` emit `thinking_blocks` on delta; `stream_chunk_builder` accumulates text parts and flushes on signature. 15 new tests.

**4. ~~Prompt caching (cache_control)~~ (DONE 2026-03-19)** — `cache_control` on system content blocks, user content blocks, image blocks, and tool definitions now passed through to Anthropic API. `_extract_system` already preserved dict blocks; `_translate_content` and `_translate_tools` updated. 7 new tests.

**5. ~~reasoning_effort → budget_tokens mapping~~ (DONE 2026-03-19)** — `_map_reasoning_effort()` maps OpenAI `reasoning_effort` to Anthropic `thinking`/`output_config`. Standard models: `low`→1024, `medium`→2048, `high`→4096 budget_tokens. Opus 4.6: `thinking={"type": "adaptive"}` + `output_config={"effort": X}`. Explicit `thinking` kwarg takes precedence. 8 new tests.

**6. ~~Empty text block filtering~~ (DONE 2026-03-19)** — `_extract_system()` and `_translate_content()` now skip empty text blocks (empty string content, empty text in list blocks, empty text dict blocks). Prevents Anthropic API rejection of `{"type": "text", "text": ""}`. 4 new tests.

**7. ~~JSON schema filtering for Anthropic~~ (DONE 2026-03-19)** — `_filter_schema()` recursively strips unsupported JSON schema fields (`maxItems`, `minItems`, `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `minLength`, `maxLength`) and appends them as description notes. Applied in `_translate_tools()` on `input_schema`. 5 new tests.

**8. ~~Usage.completion_tokens_details~~ (DONE 2026-03-19)** — Implemented as part of gap #3. `CompletionUsage` now has `completion_tokens_details` and `prompt_tokens_details` slots (plain dicts, default `None`). `dict(usage)` includes them. `stream_chunk_builder` passes through latest non-None values. Anthropic handler populates `prompt_tokens_details` from cache tokens.

### Why the ported test gap is honest

906 of 1,080 collected tests fail at import because they `from litellm.llms.anthropic.chat.transformation import ...` or `from litellm.proxy._types import ...`. These are litellm's **internal architecture** — per-provider translation classes, proxy request types, router strategy implementations. We don't replicate this architecture; we replaced it with 2,660 LOC that produces the same outputs.

The 56 passing tests exercise: mock_completion, streaming mock, n>1, kwarg stripping, exception exports, Mistral quirks, text_completion token IDs, shared sessions, responses API, embedding transformations. These are the tests that test the **API contract** rather than internal wiring — exactly our scope.

### DSPy touchpoints (exhaustive AST audit, 2026-03-16)

DSPy references litellm in exactly **16 files** (out of 138 total .py files). The actual callable API surface used:

```
litellm.completion / acompletion          — clients/lm.py
litellm.text_completion / atext_completion — clients/lm.py
litellm.responses / aresponses            — clients/lm.py
litellm.embedding / aembedding            — clients/embedding.py
litellm.stream_chunk_builder              — clients/lm.py
litellm.supports_function_calling         — adapters/base.py
litellm.supports_response_schema          — adapters/json_adapter.py
litellm.supports_reasoning                — adapters/types/reasoning.py
litellm.get_supported_openai_params       — adapters/json_adapter.py
litellm.ModelResponseStream               — streaming (3 files)
litellm.ContextWindowExceededError        — chat_adapter, react
litellm._logging.verbose_logger           — clients/__init__.py
litellm.telemetry / cache / suppress_debug_info — module attrs
```

Every one of these is implemented and verified. The remaining 1,279 litellm attributes (provider config classes, Router, proxy types, model lists, callback hooks, 250+ functions for image/audio/batch/assistant/fine-tuning APIs) are never touched by DSPy.

# Philosophy

Strip litellm to its core routing+formatting logic, prove correctness against its own tests, keep the result small enough to hold in your head. **Simplicity is the goal, not the constraint.**

- **Test each provider directly.** No "assume OpenAI-compat works." Each provider is an independent experiment.
- **Never use `NotImplementedError` stubs.** Either implement or don't export.
- **The DSPy contract is the acceptance test.** The exact functions, types, and access patterns DSPy exercises define what we must implement. Everything else is noise.
- **Claims need evidence.** A test that makes a real API call and verifies the response shape is evidence. "The code looks right" is not.

# Engineering Values

- **Outside-in**: Start from interface/dependency boundary, work inward
- **Dependencies first**: Nail down deps and contracts before implementation
- **TDD**: Tests drive design. Write failing tests first, then make them pass
- **Verify everything**: "If I didn't see it, it didn't happen." Run it. Show output. No handwaving. "The process started" is not verification — wait for it to finish and read the output
- **CI is a hard gate**: Never push code that breaks CI. Run lint + format + tests locally before every push. Check CI status after pushing. A red CI is a bug, not a TODO
- **Challenge plans**: Before implementing, question cost/benefit. Flag narrow ROI before writing code
- **Direct instructions = execute**: Don't ask confirmation when told to do something
- **No unverified claims**: Never state plans, targets, or capabilities that aren't confirmed by the user or proven by tests. Repeating something from old docs doesn't make it true
- **Write for the median user**: Public-facing content (README, docs, examples) should target the most common use case, not the most impressive or niche one

# Project Context

Version: `0.1.0` (in both `pyproject.toml` and `litelm/__init__.__version__`).

## Ported Tests

litellm's test suite is synced **unmodified** into `tests/ported/` (gitignored, not committed). A `conftest.py` shims `sys.modules["litellm"] = litelm` at runtime so `import litellm` in test files resolves to litelm without string replacement.

### Setup & Run

```bash
bash scripts/sync_litellm_tests.sh                    # shallow-clone litellm, copy tests/
bash scripts/ported_baseline.sh                        # run + categorize
# or manually:
uv run pytest tests/ported/ --tb=line -q --timeout=10 --continue-on-collection-errors
```

- `--continue-on-collection-errors` is REQUIRED or pytest aborts after collection phase
- `tests/ported/conftest.py` does sys.modules shimming + excludes out-of-scope dirs
- `tests/ported/.litellm-commit` records which upstream commit was synced
- JUnit XML: `--junit-xml=/tmp/litelm_results.xml`
- Categorization: `uv run python scripts/categorize_failures.py /tmp/litelm_results.xml`

### sys.modules shimming (how it works)

`conftest.py` maps `sys.modules["litellm"] = litelm` and `sys.modules["litellm.types"] = litelm.types` etc for all submodules that exist. Tests that `import litellm` or `from litellm.types import ModelResponse` get litelm's implementations. Tests that import deep submodules (`litellm.llms.anthropic`, `litellm.proxy.utils`) fail at import time — these are correctly categorized as `import_absent_feature`.

Advantages over the old sed approach:
- No string corruption (`"litellm/gpt-4"` model names stay correct)
- No leaked credentials (litellm's test fixtures unmodified)
- Easy sync: re-run `sync_litellm_tests.sh` to track upstream
- Thin re-exports (`litelm/types.py` → `_types`, etc) unlock submodule imports

### Baseline (synced from litellm HEAD, 2026-03-16)

56 passed, 906 import errors, 12 assertion failures, 6 runtime errors, 29 needs_api_key, 70 skipped, 1 timeout. conftest uses `pytest_ignore_collect` hook for recursive basename matching — properly excludes nested out-of-scope dirs. 0 regressions vs previous baseline.

**Experiment pipeline:** `bash scripts/ported_experiment.sh [--skip-sync]` — syncs, runs, categorizes, diffs against previous baseline. Results in `/tmp/litelm_experiment_YYYYMMDD_HHMMSS/`.

Own tests: 185 passing, 47 skipped (live tests need `set -a && . .env.test && set +a`; 37 live + 10 DSPy smoke tests all pass)

## Key Files

- `litelm/__init__.py` — public API surface, capability stubs, compat shims, `__version__`
- `litelm/_types.py` — own type system (361 LOC): `ModelResponse`, `ModelResponseStream`, `ChatCompletion`, `Choice`, `Message`, `Usage`, etc
- `litelm/_exceptions.py` — own exception hierarchy rooted at `LitelmError`
- `litelm/_providers.py` — PROVIDERS dict + `parse_model()` routing
- `litelm/_completion.py` — completion/acompletion/mock_completion/stream_chunk_builder + `_map_openai_error`
- `litelm/_embedding.py` — embedding/aembedding + `EmbeddingResponse`/`_EmbeddingItem` wrappers
- `litelm/_text_completion.py` — text_completion/atext_completion + `TextCompletionResponse`
- `litelm/_responses.py` — responses/aresponses (OpenAI Responses API)
- `litelm/_dispatch.py` — lazy handler dispatch: `get_handler()` returns module or None
- `litelm/_client_cache.py` — thread-safe OpenAI/Azure client cache + `close_async_clients()`
- `litelm/_logging.py` — `verbose_logger` NullHandler shim for DSPy
- `litelm/providers/_anthropic.py` — Anthropic native handler (518 LOC, largest file)
- `litelm/providers/_bedrock.py` — AWS Bedrock via SigV4Auth httpx hook + client cache (198 LOC)
- `litelm/providers/_cloudflare.py` — Cloudflare Workers AI via raw httpx (224 LOC)
- `litelm/providers/_mistral.py` — Mistral handler (name stripping, empty content, tool_call type fix)
- `litelm/types.py` — thin re-export of `_types` (`from litelm._types import *`)
- `litelm/exceptions.py` — thin re-export of `_exceptions`
- `litelm/responses.py` — thin re-export of `_responses`
- `litelm/py.typed` — PEP 561 marker
- `scripts/sync_litellm_tests.sh` — shallow-clones litellm, copies tests/ into tests/ported/
- `scripts/ported_baseline.sh` — runs ported tests + categorization pipeline
- `scripts/ported_experiment.sh` — full experiment pipeline: sync, discover, run, categorize, diff
- `scripts/categorize_failures.py` — parses JUnit XML into failure buckets; `--diff OLD.xml NEW.xml` mode for baseline comparison

## Architecture: Call Flow

```
completion("anthropic/claude-3-haiku", messages=[...])
  → kwargs.pop("mock_response") → if truthy, return synthetic ModelResponse
  → _prepare_call() → strips litellm kwargs, parses model, resolves api_key
  → get_handler("anthropic") → lazy-loads litelm.providers._anthropic
  → handler.completion() → translates to Anthropic SDK, calls API, translates back
  → ModelResponse(ChatCompletion(...))

completion("openai/gpt-4o", messages=[...])
  → _prepare_call() → provider="openai", model="gpt-4o"
  → get_handler("openai") → None (OpenAI-compat, no custom handler)
  → get_sync_client() → cached OpenAI() client
  → client.chat.completions.create() → SDK response
  → ModelResponse(response)
```

### Provider Routing (`_providers.py`)

`parse_model("provider/model")` → `(provider, model_name, base_url, api_key, api_version)`

- PROVIDERS dict maps 25+ provider names to `(base_url, api_key_env_var)` tuples
- `base_url=None` for openai/azure (SDK handles it) and bedrock (boto3 auth)
- Unknown providers treated as OpenAI-compatible with custom base_url
- Azure special-cased: reads `AZURE_API_BASE`, `AZURE_API_KEY`, `AZURE_API_VERSION` from env, returns 5-tuple with api_version
- Bare model name (no `/`) defaults to openai
- `api_key`, `api_base` kwargs override registry values

### Handler Dispatch (`_dispatch.py`)

`get_handler(provider)` → handler module or `None`

4 custom handlers (need SDK translation, not OpenAI-compat):
- `anthropic` → `litelm.providers._anthropic`
- `bedrock` → `litelm.providers._bedrock`
- `cloudflare` → `litelm.providers._cloudflare`
- `mistral` → `litelm.providers._mistral`

All other providers → `None` → use OpenAI SDK via `get_sync_client()`/`get_async_client()`.

Handlers are lazy-loaded via `importlib.import_module()` on first access, cached in `_loaded` dict.

### Client Cache (`_client_cache.py`)

Thread-safe double-checked locking. Cache key: `("sync"/"async", provider, base_url, api_key, max_retries, api_version)`.

- `get_sync_client()` / `get_async_client()` → returns cached `OpenAI` / `AsyncOpenAI` (or `AzureOpenAI` / `AsyncAzureOpenAI`)
- `close_async_clients()` → closes + clears all cached async clients (exported as `close_litelm_async_clients`)
- Azure: `api_version` defaults to `"2024-02-01"`, included in cache key so different versions get different clients
- `_require_openai()` → `ImportError` with install hint if openai SDK missing

### Kwarg Stripping (`_prepare_call()`)

litellm-specific kwargs that would cause OpenAI SDK errors: `cache`, `num_retries`, `retry_strategy`, `caching`. All popped before forwarding. `headers` → converted to `extra_headers`.

### mock_response

Available on `completion`, `acompletion`, `text_completion`, `atext_completion`. When passed:
- `mock_response=True` → content `"mock"`
- `mock_response="anything"` → content `str(mock_response)`
- Returns synthetic `ModelResponse` / `TextCompletionResponse` with zero usage, skips API call entirely

### timeout

Explicit kwarg on `completion`, `acompletion`, `embedding`, `aembedding`, `text_completion`, `atext_completion`. Passed through to SDK `create()` when not None.

### Error Wrapping (`_completion.py:_map_openai_error`)

All OpenAI SDK exceptions are caught and re-raised as litelm exceptions. `_map_openai_error(e)` maps:

| openai SDK exception | litelm exception |
|---------------------|-----------------|
| `openai.BadRequestError` | `BadRequestError` (or `ContextWindowExceededError` if keyword match) |
| `openai.RateLimitError` | `RateLimitError` |
| `openai.AuthenticationError` | `AuthenticationError` |
| `openai.APITimeoutError` | `Timeout` |
| `openai.APIConnectionError` | `APIConnectionError` |
| `openai.InternalServerError` | `InternalServerError` |
| `openai.NotFoundError` | `NotFoundError` |
| `openai.PermissionDeniedError` | `PermissionDeniedError` |
| `openai.UnprocessableEntityError` | `UnprocessableEntityError` |
| `openai.APIStatusError` | `APIStatusError` |
| any other `openai.APIError` | `LitelmError` |

**Coverage:** All 8 SDK call paths wrap exceptions — `completion`, `acompletion`, `embedding`, `aembedding`, `responses`, `aresponses`, `text_completion`, `atext_completion`. Custom handlers (Anthropic, Bedrock, Cloudflare) have their own `_map_error()` with the same pattern. Mistral uses the shared `_map_openai_error`.

**Two-stage catch in completion paths:** `_bad_request_errors` caught first for `ContextWindowExceededError` detection (via `_wrap_context_window_error`), then `_openai_errors` catches everything else via `_map_openai_error`. This preserves the existing context-window-specific logic while adding comprehensive wrapping.

**Note:** litelm exceptions do NOT inherit from openai SDK exceptions. Code that catches `openai.RateLimitError` won't catch `litelm.RateLimitError`. This is intentional — litelm has its own exception hierarchy with no openai dependency.

## Type System (`_types.py`)

Own types, no openai SDK dependency. All use `__slots__` for memory efficiency.

### Construction

All types accept dict construction: `Choice(**{"message": {"content": "hi"}})`. Nested dicts auto-coerced — `ChatCompletionMessage(**msg_dict)`, `CompletionUsage(**usage_dict)`, etc.

### Key types

| Type | Wraps | Notes |
|------|-------|-------|
| `ModelResponse` | `ChatCompletion` | `__getattr__` delegates to inner completion. Supports `response["choices"]` dict-access, `.json()`, `.model_dump()`, `.cache_hit`, `._hidden_params` |
| `ModelResponseStream` | `ChatCompletionChunk` | Same pattern. Has `.predict_id` (assignable, DSPy uses) |
| `CompletionUsage` | — | `__iter__` yields `(key, value)` pairs → enables `dict(usage)` (DSPy pattern). Has `.completion_tokens_details`, `.prompt_tokens_details` (plain dicts or None) |
| `ChatCompletionMessage` | — | `.role`, `.content`, `.tool_calls`, `.reasoning_content`, `.thinking_blocks`. Auto-coerces tool_calls from dicts |
| `Choice` | — | `.index`, `.message`, `.finish_reason`, `.logprobs` |
| `ChatCompletionMessageToolCall` | — | `.id`, `.type`, `.function` (auto-coerced from dict) |

### Compatibility aliases (exported from `__init__.py`)

`Message` = `ChatCompletionMessage`, `Choices` = `Choice`, `Usage` = `CompletionUsage`, `Delta` = `ChoiceDelta`, `StreamingChoices` = `ChunkChoice`. These match litellm's public names.

### Helpers

- `_to_dict(obj)` — recursive serialization (slots-aware) for `.json()`, `.model_dump()`, `.model_dump_json()`
- `_coerce_choice(dict)` — fills defaults (role="assistant", finish_reason="stop", index=0)
- `_coerce_delta_tool_call(dict)` — dict → `ChoiceDeltaToolCall`

## Exception Hierarchy (`_exceptions.py`)

```
LitelmError (base)
├── RateLimitError
├── AuthenticationError
├── BadRequestError
│   ├── ContentPolicyViolationError
│   └── ContextWindowExceededError
├── Timeout
├── InternalServerError
├── APIConnectionError
├── APIStatusError
├── PermissionDeniedError
├── NotFoundError
├── UnprocessableEntityError
├── ServiceUnavailableError
└── BadGatewayError
```

All share constructor: `__init__(message="", response=None, body=None, request=None)`.

`ContextWindowExceededError` accepts extra positional args `(llm_provider, model)` silently for litellm compat.

## Module-Level Attrs (`__init__.py`)

Exported API surface + DSPy compat shims + capability functions.

| Attr/Function | Purpose |
|---|---|
| `__version__ = "0.1.0"` | Package version |
| `telemetry = False` | DSPy disables litellm telemetry |
| `cache = None` | DSPy disables litellm caching |
| `suppress_debug_info = False` | DSPy logging config |
| `add_function_to_prompt = False` | litellm compat attr |
| `get_secret(name)` | Returns `os.environ.get(name)` |
| `close_litelm_async_clients()` | Alias for `_client_cache.close_async_clients()` |
| `supports_function_calling(model=)` | Always `True` |
| `supports_response_schema(model=, custom_llm_provider=)` | Always `True` |
| `supports_reasoning(model=)` | Pattern match: o1/o3/o4/gpt-5/deepseek-reasoner + claude-3.7+ |
| `get_llm_provider(model=, custom_llm_provider=, api_base=, api_key=)` | Returns `(model_name, provider, api_key, api_base)` — thin wrapper around `parse_model()` |
| `get_supported_openai_params(model=, custom_llm_provider=)` | Returns param list for known providers, `None` for unknown |

## Test Structure

### Own tests (`tests/`)

| File | Tests | What |
|------|-------|------|
| `test_completion.py` | 14 | `_prepare_call`, mock_response, header conversion, `_map_openai_error` (10 error type mappings), SDK error wrapping in `completion()` |
| `test_embedding.py` | 4 | sync/async, kwargs stripping, handler routing |
| `test_capabilities.py` | 8 | `supports_reasoning`, `get_supported_openai_params` |
| `test_types.py` | — | Type construction, dict access, JSON serialization, `model_dump()` |
| `test_exceptions.py` | — | Exception hierarchy, constructor args |
| `test_providers.py` | — | `parse_model` routing for all provider formats |
| `test_dispatch.py` | — | Handler lazy-loading, None for OpenAI-compat |
| `test_responses.py` | — | responses API kwarg handling |
| `test_bedrock_errors.py` | — | Bedrock error mapping |
| `test_client_cache.py` | — | Client caching, Azure api_version isolation, close_async_clients |
| `test_get_llm_provider.py` | 5 | `get_llm_provider` routing, custom_llm_provider override, api_base passthrough |
| `test_stream_chunk_builder.py` | — | Stream reassembly with real captured testdata (`stream_chunk_testdata.py`) |
| `test_live.py` | 37 | Live API tests for 6 providers (basic, stream, tools, usage, errors). Requires `-m live` + `.env.test` |
| `test_dspy_smoke.py` | 10 | All 7 DSPy execution paths. Requires `-m live` + `.env.test` |
| `conftest.py` | — | Loads `.env.test` into env, auto-skips `live`-marked tests unless `-m live` |

Total: 185 passing, 47 skipped (live tests)

### Ported tests (`tests/ported/`, gitignored)

litellm's test suite synced unmodified via `scripts/sync_litellm_tests.sh`. `conftest.py` does `sys.modules["litellm"] = litelm` shimming at runtime. Excludes ~36 out-of-scope dirs via `pytest_ignore_collect` (recursive basename matching). 56 passing, 906 import errors (deep submodule imports that don't exist in litelm).

## Provider Handler Details

### Anthropic (`providers/_anthropic.py`, 605 LOC)

Full bidirectional translation between OpenAI and Anthropic message formats.

Key functions:
- `_extract_system()` — separates system messages from conversation
- `_translate_content()` — OpenAI content (str or content blocks list) → Anthropic blocks; handles `image_url` with base64
- `_translate_messages()` — full message translation + merging adjacent same-role messages (Anthropic requires alternating roles)
- `_translate_tools()` / `_translate_tool_choice()` — OpenAI tool format → Anthropic
- `_build_request_kwargs()` — constructs Anthropic API request; per-model `max_tokens` lookup via `_get_max_tokens()`
- `_build_model_response()` — Anthropic `Message` → OpenAI `ChatCompletion` (handles `tool_use` blocks → `ChatCompletionMessageToolCall`, `thinking` blocks → `thinking_blocks` + `reasoning_content`, `stop_reason` mapping, cache tokens → `prompt_tokens_details`)
- `_map_error()` — Anthropic SDK exceptions → litelm exceptions (keyword matching for context window)
- Streaming: full chunk assembly pipeline (`message_start`, `content_block_start/delta`, `message_delta`). Handles `thinking_delta` → `thinking_blocks` + `reasoning_content`, `signature_delta` → `thinking_blocks`

### Bedrock (`providers/_bedrock.py`, 200 LOC)

Uses OpenAI-compatible endpoint with AWS SigV4 auth via httpx transport hook.

- `_get_bedrock_url(region)` — `https://bedrock-runtime.{region}.amazonaws.com/openai/v1`
- `_get_openai_client()` — gets or creates cached OpenAI client with `httpx.Client`/`AsyncClient` using boto3 `SigV4Auth` for request signing. Thread-safe double-checked locking, keyed on `(base_url, region, async_client)`
- `_map_error()` — openai SDK exceptions → litelm exceptions (same pattern as `_completion._map_openai_error`)
- Region from `AWS_REGION` or `AWS_DEFAULT_REGION` env

### Cloudflare (`providers/_cloudflare.py`, 224 LOC)

Raw httpx implementation (not OpenAI SDK — Cloudflare API is not fully OpenAI-compat).

- `_get_config()` — reads `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` from env
- `_build_url()` — `https://api.cloudflare.com/client/v4/accounts/{id}/ai/run/{model}`
- `_build_request_body()` — transforms OpenAI messages to Cloudflare format
- `_parse_response()` / `_parse_stream_line()` — Cloudflare JSON → `ModelResponse` / SSE parsing
- Full streaming via httpx `stream()` with SSE line parsing

### Mistral (`providers/_mistral.py`)

Thin wrapper around OpenAI-compat path, fixing Mistral API quirks:

- `_transform_messages()` — strips `name` field from non-tool messages (Mistral rejects it)
- `_fix_response()` — normalizes `tool_call.type` from `None` → `"function"`, converts empty string content → `None`
- Wraps responses in `ModelResponse`/`ModelResponseStream`

## DSPy Contract (the real API surface)

DSPy exercises exactly these functions/patterns on litellm:
- `completion()` / `acompletion()` — with `cache`, `num_retries`, `retry_strategy`, `headers`, `stream`, `stream_options`, `timeout`, `mock_response`
- `text_completion()` / `atext_completion()` — legacy completions API
- `responses()` / `aresponses()` — OpenAI Responses API (with `previous_response_id`)
- `embedding()` / `aembedding()`
- `stream_chunk_builder(chunks)`
- `supports_function_calling(model=)`
- `supports_response_schema(model=, custom_llm_provider=)`
- `supports_reasoning(model=)` — pattern-based: o1/o3/o4/gpt-5/deepseek-reasoner/claude-3.7+
- `get_supported_openai_params(model=, custom_llm_provider=)`
- `get_secret(secret_name)` — reads env var
- `close_litelm_async_clients()` — cleanup cached async clients
- `ContextWindowExceededError` — the only exception DSPy catches explicitly
- `ModelResponseStream` — imported by name in DSPy streaming code

Response access patterns DSPy uses:
- `response.choices[i].message.content` / `.reasoning_content` / `.thinking_blocks` / `.tool_calls` / `.finish_reason`
- `response.usage.prompt_tokens` / `.completion_tokens` / `.total_tokens` / `.completion_tokens_details` / `.prompt_tokens_details`, `dict(response.usage)`
- `getattr(response, "cache_hit", False)`
- `chunk.predict_id` (assignable on stream chunks)
- Embedding: `response.data[i]["embedding"]`

## DSPy Integration (verified 2026-03-13)

**litelm can replace litellm for all 7 DSPy execution paths.** Proven with 10 live smoke tests covering: Predict, CoT, typed Pydantic signatures (JSONAdapter + response_format), streaming (dspy.streamify + async iteration), embeddings (dspy.Embedder), tool use (dspy.ReAct), multi-output (n=3), multi-provider (Anthropic + Groq), and error recovery (ContextWindowExceededError propagation).

### Mechanism

`sys.modules["litellm"] = litelm` before `import dspy`. Works because DSPy only touches litellm's top-level namespace — no deep submodule imports except `litellm._logging.verbose_logger` (shimmed).

### DSPy's litellm touchpoints (exhaustive, 10 files)

**Module-level `import litellm`** (7 files): `clients/lm.py`, `clients/embedding.py`, `clients/__init__.py`, `adapters/base.py`, `adapters/json_adapter.py`, `adapters/types/reasoning.py`, `streaming/streamify.py`

**`from litellm import X`** (3 files): `ModelResponseStream` (streaming_listener, streamify, base_type), `ContextWindowExceededError` (chat_adapter, react)

**Submodule import**: `from litellm._logging import verbose_logger` — DSPy configures litellm's logger at import. Shimmed via `litelm/_logging.py`.

**Attribute mutations**: `litellm.telemetry = False`, `litellm.cache = None`, `litellm.suppress_debug_info = True` — all accepted as module-level attrs on litelm.

### Compatibility shims added

| Shim | Location | Purpose |
|------|----------|---------|
| `telemetry = False` | `__init__.py` | DSPy disables litellm telemetry |
| `cache = None` | `__init__.py` | DSPy disables litellm caching (uses its own) |
| `suppress_debug_info = False` | `__init__.py` | DSPy sets during logging config |
| `add_function_to_prompt = False` | `__init__.py` | litellm compat attr (ported tests check it) |
| `_logging.py` | `litelm/_logging.py` | `verbose_logger` NullHandler — DSPy configures log levels |
| `types.py`, `exceptions.py`, `responses.py` | `litelm/` | Thin re-exports for `import litelm.types` compat |

### Test

```bash
set -a && . .env.test && set +a && uv run pytest tests/test_dspy_smoke.py -m live -v --timeout=60
```

## Provider Evidence Matrix

Each provider needs verification on: basic completion, streaming, tool use, error mapping, response shape.

| Provider | Key Available | Basic | Stream | Stream+Usage | Tools | Stream Tools | Embed | Errors | Proven? |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| openai | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes (context window, wrapping) | Partial |
| anthropic | Yes | Yes | Yes | Yes (native) | Yes | Yes (chunk-inspected) | — | Yes (max_tokens, own `_map_error`) | Partial |
| groq | Yes | Yes | Yes | Yes | Yes (70b) | Yes (70b) | — | Yes (wrapping) | Partial |
| mistral | Yes | Yes | Yes | Yes | Yes (type fix) | Yes | Yes | Yes (wrapping) | Partial |
| xai | Yes | Yes | Yes | Yes (inflated total) | Yes | Yes | — | Yes (wrapping) | Partial |
| openrouter | Yes | Yes | Yes | Yes | Yes (gpt-4o-mini) | Yes (gpt-4o-mini) | — | Yes (wrapping) | Partial |
| azure | Yes | Yes | Yes | Yes | Yes (gpt-5.4-nano) | Yes (gpt-5.4-nano) | — | Yes (AuthenticationError) | Partial |
| bedrock | No | — | — | — | — | — | — | Yes (own `_map_error`) | **No** |
| cloudflare | No | — | — | — | — | — | — | Yes (own `_handle_error_response`) | **No** |
| together_ai | No | — | — | — | — | — | — | Yes (wrapping) | **No** |
| fireworks_ai | No | — | — | — | — | — | — | Yes (wrapping) | **No** |
| deepseek | No | — | — | — | — | — | — | Yes (wrapping) | **No** |
| perplexity | No | — | — | — | — | — | — | Yes (wrapping) | **No** |
| deepinfra | No | — | — | — | — | — | — | Yes (wrapping) | **No** |
| gemini | No | — | — | — | — | — | — | Yes (wrapping) | **No** |
| cohere | No | — | — | — | — | — | — | Yes (wrapping) | **No** |
| ollama | No | — | — | — | — | — | — | Yes (wrapping) | **No** |

## Streaming Usage Behavior (verified 2026-03-13)

Provider streaming usage is **not uniform** — three distinct patterns exist, all verified at chunk level in `tests/test_live.py`.

### Pattern 1: OpenAI-compat opt-in (openai, groq, mistral, xai, openrouter)

- **Without** `stream_options={"include_usage": True}`: chunks carry **no** usage data. `stream_chunk_builder` returns zeros.
- **With** `include_usage`: a single final chunk carries all usage fields. `stream_chunk_builder` extracts them directly.

### Pattern 2: Anthropic native split

Anthropic sends usage across **two** chunks without opt-in:
- `message_start` chunk: `prompt_tokens > 0, completion_tokens = 0`
- `message_delta` chunk: `prompt_tokens = 0, completion_tokens > 0`

`stream_chunk_builder` accumulates via `max()` per field, correctly combining split values.

### Pattern 3: Reasoning model inflation (xai/grok-3-mini-fast)

xAI reasoning models report `total_tokens` that **exceeds** `prompt_tokens + completion_tokens` (e.g., total=283, prompt=15, completion=1). The difference is reasoning/thinking tokens not broken out in the standard fields. `stream_chunk_builder` preserves the API's `total_tokens` via `max(existing, chunk_total, computed)` rather than recomputing.

### Accumulation logic (`_completion.py` stream_chunk_builder)

```
usage.prompt_tokens = max(existing, chunk.prompt_tokens)
usage.completion_tokens = max(existing, chunk.completion_tokens)
usage.total_tokens = max(existing, chunk.total_tokens, prompt + completion)
```

`max()` handles both single-chunk (OpenAI-compat) and split-chunk (Anthropic) patterns. The three-way max on `total_tokens` preserves inflated totals from reasoning models.

## Tool Call Behavior (verified 2026-03-13)

All 6 providers verified with live API calls — non-streaming and streaming tool calls, using `tool_choice="required"` and a simple `get_weather` tool. 13 tests in `tests/test_live.py -k tool_call`.

### Response shape (DSPy contract)

All providers return: `finish_reason="tool_calls"`, `tool_calls[0].type="function"`, `.id` (non-empty string), `.function.name` (exact match), `.function.arguments` (JSON string, parseable to dict).

### Provider quirks

- **Groq**: `llama-3.1-8b-instant` fails `tool_choice="required"` (generates malformed tool XML). Tests use `llama-3.3-70b-versatile`.
- **Mistral**: Returns `type=None` on tool calls. Fixed in `_mistral.py:_fix_response()` — normalizes to `"function"`.
- **OpenRouter**: `meta-llama/llama-3.1-8b-instruct` unreliable for tools. Tests use `openai/gpt-4o-mini`.
- **xAI grok-3-mini-fast**: Works correctly with tools. Includes `reasoning_content` in response but tool call shape is standard.
- **Anthropic**: Full bidirectional pipeline verified — `_translate_tools` → API → `tool_use` blocks → `ChatCompletionMessageToolCall`. `stop_reason="tool_use"` → `finish_reason="tool_calls"`.

### Streaming tool call reassembly

`stream_chunk_builder` correctly reassembles tool calls from all 6 providers. Streaming does **not** include usage without `stream_options={"include_usage": True}` (same as text streaming).

Anthropic streaming tool calls inspected at chunk level:
- `content_block_start` → `ChoiceDeltaToolCall(id=, function.name=)`
- `content_block_delta` (input_json_delta) → argument fragments accumulate
- Minimum 2 chunks with `delta.tool_calls` (start + deltas)

## Embedding Behavior (verified 2026-03-13)

`embedding()` returns `EmbeddingResponse` wrapping the raw SDK response. `EmbeddingResponse.data` contains `_EmbeddingItem` wrappers that support both dict-access (`data[0]["embedding"]`) and attribute-access (`data[0].embedding`). Usage delegates to the raw response (`response.usage.prompt_tokens`).

**Providers verified:** OpenAI (`text-embedding-3-small`, 1536 dims), Mistral (`mistral-embed`, via OpenAI-compat path — no Mistral embedding handler exists, falls through to `get_sync_client`).

**Known limitation:** Mock tests in `test_embedding.py` return raw dicts, which pass through `EmbeddingResponse` as plain dicts (not `_EmbeddingItem`). Dict-access works natively on dicts, so the tests pass but don't exercise the `_EmbeddingItem` wrapper. Live tests are the real proof.

## ContextWindowExceededError (verified 2026-03-13)

DSPy's only explicit exception catch. Triggered by keyword matching on error messages.

### OpenAI (true context window error)

Send `"word " * 200000` (200,008 tokens) to `gpt-4o-mini` (128k context). API returns:
```
"This model's maximum context length is 128000 tokens. However, your messages resulted in 200008 tokens."
```
Keywords matched: "context" ✓, "token" ✓, "length" ✓. Correctly mapped to `ContextWindowExceededError`.

### Anthropic (max-output-tokens error, not context window)

Cannot trigger a true context-window error: haiku has 200k context but 100k/min rate limit — rate limit fires first. Anthropic SDK also rejects `max_tokens > ~21k` with a client-side `ValueError` (timeout estimation).

Workaround: send `max_tokens=20000` (haiku max output is 4096). Anthropic API returns:
```
"max_tokens: 20000 > 4096, which is the maximum allowed number of output tokens for claude-3-haiku-20240307"
```
Keyword matched: "token" ✓ (from "max_tokens"). **Semantically this is a max-output-tokens error**, not context-window. But litellm uses the same keyword matching, and DSPy only catches this to retry with shorter prompts. Accepted as compatible behavior.

### Keyword matching logic (`_completion.py:_wrap_context_window_error` + `_anthropic.py:_map_error`)

Both check `msg.lower()` for: "context", "token", "length", "too long". Any match → `ContextWindowExceededError`. This is intentionally broad to catch provider-specific phrasings, with the trade-off of false positives on unrelated "token" errors (like max-output-tokens).

## text_completion() (verified 2026-03-13)

Legacy completions API in `_text_completion.py`. DSPy calls with `text-completion-openai/` prefix.

- `_strip_text_prefix`: `text-completion-openai/gpt-3.5-turbo-instruct` → `openai/gpt-3.5-turbo-instruct`
- Reuses `_prepare_call` from `_completion.py` for kwarg stripping
- `TextCompletionResponse` wrapper: delegates to raw SDK `Completion` via `__getattr__`, adds `cache_hit` and `_hidden_params`
- Supports `mock_response` and `timeout` kwargs (same pattern as `completion()`)
- Mock uses real `openai.types.Completion` object (not custom types)
- Both sync and async verified live against `gpt-3.5-turbo-instruct` (still available as of 2026-03-13)

DSPy access pattern verified: `response.choices[0].text`, `.finish_reason`, `response.usage.prompt_tokens`, `response.usage.completion_tokens`.

## responses() / aresponses() (`_responses.py`)

OpenAI Responses API wrapper. Params: `model`, `input=`, `previous_response_id=`, `cache=`, `num_retries=`, `retry_strategy=`, `headers=`, `shared_session=`.

- Strips `caching` kwarg
- Passes `input` and `previous_response_id` through to SDK kwargs
- Dispatches to custom handler if available (`handler.responses()`)
- Otherwise uses `client.responses.create()` via OpenAI SDK
- Returns raw SDK response (no `ModelResponse` wrapping — Responses API has its own format)

## Known Gaps

GH issues #1-#10 all closed. Gap analysis bugs (Azure cache key, Bedrock client leak, SDK exception leaking, missing `model_dump()`) all fixed 2026-03-13.

**Remaining actionable:** None. Error mappings for NotFoundError/PermissionDeniedError/UnprocessableEntityError added to all 4 handlers (completion, bedrock, anthropic, cloudflare). `get_llm_provider()` exported as thin wrapper around `parse_model()`. 129 own tests pass.

**Low priority:** `text_completion` mock path depends on `openai.types.Completion` (openai SDK effectively required).

## Releasing

CI-triggered via GitHub Actions (`.github/workflows/publish.yml`). Uses PyPI trusted publishing (OIDC, no token needed).

**Steps:**
1. Bump version in `pyproject.toml` + `litelm/__init__.py`
2. Commit, push
3. `gh release create v0.X.0 --title "v0.X.0" --notes "..."`
4. CI builds and publishes to PyPI automatically

**Do NOT `uv publish` manually** — it races with CI and causes `400 Bad Request` on duplicate uploads.

Tests run on every push/PR via `.github/workflows/test.yml` (Python 3.10–3.13).

## API Keys (in `.env.test`)

| Provider | Env Var | Status |
|----------|---------|--------|
| OpenAI | `OPENAI_API_KEY` | **Live** (verified 2026-03-13) |
| Mistral | `MISTRAL_API_KEY` | **Live** (verified 2026-03-13) |
| Groq | `GROQ_API_KEY` | **Live** (verified 2026-03-13) |
| Anthropic | `ANTHROPIC_API_KEY` | **Live** (verified 2026-03-13) |
| xAI | `XAI_API_KEY` | **Live** (verified 2026-03-13) |
| OpenRouter | `OPENROUTER_API_KEY` | **Live** (verified 2026-03-13) |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_URL` + `AZURE_OPENAI_MODEL` | **Live** (verified 2026-04-17, `gpt-5.4-nano` on aedna-foundry) |

Load with `source .env.test` before running integration tests. Use cheapest models (gpt-4o-mini, claude-3-haiku, llama-3.1-8b, etc.) to minimize cost.

Azure env var format: `AZURE_OPENAI_URL` is parsed by `_azure_env()` in `tests/test_live.py` to extract the base endpoint and `api-version` query param. Our code also accepts the canonical `AZURE_API_BASE` / `AZURE_API_KEY` / `AZURE_API_VERSION` env var names via `_providers.parse_model()`.

## Pre-push Checklist

Every push must pass these locally first. No exceptions.

```bash
uvx ruff check litelm/ tests/ --exclude tests/ported          # lint
uvx ruff format --check litelm/ tests/ --exclude tests/ported  # format
uv run pytest tests/ --ignore=tests/ported --timeout=10 -q     # tests
```

After pushing, verify CI is green: `gh run list --limit 1`. If red, fix immediately before doing anything else.

## Gotchas

- `uv run pytest` not `pytest` — pytest isn't on PATH, must go through uv
- **Ported tests not in git**: `tests/ported/` is gitignored. Run `bash scripts/sync_litellm_tests.sh` to populate. Our `conftest.py` is committed separately and restored by the sync script.
- **conftest exclusions**: `tests/ported/conftest.py` uses `pytest_ignore_collect` hook to exclude ~36 dirs and ~25 files by basename at any nesting depth. Add dirs to `_EXCLUDE_DIRS` / `_EXCLUDE_FILES` as needed.
- Many litellm tests import from deep submodules (`litellm.llms.anthropic`, `litellm.proxy.utils`, etc) that don't exist in litelm. These fail at collection time.
- `litelm.exceptions`, `litelm.types`, `litelm.responses` — thin re-export modules exist and work (wildcard re-exports of `_exceptions`, `_types`, `_responses`)
