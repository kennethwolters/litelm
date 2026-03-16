"""Conftest for ported litellm tests — runtime sys.modules shimming.

Instead of sed-replacing 'litellm' → 'litelm' in test files, we shim
sys.modules so that `import litellm` resolves to litelm at runtime.
This keeps litellm's test files unmodified and avoids string corruption.
"""

import sys

import litelm
import litelm._logging
import litelm._types
import litelm._exceptions
import litelm._completion
import litelm._embedding
import litelm._text_completion
import litelm._responses
import litelm.types
import litelm.exceptions
import litelm.responses

# ---------------------------------------------------------------------------
# sys.modules shim: litellm → litelm
# ---------------------------------------------------------------------------

# Top-level module
sys.modules["litellm"] = litelm

# Submodules that exist in litelm
_SUBMODULE_MAP = {
    "litellm.types": litelm.types,
    "litellm.exceptions": litelm.exceptions,
    "litellm.responses": litelm.responses,
    "litellm._logging": litelm._logging,
    "litellm._types": litelm._types,
    "litellm._exceptions": litelm._exceptions,
    "litellm._completion": litelm._completion,
    "litellm._embedding": litelm._embedding,
    "litellm._text_completion": litelm._text_completion,
    "litellm._responses": litelm._responses,
}

# Lazy-load provider submodules only if available
for _litellm_path, _litelm_path in [
    ("litellm._providers", "litelm._providers"),
    ("litellm._dispatch", "litelm._dispatch"),
    ("litellm._client_cache", "litelm._client_cache"),
    ("litellm.providers", "litelm.providers"),
    ("litellm.providers._anthropic", "litelm.providers._anthropic"),
    ("litellm.providers._bedrock", "litelm.providers._bedrock"),
    ("litellm.providers._cloudflare", "litelm.providers._cloudflare"),
    ("litellm.providers._mistral", "litelm.providers._mistral"),
]:
    try:
        _mod = __import__(_litelm_path, fromlist=[""])
        _SUBMODULE_MAP[_litellm_path] = _mod
    except ImportError:
        pass

for _k, _v in _SUBMODULE_MAP.items():
    sys.modules[_k] = _v

# ---------------------------------------------------------------------------
# Exclude test dirs that exercise features litelm doesn't implement
# ---------------------------------------------------------------------------

# Match actual litellm test dir names
_EXCLUDE_DIRS = [
    # Proxy (multiple dirs)
    "basic_proxy_startup_tests",
    "old_proxy_tests",
    "proxy_admin_ui_tests",
    "proxy_e2e_anthropic_messages_tests",
    "proxy_e2e_azure_batches_tests",
    "proxy_security_tests",
    "proxy_unit_tests",
    "litellm-proxy-extras",
    # Router
    "router_unit_tests",
    # Agents
    "agent_tests",
    "mcp_tests",
    # Audio / image / OCR
    "audio_tests",
    "image_gen_tests",
    "ocr_tests",
    # Batch / fine-tuning / assistants
    "batches_tests",
    # Caching / budget / guardrails
    "spend_tracking_tests",
    "guardrails_tests",
    # Vector stores / search
    "vector_store_tests",
    "search_tests",
    # Pass-through
    "pass_through_tests",
    "pass_through_unit_tests",
    # Logging / observability
    "logging_callback_tests",
    "otel_tests",
    # Infra / config
    "load_tests",
    "multi_instance_e2e_tests",
    "store_model_in_db_tests",
    "enterprise",
    "scim_tests",
    "documentation_tests",
    "code_coverage_tests",
    "windows_tests",
    # Responses API (separate from our _responses)
    "llm_responses_api_testing",
    # Google-specific
    "unified_google_tests",
    # Internal litellm utils (not our code)
    "litellm_core_utils",
    "litellm_utils_tests",
    # Upstream restructure (2026-03-16) — nested under test_litellm/
    "integrations",            # opik, datadog, arize, cloudzero, SlackAlerting
    "interactions",            # Google Interactions API
    "anthropic_interface",     # litellm.anthropic_interface deep module
    "google_genai",            # Google GenAI specific
    "a2a_protocol",            # a2a protocol
    "experimental_mcp_client", # MCP client
]

# Also exclude root-level proxy/config test files
_EXCLUDE_FILES = [
    "test_budget_management.py",
    "test_callbacks_on_proxy.py",
    "test_config.py",
    "test_end_users.py",
    "test_entrypoint.py",
    "test_fallbacks.py",
    "test_health.py",
    "test_keys.py",
    "test_models.py",
    "test_new_vector_store_endpoints.py",
    "test_openai_endpoints.py",
    "test_organizations.py",
    "test_otel_thread_leak.py",
    "test_passthrough_endpoints.py",
    "test_presidio_latency.py",
    "test_proxy_server_non_root.py",
    "test_ratelimit.py",
    "test_resource_cleanup.py",
    "test_service_logger_otel.py",
    "test_spend_logs.py",
    "test_team_logging.py",
    "test_team_members.py",
    "test_team.py",
    "test_users.py",
    "test_litellm_proxy_responses_config.py",
    "test_debug_warning.py",
    "test_default_encoding_non_root.py",
    "test_gpt5_azure_temperature_support.py",
]

_EXCLUDE_DIRS_SET = frozenset(_EXCLUDE_DIRS)
_EXCLUDE_FILES_SET = frozenset(_EXCLUDE_FILES)

def pytest_ignore_collect(collection_path, config):
    """Skip dirs/files by basename at any nesting depth."""
    name = collection_path.name
    if name in _EXCLUDE_FILES_SET:
        return True
    if collection_path.is_dir() and name in _EXCLUDE_DIRS_SET:
        return True
    return None
