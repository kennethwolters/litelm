"""Collection-time exclusions for ported tests.

Skip directories testing features litelm will never implement
(proxy, router, caching, guardrails, agents, audio, images, etc).
"""

# ── Dead top-level dirs ──────────────────────────────────────────────
_dead_top = [
    "proxy_unit_tests",
    "old_proxy_tests",
    "proxy_admin_ui_tests",
    "proxy_e2e_*",
    "proxy_security_tests",
    "basic_proxy_startup_tests",
    "router_unit_tests",
    "agent_tests",
    "audio_tests",
    "batches_tests",
    "guardrails_tests",
    "image_gen_tests",
    "ocr_tests",
    "otel_tests",
    "logging_callback_tests",
    "search_tests",
    "vector_store_tests",
    "enterprise",
    "scim_tests",
    "multi_instance_e2e_tests",
    "spend_tracking_tests",
    "store_model_in_db_tests",
    "mcp_tests",
    "code_coverage_tests",
    "documentation_tests",
    "load_tests",
    "windows_tests",
    "litellm-proxy-extras",
    "pass_through_unit_tests",
    "pass_through_tests",
    "openai_endpoints_tests",
    "unified_google_tests",
    "litellm_utils_tests",
    "litellm_core_utils",
    "litellm",
    "llm_translation",
    "llm_responses_api_testing",
    "local_testing",
]

# ── Dead top-level test files (proxy/team/otel/spend/passthrough) ──
_dead_top_files = [
    "test_budget_management.py",
    "test_callbacks_on_proxy.py",
    "test_config.py",
    "test_end_users.py",
    "test_entrypoint.py",
    "test_health.py",
    "test_keys.py",
    "test_litellm_proxy_responses_config.py",
    "test_models.py",
    "test_new_vector_store_endpoints.py",
    "test_openai_endpoints.py",
    "test_organizations.py",
    "test_otel_thread_leak.py",
    "test_passthrough_endpoints.py",
    "test_presidio_latency.py",
    "test_proxy_server_non_root.py",
    "test_ratelimit.py",
    "test_service_logger_otel.py",
    "test_spend_logs.py",
    "test_team_logging.py",
    "test_team_members.py",
    "test_team.py",
    "test_users.py",
]

# ── Dead subdirs within test_litellm/ ────────────────────────────────
_dead_test_litellm = [
    "test_litellm/proxy",
    "test_litellm/enterprise",
    "test_litellm/integrations",
    "test_litellm/caching",
    "test_litellm/secret_managers",
    "test_litellm/router_strategy",
    "test_litellm/router_utils",
    "test_litellm/test_router",
    "test_litellm/vector_stores",
    "test_litellm/containers",
    "test_litellm/a2a_protocol",
    "test_litellm/experimental_mcp_client",
    "test_litellm/ocr",
    "test_litellm/passthrough",
    "test_litellm/interactions",
    "test_litellm/images",
    "test_litellm/google_genai",
    "test_litellm/litellm_core_utils",
]

# ── Dead provider dirs within test_litellm/llms/ ─────────────────────
# Keep only providers litelm actually supports (from PROVIDERS dict + README).
# Incrementally whitelist as support is added.
_keep_llms = {
    "openai", "anthropic", "azure", "bedrock", "mistral", "cloudflare",
    "chat", "custom_httpx", "base_llm", "openai_like", "gemini",
    "fireworks_ai", "deepinfra", "cohere", "openrouter", "perplexity",
    "xai", "ollama", "lm_studio", "sambanova",
}

import os as _os

_llms_dir = _os.path.join(_os.path.dirname(__file__), "test_litellm", "llms")
_dead_llms = []
if _os.path.isdir(_llms_dir):
    for d in _os.listdir(_llms_dir):
        if _os.path.isdir(_os.path.join(_llms_dir, d)) and d not in _keep_llms and not d.startswith("__"):
            _dead_llms.append(f"test_litellm/llms/{d}")

# ── Assemble collect_ignore_glob ─────────────────────────────────────
collect_ignore_glob = (
    [f"{d}/*" for d in _dead_top]
    + _dead_top_files
    + [f"{d}/*" for d in _dead_test_litellm]
    + [f"{d}/*" for d in _dead_llms]
)
