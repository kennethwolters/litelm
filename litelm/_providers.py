"""Provider registry: maps model prefixes to base URLs and API key env vars."""

import os

# (base_url, api_key_env_var)
PROVIDERS = {
    # --- OpenAI-native ---
    "openai": (None, "OPENAI_API_KEY"),
    "azure": (None, "AZURE_API_KEY"),  # special-cased in parse_model
    # --- Drop-in OpenAI-compatible ---
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together_ai": ("https://api.together.xyz/v1", "TOGETHERAI_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHERAI_API_KEY"),
    "fireworks_ai": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "perplexity": ("https://api.perplexity.ai", "PERPLEXITYAI_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "deepinfra": ("https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_TOKEN"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "cohere": ("https://api.cohere.ai/compatibility/v1", "COHERE_API_KEY"),
    # --- Native SDK (dispatch handles routing, registry provides api_key env) ---
    "anthropic": ("https://api.anthropic.com", "ANTHROPIC_API_KEY"),
    "bedrock": (None, None),  # auth via boto3
    "cloudflare": (None, "CLOUDFLARE_API_TOKEN"),
    # --- Self-hosted (default localhost, override with api_base) ---
    "ollama": ("http://localhost:11434/v1", "OLLAMA_API_KEY"),
    "vllm": ("http://localhost:8000/v1", "VLLM_API_KEY"),
    "lm_studio": ("http://localhost:1234/v1", "LM_STUDIO_API_KEY"),
    "lmstudio": ("http://localhost:1234/v1", "LM_STUDIO_API_KEY"),
}


def parse_model(model_str, **kwargs):
    """Parse a model string into (provider, model_name, base_url, api_key).

    Supports formats:
    - "openai/gpt-4o" -> ("openai", "gpt-4o", None, <OPENAI_API_KEY>)
    - "gpt-4o" -> ("openai", "gpt-4o", None, <OPENAI_API_KEY>)
    - "azure/gpt-4o" -> ("azure", "gpt-4o", <AZURE_API_BASE>, <AZURE_API_KEY>)
    - "ollama/llama3" -> ("ollama", "llama3", "http://localhost:11434/v1", None)

    kwargs `api_base` and `api_key` override registry values.
    """
    if "/" in model_str:
        provider, model_name = model_str.split("/", 1)
    else:
        provider, model_name = "openai", model_str

    api_key = kwargs.get("api_key")
    api_base = kwargs.get("api_base") or kwargs.get("base_url")

    if provider == "azure":
        api_base = api_base or os.environ.get("AZURE_API_BASE")
        api_key = api_key or os.environ.get("AZURE_API_KEY")
        api_version = kwargs.get("api_version") or os.environ.get("AZURE_API_VERSION", "2024-02-01")
        return provider, model_name, api_base, api_key, api_version

    if provider in PROVIDERS:
        base_url, key_env = PROVIDERS[provider]
        api_base = api_base or base_url
        api_key = api_key or os.environ.get(key_env)
    else:
        # Unknown provider: treat as OpenAI-compatible with custom base_url
        api_key = api_key or os.environ.get("OPENAI_API_KEY")

    return provider, model_name, api_base, api_key, None
