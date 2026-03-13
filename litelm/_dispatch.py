"""Lazy-load registry for providers that need native SDK translation."""

CUSTOM_HANDLERS = {
    "anthropic": "litelm.providers._anthropic",
    "bedrock": "litelm.providers._bedrock",
    "cloudflare": "litelm.providers._cloudflare",
    "mistral": "litelm.providers._mistral",
}

_loaded = {}


def get_handler(provider):
    """Return the handler module for a custom provider, or None for OpenAI-compat."""
    if provider not in CUSTOM_HANDLERS:
        return None
    if provider not in _loaded:
        import importlib
        _loaded[provider] = importlib.import_module(CUSTOM_HANDLERS[provider])
    return _loaded[provider]
