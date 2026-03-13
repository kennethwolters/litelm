import os
import pytest
from pathlib import Path


def _load_env_test():
    """Load .env.test into os.environ (no python-dotenv dep)."""
    env_file = Path(__file__).parent.parent / ".env.test"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env_test()


def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires real API keys (skipped by default)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("-m", default=None) or "live" not in config.getoption("-m", default=""):
        skip_live = pytest.mark.skip(reason="live tests require -m live")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)
