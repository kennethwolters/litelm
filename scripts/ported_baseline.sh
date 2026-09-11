#!/usr/bin/env bash
# Run in-scope ported tests, produce JUnit XML, print categorized summary.
# Prereq: bash scripts/sync_litellm_tests.sh
set -euo pipefail

if [ ! -d "tests/ported/test_litellm" ]; then
    echo "No ported tests found. Run: bash scripts/sync_litellm_tests.sh"
    exit 1
fi

# Upstream now has nested conftest.py files that import LiteLLM internals before
# collection filtering runs. Load only our shim as a plugin and disable upstream
# conftest auto-loading so collection errors remain categorizable.
PYTEST_PORTED_ARGS=(-p tests.ported.conftest --noconftest tests/ported/)

XML=/tmp/litelm_results.xml
echo "=== Running ported tests ==="
uv run pytest "${PYTEST_PORTED_ARGS[@]}" --tb=line -q --timeout=10 --continue-on-collection-errors --junit-xml="$XML" || true
echo ""
echo "=== Categorized summary ==="
uv run python scripts/categorize_failures.py "$XML"
