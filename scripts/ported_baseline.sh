#!/usr/bin/env bash
# Run in-scope ported tests, produce JUnit XML, print categorized summary.
# Prereq: bash scripts/sync_litellm_tests.sh
set -euo pipefail

if [ ! -d "tests/ported/test_litellm" ]; then
    echo "No ported tests found. Run: bash scripts/sync_litellm_tests.sh"
    exit 1
fi

XML=/tmp/litelm_results.xml
echo "=== Running ported tests ==="
uv run pytest tests/ported/ --tb=line -q --timeout=10 --continue-on-collection-errors --junit-xml="$XML" || true
echo ""
echo "=== Categorized summary ==="
uv run python scripts/categorize_failures.py "$XML"
