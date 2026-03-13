#!/usr/bin/env bash
# Run in-scope ported tests, produce JUnit XML, print categorized summary.
set -euo pipefail
XML=/tmp/litelm_results.xml
echo "=== Running ported tests ==="
uv run pytest tests/ported/ --tb=line -q --timeout=10 --continue-on-collection-errors --junit-xml="$XML" || true
echo ""
echo "=== Categorized summary ==="
uv run python scripts/categorize_failures.py "$XML"
