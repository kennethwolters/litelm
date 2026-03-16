#!/usr/bin/env bash
# Full ported-test experiment pipeline: sync, run, categorize, diff.
# Usage: bash scripts/ported_experiment.sh [--skip-sync]
set -euo pipefail

SKIP_SYNC=false
[[ "${1:-}" == "--skip-sync" ]] && SKIP_SYNC=true

RESULTS_DIR="/tmp/litelm_experiment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo "Results dir: $RESULTS_DIR"

# Step 0: Save previous baseline
OLD_XML=/tmp/litelm_results.xml
if [ -f "$OLD_XML" ]; then
    cp "$OLD_XML" "$RESULTS_DIR/old_baseline.xml"
    echo "Saved previous baseline XML"
else
    echo "No previous baseline found at $OLD_XML"
fi
if [ -f "tests/ported/.litellm-commit" ]; then
    echo "old=$(cat tests/ported/.litellm-commit)" > "$RESULTS_DIR/litellm_commits.txt"
fi

# Step 1: Sync
if [ "$SKIP_SYNC" = false ]; then
    echo "=== Step 1: Syncing from litellm HEAD ==="
    bash scripts/sync_litellm_tests.sh
else
    echo "=== Step 1: Skipping sync (--skip-sync) ==="
fi
if [ -f "tests/ported/.litellm-commit" ]; then
    echo "new=$(cat tests/ported/.litellm-commit)" >> "$RESULTS_DIR/litellm_commits.txt"
    echo "Synced from: $(cat tests/ported/.litellm-commit)"
fi

# Step 2: Discovery
echo "=== Step 2: Discovery ==="
echo "Test files per directory:" | tee "$RESULTS_DIR/test_file_counts_by_dir.txt"
find tests/ported -name 'test_*.py' -o -name '*_test.py' | sed 's|/[^/]*$||' | sort | uniq -c | sort -rn | tee -a "$RESULTS_DIR/test_file_counts_by_dir.txt"
echo ""
echo "Collection count (dry run):"
uv run pytest tests/ported/ --collect-only -q --continue-on-collection-errors 2>&1 | tail -5 | tee "$RESULTS_DIR/collection_summary.txt" || true

# Step 3: Run
echo ""
echo "=== Step 3: Running ported tests ==="
NEW_XML="$RESULTS_DIR/new_baseline.xml"
uv run pytest tests/ported/ --tb=line -q --timeout=10 --continue-on-collection-errors --junit-xml="$NEW_XML" 2>&1 | tee "$RESULTS_DIR/pytest_output.txt" || true
# Also copy to standard location for future runs
cp "$NEW_XML" "$OLD_XML"

# Step 4: Categorize
echo ""
echo "=== Step 4: Categorize ==="
uv run python scripts/categorize_failures.py "$NEW_XML" | tee "$RESULTS_DIR/categorized.txt"

# Step 5: Diff
echo ""
echo "=== Step 5: Diff ==="
if [ -f "$RESULTS_DIR/old_baseline.xml" ]; then
    uv run python scripts/categorize_failures.py --diff "$RESULTS_DIR/old_baseline.xml" "$NEW_XML" "$RESULTS_DIR"
else
    echo "No old baseline to diff against. First run — skipping diff."
fi

# Step 6: Summary
echo ""
echo "=== Done ==="
echo "Results: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
