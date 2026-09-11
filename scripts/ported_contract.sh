#!/usr/bin/env bash
# Fast upstream compatibility gate for litelm's explicit public contract.
# Prereq: bash scripts/sync_litellm_tests.sh
set -euo pipefail

if [ ! -d "tests/ported/test_litellm" ]; then
    echo "No ported tests found. Run: bash scripts/sync_litellm_tests.sh" >&2
    exit 1
fi

NODES=()
while IFS= read -r node; do
    NODES+=("$node")
done < <(uv run python scripts/ported_contract.py)

echo "=== Running ${#NODES[@]} public-contract nodes ==="
uv run pytest -p tests.ported.conftest --noconftest "${NODES[@]}" --tb=short -q --timeout=10 "$@"
