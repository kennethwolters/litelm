#!/usr/bin/env bash
# Sync litellm's test suite into tests/ported/ (unmodified).
# Uses shallow clone to minimize download. Run periodically to track upstream.
set -euo pipefail

REPO="https://github.com/BerriAI/litellm.git"
TARGET="tests/ported"
TMPDIR=$(mktemp -d)

echo "=== Shallow-cloning litellm tests ==="
git clone --depth=1 --filter=blob:none --sparse "$REPO" "$TMPDIR/litellm"
cd "$TMPDIR/litellm"
git sparse-checkout set tests/
cd -

echo "=== Copying tests to $TARGET ==="
# Preserve our conftest.py (sys.modules shim)
CONFTEST=""
if [ -f "$TARGET/conftest.py" ]; then
    CONFTEST=$(cat "$TARGET/conftest.py")
fi
rm -rf "$TARGET"
cp -r "$TMPDIR/litellm/tests" "$TARGET"
# Restore our conftest over litellm's
if [ -n "$CONFTEST" ]; then
    echo "$CONFTEST" > "$TARGET/conftest.py"
fi

# Record which commit we synced from
COMMIT=$(git -C "$TMPDIR/litellm" rev-parse HEAD)
echo "$COMMIT" > "$TARGET/.litellm-commit"
echo "=== Synced from litellm commit $COMMIT ==="

rm -rf "$TMPDIR"
echo "=== Done. Run: bash scripts/ported_baseline.sh ==="
