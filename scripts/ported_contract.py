#!/usr/bin/env python3
"""Validate and print the explicit upstream public-contract test allowlist."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = Path(__file__).with_name("ported_contract_tests.txt")


def load_contract_nodes(manifest: Path, *, root: Path = REPO_ROOT) -> list[str]:
    """Return validated pytest node IDs from *manifest*."""
    nodes: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(manifest.read_text().splitlines(), 1):
        node = raw_line.strip()
        if not node or node.startswith("#"):
            continue
        path_text, separator, test_name = node.partition("::")
        path = Path(path_text)
        if (
            not separator
            or not test_name
            or path.is_absolute()
            or ".." in path.parts
            or path.parts[:2] != ("tests", "ported")
            or path.suffix != ".py"
        ):
            raise ValueError(f"invalid contract node at {manifest}:{line_number}: {node}")
        if node in seen:
            raise ValueError(f"duplicate contract node at {manifest}:{line_number}: {node}")
        if not (root / path).is_file():
            raise FileNotFoundError(f"upstream contract test file was removed: {path}")
        seen.add(node)
        nodes.append(node)
    if not nodes:
        raise ValueError(f"contract manifest is empty: {manifest}")
    return nodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    try:
        nodes = load_contract_nodes(args.manifest, root=args.root)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print("\n".join(nodes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
