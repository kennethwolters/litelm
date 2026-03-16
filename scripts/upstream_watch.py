#!/usr/bin/env python3
"""Upstream litellm drift detection — AST-based diffing of public API surface.

Clones litellm, extracts public API/types/exceptions/providers via AST parsing,
diffs against committed snapshots, and optionally opens a GitHub issue.

Usage:
    python scripts/upstream_watch.py                        # diff + print report
    python scripts/upstream_watch.py --snapshot-only        # bootstrap initial snapshots
    python scripts/upstream_watch.py --create-issue         # open GH issue if changes found
    python scripts/upstream_watch.py --create-issue --dry-run  # print issue body only
    python scripts/upstream_watch.py --local /tmp/litellm   # use existing clone
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT_DIR = Path(__file__).resolve().parent / "upstream_snapshots"
LITELM_ROOT = Path(__file__).resolve().parent.parent / "litelm"
REPO_URL = "https://github.com/BerriAI/litellm.git"


# ---------------------------------------------------------------------------
# AST extraction helpers
# ---------------------------------------------------------------------------


def _parse_file(path: Path) -> ast.Module | None:
    """Parse a Python file, returning None on missing/syntax errors."""
    try:
        return ast.parse(path.read_text())
    except (FileNotFoundError, SyntaxError, UnicodeDecodeError) as e:
        print(f"WARNING: could not parse {path}: {e}", file=sys.stderr)
        return None


def _extract_list_assign(tree: ast.Module, name: str) -> list[str]:
    """Extract a module-level list assignment like `__all__ = [...]`."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        return [
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]
    return []


def _extract_functions(tree: ast.Module) -> dict[str, list[str]]:
    """Extract top-level function signatures: {name: [param_names]}."""
    funcs = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = []
            for arg in node.args.args:
                params.append(arg.arg)
            if node.args.vararg:
                params.append(f"*{node.args.vararg.arg}")
            for arg in node.args.kwonlyargs:
                params.append(arg.arg)
            if node.args.kwarg:
                params.append(f"**{node.args.kwarg.arg}")
            funcs[node.name] = params
    return funcs


def _extract_classes(tree: ast.Module) -> dict[str, dict]:
    """Extract class definitions: {name: {bases: [...], fields: [...], status_code: int|None}}."""
    classes = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(ast.unparse(base))
            fields = []
            status_code = None
            for item in node.body:
                # Annotated fields (Pydantic models, dataclasses)
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    fields.append(item.target.id)
                # __slots__ = (...)
                if (
                    isinstance(item, ast.Assign)
                    and len(item.targets) == 1
                    and isinstance(item.targets[0], ast.Name)
                ):
                    tname = item.targets[0].id
                    if tname == "__slots__" and isinstance(item.value, (ast.Tuple, ast.List)):
                        fields = [
                            elt.value
                            for elt in item.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]
                    if tname == "_default_status_code" and isinstance(item.value, ast.Constant):
                        status_code = item.value.value
            info: dict = {"bases": bases, "fields": sorted(fields)}
            if status_code is not None:
                info["status_code"] = status_code
            classes[node.name] = info
    return classes


def _extract_dict_keys(tree: ast.Module, name: str) -> list[str]:
    """Extract keys from a module-level dict assignment like `PROVIDERS = {...}`."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    if isinstance(node.value, ast.Dict):
                        return [
                            k.value
                            for k in node.value.keys
                            if isinstance(k, ast.Constant) and isinstance(k.value, str)
                        ]
    return []


def _extract_enum_values(tree: ast.Module, class_name: str) -> list[str]:
    """Extract string values from a str Enum class (e.g. LlmProviders)."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            values = []
            for item in node.body:
                if isinstance(item, ast.Assign) and len(item.targets) == 1:
                    if isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                        values.append(item.value.value)
            return values
    return []


_SKIP_MODULES = frozenset({
    "typing", "__future__", "os", "re", "sys", "threading", "warnings",
    "dotenv", "pathlib",
})

_SKIP_NAMES = frozenset({
    # typing re-exports that aren't API
    "Callable", "Dict", "List", "Optional", "Union", "Any", "Literal",
    "Set", "Tuple", "Type", "get_args", "TYPE_CHECKING", "overload",
    "annotations",
    # star imports
    "*",
})


def _extract_imported_names(tree: ast.Module) -> list[str]:
    """Extract names imported via `from X import Y` at module level.

    Filters out typing, stdlib, and other non-API imports.
    """
    names = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            # Skip stdlib/typing modules
            if any(mod == skip or mod.startswith(skip + ".") for skip in _SKIP_MODULES):
                continue
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                if not name.startswith("_") and name not in _SKIP_NAMES:
                    names.append(name)
    return names


# ---------------------------------------------------------------------------
# Upstream extraction
# ---------------------------------------------------------------------------


def extract_upstream(litellm_dir: Path) -> dict[str, dict]:
    """Extract structured data from litellm source."""
    snapshots: dict[str, dict] = {}

    # API surface: functions from main.py (core API) + __init__.py (helpers)
    # litellm doesn't use __all__; core functions live in main.py, exposed via __getattr__
    functions: dict[str, list[str]] = {}
    main_tree = _parse_file(litellm_dir / "main.py")
    if main_tree:
        functions.update(_extract_functions(main_tree))
    init_tree = _parse_file(litellm_dir / "__init__.py")
    if init_tree:
        functions.update(_extract_functions(init_tree))

    # Collect exported names from __init__.py imports (litellm has no __all__)
    exports: list[str] = []
    if init_tree:
        exports = _extract_imported_names(init_tree)
        # Also add top-level function names from __init__.py
        for node in ast.iter_child_nodes(init_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    exports.append(node.name)
    snapshots["api_surface"] = {
        "exports": sorted(set(exports)),
        "functions": dict(sorted(functions.items())),
    }

    # Providers from LlmProviders enum in types/utils.py
    types_tree = _parse_file(litellm_dir / "types" / "utils.py")
    provider_list: list[str] = []
    if types_tree:
        provider_list = _extract_enum_values(types_tree, "LlmProviders")
        snapshots["types"] = {"classes": dict(sorted(_extract_classes(types_tree).items()))}
    else:
        snapshots["types"] = {"classes": {}}
    snapshots["providers"] = {"provider_list": sorted(provider_list)}

    # Exceptions from exceptions.py
    exc_tree = _parse_file(litellm_dir / "exceptions.py")
    if exc_tree:
        snapshots["exceptions"] = {"classes": dict(sorted(_extract_classes(exc_tree).items()))}
    else:
        snapshots["exceptions"] = {"classes": {}}

    return snapshots


def extract_ours() -> dict[str, dict]:
    """Extract our own surface for overlap detection."""
    result: dict[str, dict] = {}

    init_tree = _parse_file(LITELM_ROOT / "__init__.py")
    if init_tree:
        result["exports"] = set(_extract_list_assign(init_tree, "__all__"))
        result["functions"] = _extract_functions(init_tree)
    else:
        result["exports"] = set()
        result["functions"] = {}

    types_tree = _parse_file(LITELM_ROOT / "_types.py")
    result["type_classes"] = set(_extract_classes(types_tree).keys()) if types_tree else set()

    exc_tree = _parse_file(LITELM_ROOT / "_exceptions.py")
    result["exception_classes"] = set(_extract_classes(exc_tree).keys()) if exc_tree else set()

    prov_tree = _parse_file(LITELM_ROOT / "_providers.py")
    result["providers"] = set(_extract_dict_keys(prov_tree, "PROVIDERS")) if prov_tree else set()

    return result


# ---------------------------------------------------------------------------
# Snapshot I/O (atomic writes)
# ---------------------------------------------------------------------------


def load_snapshot(name: str) -> dict:
    path = SNAPSHOT_DIR / f"{name}.json"
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"WARNING: could not load {path}: {e}", file=sys.stderr)
        return {}


def write_snapshots(snapshots: dict[str, dict], meta: dict) -> None:
    """Write all snapshots atomically via tempdir rename."""
    tmp = Path(tempfile.mkdtemp(prefix="upstream_snap_"))
    try:
        for name, data in snapshots.items():
            (tmp / f"{name}.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        # Move files into real snapshot dir
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        for f in tmp.iterdir():
            shutil.move(str(f), str(SNAPSHOT_DIR / f.name))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------


def diff_list(old: list, new: list) -> tuple[list, list]:
    """Return (added, removed) between two sorted lists."""
    old_set, new_set = set(old), set(new)
    return sorted(new_set - old_set), sorted(old_set - new_set)


def diff_dict_keys(old: dict, new: dict) -> tuple[list, list]:
    """Return (added_keys, removed_keys)."""
    return sorted(set(new) - set(old)), sorted(set(old) - set(new))


def diff_snapshots(old: dict[str, dict], new: dict[str, dict], ours: dict) -> list[dict]:
    """Compute meaningful changes. Returns list of change dicts."""
    changes: list[dict] = []

    # --- API surface ---
    old_api = old.get("api_surface", {})
    new_api = new.get("api_surface", {})

    added_exports, removed_exports = diff_list(
        old_api.get("exports", []), new_api.get("exports", [])
    )
    for name in added_exports:
        if not name.startswith("_"):
            impact = "We export this" if name in ours["exports"] else "Not implemented"
            changes.append(
                {"category": "API Surface", "change": "ADDED", "name": name, "detail": "New public export", "impact": impact}
            )
    for name in removed_exports:
        if not name.startswith("_"):
            impact = "We export this — check if upstream renamed" if name in ours["exports"] else "Not implemented"
            changes.append(
                {"category": "API Surface", "change": "REMOVED", "name": name, "detail": "Removed from exports", "impact": impact}
            )

    # Function signature changes
    old_funcs = old_api.get("functions", {})
    new_funcs = new_api.get("functions", {})
    for fname in set(old_funcs) & set(new_funcs):
        if old_funcs[fname] != new_funcs[fname]:
            old_params = set(old_funcs[fname])
            new_params = set(new_funcs[fname])
            added_p = sorted(new_params - old_params)
            removed_p = sorted(old_params - new_params)
            parts = []
            if added_p:
                parts.append("+" + ", ".join(f"`{p}`" for p in added_p))
            if removed_p:
                parts.append("-" + ", ".join(f"`{p}`" for p in removed_p))
            detail = " ".join(parts)
            impact = "May need passthrough" if fname in ours["functions"] else "Not implemented"
            changes.append(
                {"category": "API Surface", "change": "CHANGED", "name": f"`{fname}()`", "detail": detail, "impact": impact}
            )

    # --- Types ---
    old_types = old.get("types", {}).get("classes", {})
    new_types = new.get("types", {}).get("classes", {})
    added_cls, removed_cls = diff_dict_keys(old_types, new_types)
    for cls in added_cls:
        changes.append(
            {"category": "Types", "change": "ADDED", "name": cls, "detail": f"bases: {new_types[cls].get('bases', [])}", "impact": "New class"}
        )
    for cls in removed_cls:
        impact = "We implement this" if cls in ours["type_classes"] else "Not implemented"
        changes.append(
            {"category": "Types", "change": "REMOVED", "name": cls, "detail": "", "impact": impact}
        )
    # Field changes on classes we both have
    for cls in set(old_types) & set(new_types):
        old_fields = set(old_types[cls].get("fields", []))
        new_fields = set(new_types[cls].get("fields", []))
        added_f = sorted(new_fields - old_fields)
        removed_f = sorted(old_fields - new_fields)
        if added_f or removed_f:
            parts = []
            if added_f:
                parts.append("+" + ", ".join(f"`{f}`" for f in added_f))
            if removed_f:
                parts.append("-" + ", ".join(f"`{f}`" for f in removed_f))
            impact = "Check our implementation" if cls in ours["type_classes"] else "Not implemented"
            changes.append(
                {"category": "Types", "change": "CHANGED", "name": cls, "detail": " ".join(parts), "impact": impact}
            )

    # --- Exceptions ---
    old_exc = old.get("exceptions", {}).get("classes", {})
    new_exc = new.get("exceptions", {}).get("classes", {})
    added_exc, removed_exc = diff_dict_keys(old_exc, new_exc)
    for cls in added_exc:
        base = new_exc[cls].get("bases", [])
        changes.append(
            {"category": "Exceptions", "change": "ADDED", "name": cls, "detail": f"bases: {base}", "impact": "Consumers may catch this"}
        )
    for cls in removed_exc:
        impact = "We implement this" if cls in ours["exception_classes"] else "Not implemented"
        changes.append(
            {"category": "Exceptions", "change": "REMOVED", "name": cls, "detail": "", "impact": impact}
        )
    # Hierarchy changes on shared exceptions
    for cls in set(old_exc) & set(new_exc):
        if old_exc[cls].get("bases") != new_exc[cls].get("bases"):
            impact = "Check our hierarchy" if cls in ours["exception_classes"] else "Not implemented"
            changes.append(
                {"category": "Exceptions", "change": "CHANGED", "name": cls,
                 "detail": f"bases: {old_exc[cls].get('bases')} → {new_exc[cls].get('bases')}", "impact": impact}
            )

    # --- Providers ---
    old_prov = old.get("providers", {}).get("provider_list", [])
    new_prov = new.get("providers", {}).get("provider_list", [])
    added_prov, removed_prov = diff_list(old_prov, new_prov)
    for p in added_prov:
        impact = "Already in our PROVIDERS" if p in ours["providers"] else "Not implemented"
        changes.append(
            {"category": "Providers", "change": "ADDED", "name": p, "detail": "", "impact": impact}
        )
    for p in removed_prov:
        changes.append(
            {"category": "Providers", "change": "REMOVED", "name": p, "detail": "", "impact": "Check if renamed"}
        )

    return changes


# ---------------------------------------------------------------------------
# Report / issue formatting
# ---------------------------------------------------------------------------


def format_report(changes: list[dict], old_meta: dict, new_meta: dict) -> str:
    """Format changes into a markdown issue body."""
    old_commit = old_meta.get("commit", "unknown")[:7]
    old_commit_full = old_meta.get("commit", "unknown")
    new_commit = new_meta.get("commit", "unknown")[:7]
    new_commit_full = new_meta.get("commit", "unknown")

    lines = [
        "## Upstream litellm changes detected",
        "",
        f"**Commit:** [`{new_commit}`](https://github.com/BerriAI/litellm/commit/{new_commit_full}) (prev: `{old_commit}`)",
        "",
    ]

    # Group by category
    categories = {}
    for c in changes:
        categories.setdefault(c["category"], []).append(c)

    for cat in ["API Surface", "Types", "Exceptions", "Providers"]:
        if cat not in categories:
            continue
        lines.append(f"### {cat}")
        lines.append("")
        lines.append("| Change | Name | Detail | litelm impact |")
        lines.append("|--------|------|--------|---------------|")
        for c in categories[cat]:
            lines.append(f"| {c['change']} | {c['name']} | {c['detail']} | {c['impact']} |")
        lines.append("")

    # Suggested actions
    has_impact = [c for c in changes if "We" in c["impact"] or "Check" in c["impact"] or "May need" in c["impact"]]
    if has_impact:
        lines.append("### Suggested actions")
        lines.append("")
        for c in has_impact:
            lines.append(f"- [ ] {c['change']} {c['name']}: {c['impact']}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Git clone
# ---------------------------------------------------------------------------


def clone_litellm(target: Path) -> str:
    """Shallow-clone litellm, return HEAD commit hash."""
    subprocess.run(
        ["git", "clone", "--depth=1", "--filter=blob:none", "--sparse", REPO_URL, str(target)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", "litellm/"],
        cwd=str(target), check=True, capture_output=True, text=True,
    )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(target), check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# GitHub issue creation
# ---------------------------------------------------------------------------


def create_or_comment_issue(body: str, dry_run: bool = False) -> None:
    """Create a new issue or comment on existing open upstream-watch issue."""
    if dry_run:
        print("=== DRY RUN: Issue body ===")
        print(body)
        return

    # Check for existing open issue
    result = subprocess.run(
        ["gh", "issue", "list", "--label", "upstream-watch", "--state", "open", "--json", "number", "--limit", "1"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        issues = json.loads(result.stdout) if result.stdout.strip() else []
        if issues:
            issue_num = issues[0]["number"]
            subprocess.run(
                ["gh", "issue", "comment", str(issue_num), "--body", body],
                check=True, capture_output=True, text=True,
            )
            print(f"Commented on issue #{issue_num}")
            return

    # Create new issue
    subprocess.run(
        ["gh", "issue", "create",
         "--title", "Upstream litellm drift detected",
         "--label", "upstream-watch",
         "--body", body],
        check=True, capture_output=True, text=True,
    )
    print("Created new upstream-watch issue")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Detect upstream litellm API drift")
    parser.add_argument("--snapshot-only", action="store_true", help="Write snapshots without diffing")
    parser.add_argument("--create-issue", action="store_true", help="Create/comment GH issue if changes found")
    parser.add_argument("--dry-run", action="store_true", help="Print issue body without calling gh")
    parser.add_argument("--local", type=str, help="Use existing litellm clone at this path")
    args = parser.parse_args()

    # Resolve litellm source dir
    tmpdir = None
    commit = "unknown"
    try:
        if args.local:
            local_path = Path(args.local)
            if not (local_path / "litellm").is_dir():
                print(f"ERROR: {local_path / 'litellm'} not found", file=sys.stderr)
                return 2
            litellm_dir = local_path / "litellm"
            # Try to get commit hash
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=str(local_path),
                    capture_output=True, text=True,
                )
                commit = result.stdout.strip() if result.returncode == 0 else "local"
            except FileNotFoundError:
                commit = "local"
        else:
            tmpdir = tempfile.mkdtemp(prefix="litellm_watch_")
            try:
                commit = clone_litellm(Path(tmpdir))
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"ERROR: git clone failed: {e}", file=sys.stderr)
                return 2
            litellm_dir = Path(tmpdir) / "litellm"

        # Extract upstream
        new_snapshots = extract_upstream(litellm_dir)
        new_meta = {
            "commit": commit,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        if args.snapshot_only:
            write_snapshots(new_snapshots, new_meta)
            print(f"Snapshots written to {SNAPSHOT_DIR}")
            return 0

        # Load old snapshots
        old_snapshots = {
            name: load_snapshot(name)
            for name in ("api_surface", "types", "exceptions", "providers")
        }
        old_meta = load_snapshot("meta")

        # Extract our surface for overlap detection
        ours = extract_ours()

        # Diff
        changes = diff_snapshots(old_snapshots, new_snapshots, ours)

        # Always update snapshots
        write_snapshots(new_snapshots, new_meta)

        if not changes:
            print("No meaningful upstream changes detected.")
            return 0

        print(f"Found {len(changes)} upstream change(s).")
        report = format_report(changes, old_meta, new_meta)

        if args.create_issue:
            create_or_comment_issue(report, dry_run=args.dry_run)
        else:
            print(report)

        return 1

    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
