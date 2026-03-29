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
# Change classification
# ---------------------------------------------------------------------------

_ACTIONABLE_KEYWORDS = ("We ", "Check our", "May need", "Already in our")


def _is_actionable(change: dict) -> bool:
    """Check if a change requires action on our side."""
    return any(kw in change["impact"] for kw in _ACTIONABLE_KEYWORDS)


# ---------------------------------------------------------------------------
# Report / issue formatting
# ---------------------------------------------------------------------------


def _format_change_table(changes: list[dict]) -> list[str]:
    """Format changes as markdown tables grouped by category."""
    lines: list[str] = []
    categories: dict[str, list[dict]] = {}
    for c in changes:
        categories.setdefault(c["category"], []).append(c)
    for cat in ["API Surface", "Types", "Exceptions", "Providers"]:
        if cat not in categories:
            continue
        lines.append(f"#### {cat}")
        lines.append("")
        lines.append("| Change | Name | Detail | Impact |")
        lines.append("|--------|------|--------|--------|")
        for c in categories[cat]:
            lines.append(f"| {c['change']} | {c['name']} | {c['detail']} | {c['impact']} |")
        lines.append("")
    return lines


def format_report(changes: list[dict], old_meta: dict, new_meta: dict) -> str:
    """Format changes into a markdown issue body."""
    old_commit = old_meta.get("commit", "unknown")[:7]
    old_commit_full = old_meta.get("commit", "unknown")
    new_commit = new_meta.get("commit", "unknown")[:7]
    new_commit_full = new_meta.get("commit", "unknown")

    lines = [
        "## Upstream litellm changes detected",
        "",
        f"**Range:** [`{old_commit}...{new_commit}`](https://github.com/BerriAI/litellm/compare/{old_commit_full}...{new_commit_full})",
        "",
    ]

    actionable = [c for c in changes if _is_actionable(c)]
    informational = [c for c in changes if not _is_actionable(c)]

    if actionable:
        lines.extend(_format_change_table(actionable))
        lines.append("### Suggested actions")
        lines.append("")
        for c in actionable:
            lines.append(f"- [ ] {c['change']} {c['name']}: {c['impact']}")
        lines.append("")

    if informational:
        lines.append("<details>")
        lines.append(f"<summary>{len(informational)} other change(s) outside our scope</summary>")
        lines.append("")
        lines.extend(_format_change_table(informational))
        lines.append("</details>")
        lines.append("")

    if not actionable:
        lines.append("*No actionable changes — all detected drift is outside our scope.*")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Commit log extraction (runs in watch script, before Pi)
# ---------------------------------------------------------------------------

_UPSTREAM_FILES = [
    "litellm/main.py",
    "litellm/__init__.py",
    "litellm/types/utils.py",
    "litellm/exceptions.py",
    "litellm/llms/anthropic/",
    "litellm/llms/bedrock/",
    "litellm/llms/mistral/",
]

_OUR_FILES = [
    "litelm/_completion.py",
    "litelm/_types.py",
    "litelm/_exceptions.py",
    "litelm/_embedding.py",
    "litelm/providers/_anthropic.py",
    "litelm/providers/_bedrock.py",
    "litelm/providers/_mistral.py",
    "litelm/__init__.py",
]

# Upstream → our implementation mapping
_FILE_MAP = {
    "litellm/main.py": "litelm/_completion.py",
    "litellm/__init__.py": "litelm/__init__.py",
    "litellm/types/utils.py": "litelm/_types.py",
    "litellm/exceptions.py": "litelm/_exceptions.py",
    "litellm/llms/anthropic/": "litelm/providers/_anthropic.py",
    "litellm/llms/bedrock/": "litelm/providers/_bedrock.py",
    "litellm/llms/mistral/": "litelm/providers/_mistral.py",
}


def fetch_commit_log(old_commit: str, new_commit: str) -> str:
    """Fetch relevant commits between old and new from GitHub API."""
    import urllib.request
    import urllib.error

    url = f"https://api.github.com/repos/BerriAI/litellm/compare/{old_commit[:12]}...{new_commit[:12]}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if token:
            req.add_header("Authorization", f"token {token}")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
        return f"ERROR: could not fetch commit log: {e}"

    commits = data.get("commits", [])
    files = data.get("files", [])

    relevant_prefixes = tuple(f.rstrip("/") for f in _UPSTREAM_FILES)
    relevant_files = [f for f in files if any(f["filename"].startswith(p) for p in relevant_prefixes)]

    lines = [f"{len(commits)} total commits, {len(files)} files changed, "
             f"{len(relevant_files)} in scope\n"]

    if relevant_files:
        lines.append("Files changed in our scope:")
        for f in relevant_files:
            lines.append(f"  {f['filename']} (+{f.get('additions', 0)}/-{f.get('deletions', 0)})")
        lines.append("")

    lines.append("Commits:")
    for c in commits:
        lines.append(f"  {c['sha'][:7]} {c['commit']['message'].splitlines()[0]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pi agent prompt generation
# ---------------------------------------------------------------------------


def generate_pi_prompt(
    commit_log: str, old_meta: dict, new_meta: dict, clone_path: str,
) -> str:
    """Generate verification prompt for the Pi agent."""
    old_commit = old_meta.get("commit", "unknown")
    new_commit = new_meta.get("commit", "unknown")

    file_map_str = "\n".join(
        f"  {up} → {ours}" for up, ours in _FILE_MAP.items()
    )

    return f"""# Upstream Behavioral Verification

## Your role
You are a verification agent for litelm, a 2,660 LOC reimplementation of litellm's
core routing+formatting. A structural AST diff has already detected field/signature
changes and is posted in the GitHub issue. Your job is different: find what the AST
diff CANNOT see — behavioral changes inside function bodies, bug fixes, new edge
cases — and verify every finding with evidence.

## Rules
1. **Every claim requires evidence.** Show the command and its output. "I believe X"
   is not a finding. "Verified: `grep -n reasoning_items litelm/_types.py` → 0 hits"
   is a finding.
2. **Do not restate the structural diff.** It is already in the issue. You add zero
   value by repeating it.
3. **If you find nothing new, say so.** "No behavioral divergences found beyond the
   structural diff." is a valid and useful report. Padding is worse than silence.
4. **Do not speculate about impact.** Either verify it or mark it "needs investigation."

## Upstream clone
{clone_path}

## File mapping (upstream → ours)
{file_map_str}

## Commit log (fetched by watch script)
{commit_log}

## Compare
https://github.com/BerriAI/litellm/compare/{old_commit}...{new_commit}

## Procedure

### 1. Identify in-scope commits
From the commit log above, filter for commits touching files in the file mapping.
Ignore commits to Router, proxy, caching, budgeting, callbacks, model registries.

### 2. Read upstream source at changed locations
For each in-scope file that changed, read the upstream version at `{clone_path}/`
and compare the behavioral logic against our implementation. Look for:
- Logic changes inside function bodies (invisible to AST diff)
- Bug fixes we may need to port
- New error handling or edge cases
- Changed streaming behavior or message translation

### 3. Verify each finding
For each potential divergence:
a) Show the upstream code (file:line, quote the relevant lines)
b) Show our code (file:line, quote the relevant lines)
c) Run a verification command (grep, test, or comparison) that proves the
   divergence exists or matters

### 4. Check DSPy impact
For any field or function that changed, verify whether DSPy actually uses it:
```bash
# Clone is at {clone_path} — DSPy source may not be available locally.
# Instead, grep our own codebase and CLAUDE.md for DSPy touchpoints.
grep -rn "reasoning_items\\|<field_name>" litelm/ CLAUDE.md
```

### 5. Run our tests
```bash
uv run pytest tests/ --ignore=tests/ported --timeout=10 -q
```

## Output format

Use this structure for each finding. If no findings, skip to the verdict.

### Finding: [short title]
**Upstream:** `file:line` — [quote changed code]
**Ours:** `file:line` — [quote our code]
**Evidence:**
```
[command you ran]
[its output]
```
**Verdict:** actionable / not actionable / needs investigation

---

### Test results
```
[paste test output]
```

### Bottom line
[One sentence: what we should do, or "nothing — no behavioral divergences found."]

Do NOT modify any files. Read-only analysis.
"""


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


def _ensure_label(name: str, color: str = "0e8a16", description: str = "") -> None:
    """Create a GitHub label if it doesn't exist."""
    result = subprocess.run(
        ["gh", "label", "list", "--search", name, "--json", "name"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        labels = json.loads(result.stdout) if result.stdout.strip() else []
        if any(lb["name"] == name for lb in labels):
            return
    cmd = ["gh", "label", "create", name, "--color", color]
    if description:
        cmd += ["--description", description]
    subprocess.run(cmd, capture_output=True, text=True)


def create_or_comment_issue(body: str, dry_run: bool = False) -> int | None:
    """Create a new issue or comment on existing. Returns issue number or None."""
    if dry_run:
        print("=== DRY RUN: would create/comment on upstream-watch issue ===")
        return None

    _ensure_label("upstream-watch", color="0e8a16", description="Automated upstream litellm drift detection")

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
            return issue_num

    # Create new issue
    result = subprocess.run(
        ["gh", "issue", "create",
         "--title", "Upstream litellm drift detected",
         "--label", "upstream-watch",
         "--body", body],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"gh issue create stderr: {result.stderr}", file=sys.stderr)
        result.check_returncode()  # raises CalledProcessError
    url = result.stdout.strip()
    issue_num = int(url.rstrip("/").split("/")[-1])
    print(f"Created new upstream-watch issue: {url}")
    return issue_num


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _write_output_dir(
    output_dir: str,
    report: str,
    old_meta: dict,
    new_meta: dict,
    clone_path: str,
    issue_num: int | None,
    has_actionable: bool,
) -> None:
    """Write analysis artifacts for the Pi agent step."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report)
    (out / "meta.json").write_text(json.dumps({
        "old_commit": old_meta.get("commit", "unknown"),
        "new_commit": new_meta.get("commit", "unknown"),
    }, indent=2) + "\n")

    # Fetch commit log (has network + GH_TOKEN here, Pi may not)
    old_commit = old_meta.get("commit", "unknown")
    new_commit = new_meta.get("commit", "unknown")
    commit_log = fetch_commit_log(old_commit, new_commit)
    (out / "commit_log.txt").write_text(commit_log)
    print(f"Fetched commit log: {commit_log.splitlines()[0]}")

    (out / "pi_prompt.md").write_text(
        generate_pi_prompt(commit_log, old_meta, new_meta, clone_path)
    )
    if has_actionable:
        (out / "has_actionable").write_text("true\n")
    if issue_num is not None:
        (out / "issue_number.txt").write_text(str(issue_num) + "\n")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Detect upstream litellm API drift")
    parser.add_argument("--snapshot-only", action="store_true", help="Write snapshots without diffing")
    parser.add_argument("--create-issue", action="store_true", help="Create/comment GH issue if changes found")
    parser.add_argument("--dry-run", action="store_true", help="Print issue body without calling gh")
    parser.add_argument("--local", type=str, help="Use existing litellm clone at this path")
    parser.add_argument("--keep-clone", action="store_true", help="Keep litellm clone for downstream analysis")
    parser.add_argument("--output-dir", type=str, metavar="DIR",
                        help="Write analysis artifacts (report, Pi prompt, meta) to DIR")
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
            clone_root = str(local_path)
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
            clone_root = tmpdir

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

        if not changes:
            if not args.dry_run:
                write_snapshots(new_snapshots, new_meta)
            print("No meaningful upstream changes detected.")
            return 0

        actionable = [c for c in changes if _is_actionable(c)]
        report = format_report(changes, old_meta, new_meta)
        print(f"Found {len(changes)} change(s) ({len(actionable)} actionable).")
        print(report)

        issue_num: int | None = None
        if args.create_issue:
            if actionable:
                try:
                    issue_num = create_or_comment_issue(report, dry_run=args.dry_run)
                except subprocess.CalledProcessError as e:
                    print(f"ERROR: issue creation failed: {e}", file=sys.stderr)
                    # Don't update snapshots — next run will re-detect
                    return 2
            else:
                print("No actionable changes — skipping issue creation.")

        # Write analysis artifacts for downstream Pi agent
        if args.output_dir:
            _write_output_dir(
                args.output_dir, report, old_meta, new_meta,
                clone_root, issue_num, bool(actionable),
            )

        # Update snapshots after successful reporting (skip in dry-run)
        if not args.dry_run:
            write_snapshots(new_snapshots, new_meta)
        return 0

    finally:
        if tmpdir and not args.keep_clone:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
