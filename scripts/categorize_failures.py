#!/usr/bin/env python3
"""Parse JUnit XML from ported litellm tests and categorize failures."""

import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# Known litelm top-level exports (from __init__.py)
LITELM_EXPORTS = {
    "acompletion", "aembedding", "aresponses", "atext_completion",
    "AuthenticationError", "BadRequestError", "completion",
    "ContextWindowExceededError", "embedding", "get_supported_openai_params",
    "ModelResponse", "ModelResponseStream", "RateLimitError", "responses",
    "stream_chunk_builder", "supports_function_calling", "supports_reasoning",
    "supports_response_schema", "text_completion", "Timeout",
    "ChatCompletionMessageToolCall", "Choices", "Delta", "Function",
    "Message", "StreamingChoices", "Usage",
}

# Known litelm submodules (files that exist as importable modules)
LITELM_SUBMODULES = {
    "_client_cache", "_completion", "_dispatch", "_embedding",
    "_exceptions", "_providers", "_responses", "_types", "providers",
}

# Low-relevance path keywords — features litelm doesn't implement
LOW_RELEVANCE_KEYWORDS = [
    "proxy", "router", "caching", "budget", "guardrail", "agent",
    "image_gen", "audio", "ocr", "fine_tun", "batch", "assistant",
    "scheduler", "secret", "vector_store", "rag", "enterprise",
    "realtime", "rerank", "moderation", "speech", "pass_through",
    "store_model", "otel", "load_test", "e2e", "documentation_test",
    "openai_endpoints", "llm_translation", "llm_response_utils",
    "code_coverage", "spend_log", "team", "key_logging",
    "model_info", "model_cost", "prompt_factory", "containers_api",
    "evals_api", "skills_api", "skills_e2e",
]

# Error message patterns that indicate need for API keys / network (low relevance)
API_KEY_PATTERNS = [
    "api_key client option must be set",
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "Connection refused",
    "Connection error",
    "APIConnectionError",
    "Missing credentials",
    "All connection attempts failed",
]


def _has_litelm_submodule_import(text: str) -> tuple[bool, str]:
    """Check if text contains an import from a litelm submodule. Returns (is_absent, submod)."""
    match = re.search(r"(?:from|import)\s+litelm\.(\w+)", text)
    if match:
        submod = match.group(1)
        return submod not in LITELM_SUBMODULES, submod
    return False, ""


def _has_litelm_attr_missing(text: str) -> tuple[bool, str]:
    """Check for 'module litelm has no attribute X'."""
    match = re.search(r"module 'litelm' has no attribute '(\w+)'", text)
    if match:
        attr = match.group(1)
        return attr not in LITELM_EXPORTS, attr
    return False, ""


def _has_top_level_import_fail(text: str) -> tuple[bool, str]:
    """Check for 'cannot import name X from litelm'."""
    match = re.search(r"cannot import name '(\w+)' from 'litelm'", text)
    if match:
        name = match.group(1)
        return name not in LITELM_EXPORTS, name
    return False, ""


def classify(testcase: ET.Element) -> tuple[str, str]:
    """Classify a test case into (bucket, detail)."""
    failure = testcase.find("failure")
    error = testcase.find("error")
    skipped = testcase.find("skipped")

    if failure is None and error is None and skipped is None:
        return "passed", ""

    if skipped is not None:
        return "skipped", ""

    elem = error if error is not None else failure
    msg = elem.get("message", "") or ""
    etype = elem.get("type", "") or ""
    text = elem.text or ""
    combined = f"{etype}\n{msg}\n{text}"

    # Timeout
    if "timeout" in combined.lower() and ("Timeout" in msg or "timeout" in etype.lower()):
        return "timeout", ""

    # --- Import / ModuleNotFoundError checks (including collection failures) ---
    is_import_error = any(kw in combined for kw in ("ImportError", "ModuleNotFoundError"))
    is_collection_failure = msg == "collection failure"

    if is_import_error or is_collection_failure:
        # Check for absent submodule import
        is_absent, submod = _has_litelm_submodule_import(combined)
        if submod:
            if is_absent:
                return "import_absent_feature", f"litelm.{submod}"
            else:
                return "import_exists_wrong_path", f"litelm.{submod}"

        # Check for top-level import name failure
        is_absent, name = _has_top_level_import_fail(combined)
        if name:
            if is_absent:
                return "import_absent_feature", name
            else:
                return "import_exists_wrong_path", name

        # Check for non-litelm module imports (third-party deps)
        non_litelm = re.search(r"No module named '(\w+)'", combined)
        if non_litelm:
            mod = non_litelm.group(1)
            if mod != "litelm":
                return "import_absent_feature", f"third-party:{mod}"

        # Generic import error
        return "import_absent_feature", "generic"

    # Attribute errors on litelm module
    if "AttributeError" in combined:
        is_absent, attr = _has_litelm_attr_missing(combined)
        if attr:
            if is_absent:
                return "import_absent_feature", f"attr:{attr}"
            else:
                return "import_exists_wrong_path", f"attr:{attr}"

    # Connection / API key errors — needs real credentials
    if any(pat in combined for pat in API_KEY_PATTERNS):
        return "needs_api_key", msg[:200]

    # Missing files that litellm has but litelm doesn't
    if "model_prices_and_context_window" in combined:
        return "import_absent_feature", "model_prices_json"

    # pytest-asyncio missing marker
    if "async def functions are not natively supported" in combined:
        return "async_marker_missing", msg[:200]

    # Assertion errors
    if "AssertionError" in combined or "AssertionError" in etype:
        return "assertion_error", msg[:200]

    if "AssertionError" in msg or "assert " in msg.lower():
        return "assertion_error", msg[:200]

    return "runtime_error", msg[:200]


def relevance(bucket: str, testname: str, classname: str) -> str:
    """Classify as high or low relevance."""
    if bucket == "passed":
        return "high"
    if bucket in ("import_absent_feature", "timeout", "skipped",
                   "needs_api_key", "async_marker_missing"):
        return "low"
    if bucket in ("import_exists_wrong_path",):
        return "high"
    # assertion_error and runtime_error: check path for absent-feature keywords
    combined = f"{testname} {classname}".lower()
    if any(kw in combined for kw in LOW_RELEVANCE_KEYWORDS):
        return "low"
    return "high"


def parse_xml_to_buckets(xml_path: str) -> dict[str, str]:
    """Parse JUnit XML into {test_id: bucket} dict."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    results = {}
    for suite in root.iter("testsuite"):
        for tc in suite.iter("testcase"):
            name = tc.get("name", "unknown")
            classname = tc.get("classname", "")
            test_id = f"{classname}::{name}"
            bucket, _ = classify(tc)
            results[test_id] = bucket
    return results


def diff_baselines(old_xml: str, new_xml: str, out_dir: str | None = None):
    """Compare two baseline XMLs and print/write diff report."""
    old = parse_xml_to_buckets(old_xml)
    new = parse_xml_to_buckets(new_xml)
    all_ids = sorted(set(old) | set(new))

    categories = Counter()
    regressions = []
    improvements = []
    new_passes = []

    for tid in all_ids:
        ob = old.get(tid)
        nb = new.get(tid)
        if ob is None:
            cat = "NEW_PASS" if nb == "passed" else "NEW_TEST"
            if nb == "passed":
                new_passes.append(tid)
        elif nb is None:
            cat = "REMOVED"
        elif ob == nb:
            cat = "STABLE_PASS" if ob == "passed" else "STABLE_FAIL"
        elif ob == "passed" and nb != "passed":
            cat = "REGRESSION"
            regressions.append((tid, ob, nb))
        elif ob != "passed" and nb == "passed":
            cat = "IMPROVEMENT"
            improvements.append((tid, ob, nb))
        else:
            cat = "BUCKET_CHANGE"
        categories[cat] += 1

    # Print report
    print("=" * 60)
    print("BASELINE DIFF REPORT")
    print("=" * 60)
    print(f"\nOld: {old_xml} ({len(old)} tests)")
    print(f"New: {new_xml} ({len(new)} tests)")
    print()
    for cat in ["STABLE_PASS", "STABLE_FAIL", "IMPROVEMENT", "REGRESSION",
                "NEW_PASS", "NEW_FAIL", "NEW_TEST", "REMOVED", "BUCKET_CHANGE"]:
        if categories[cat]:
            print(f"  {cat:<20} {categories[cat]:>6}")

    if regressions:
        print(f"\n--- REGRESSIONS ({len(regressions)}) ---")
        for tid, ob, nb in regressions:
            print(f"  {tid}")
            print(f"    was: {ob} → now: {nb}")

    if improvements:
        print(f"\n--- IMPROVEMENTS ({len(improvements)}) ---")
        for tid, ob, nb in improvements:
            print(f"  {tid}")
            print(f"    was: {ob} → now: {nb}")

    if new_passes:
        print(f"\n--- NEW PASSES ({len(new_passes)}) ---")
        for tid in new_passes[:50]:
            print(f"  {tid}")
        if len(new_passes) > 50:
            print(f"  ... and {len(new_passes) - 50} more")

    # Write TSV
    if out_dir:
        tsv_path = Path(out_dir) / "diff_report.tsv"
        txt_path = Path(out_dir) / "diff_report.txt"
    else:
        tsv_path = Path(new_xml).parent / "diff_report.tsv"
        txt_path = Path(new_xml).parent / "diff_report.txt"

    with open(tsv_path, "w") as f:
        f.write("test_id\told_bucket\tnew_bucket\tcategory\n")
        for tid in all_ids:
            ob = old.get(tid, "")
            nb = new.get(tid, "")
            cat = "STABLE" if ob == nb else "CHANGED"
            if ob == "" :
                cat = "NEW"
            elif nb == "":
                cat = "REMOVED"
            elif ob == "passed" and nb != "passed":
                cat = "REGRESSION"
            elif ob != "passed" and nb == "passed":
                cat = "IMPROVEMENT"
            f.write(f"{tid}\t{ob}\t{nb}\t{cat}\n")
    print(f"\nTSV written to {tsv_path}")

    # Capture report to txt
    import io, contextlib
    # Already printed above, just write summary to file
    with open(txt_path, "w") as f:
        f.write(f"Old: {old_xml} ({len(old)} tests)\n")
        f.write(f"New: {new_xml} ({len(new)} tests)\n\n")
        for cat in ["STABLE_PASS", "STABLE_FAIL", "IMPROVEMENT", "REGRESSION",
                    "NEW_PASS", "NEW_FAIL", "NEW_TEST", "REMOVED", "BUCKET_CHANGE"]:
            if categories[cat]:
                f.write(f"  {cat:<20} {categories[cat]:>6}\n")
        if regressions:
            f.write(f"\nREGRESSIONS ({len(regressions)}):\n")
            for tid, ob, nb in regressions:
                f.write(f"  {tid}: {ob} → {nb}\n")
        if improvements:
            f.write(f"\nIMPROVEMENTS ({len(improvements)}):\n")
            for tid, ob, nb in improvements:
                f.write(f"  {tid}: {ob} → {nb}\n")
    print(f"Report written to {txt_path}")


def main():
    # --diff mode
    if len(sys.argv) >= 4 and sys.argv[1] == "--diff":
        out_dir = None
        if len(sys.argv) >= 5 and sys.argv[4:]:
            out_dir = sys.argv[4]
        diff_baselines(sys.argv[2], sys.argv[3], out_dir)
        return

    xml_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/litelm_results.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()

    buckets = Counter()
    detail_counts = Counter()
    relevance_counts = Counter()
    high_relevance_failures = []

    for suite in root.iter("testsuite"):
        for tc in suite.iter("testcase"):
            name = tc.get("name", "unknown")
            classname = tc.get("classname", "")
            bucket, detail = classify(tc)
            buckets[bucket] += 1
            if bucket == "import_absent_feature":
                detail_counts[detail] += 1
            rel = relevance(bucket, name, classname)
            relevance_counts[(bucket, rel)] += 1

            if rel == "high" and bucket != "passed":
                error_elem = tc.find("error")
                if error_elem is None:
                    error_elem = tc.find("failure")
                error_msg = ""
                if error_elem is not None:
                    error_msg = (error_elem.get("message", "") or "")[:200]
                high_relevance_failures.append({
                    "test": f"{classname}::{name}",
                    "bucket": bucket,
                    "detail": detail,
                    "error": error_msg,
                })

    # Print summary
    print("=" * 60)
    print("PORTED TEST RESULTS CATEGORIZATION")
    print("=" * 60)
    total = sum(buckets.values())
    print(f"\nTotal tests: {total}\n")

    print(f"{'Bucket':<30} {'Count':>6} {'High':>6} {'Low':>6}")
    print("-" * 50)
    for bucket in ["passed", "import_absent_feature", "import_exists_wrong_path",
                    "assertion_error", "runtime_error", "needs_api_key",
                    "async_marker_missing", "timeout", "skipped"]:
        count = buckets.get(bucket, 0)
        high = relevance_counts.get((bucket, "high"), 0)
        low = relevance_counts.get((bucket, "low"), 0)
        print(f"{bucket:<30} {count:>6} {high:>6} {low:>6}")

    print(f"\n{'TOTAL':<30} {total:>6} "
          f"{sum(v for (_, r), v in relevance_counts.items() if r == 'high'):>6} "
          f"{sum(v for (_, r), v in relevance_counts.items() if r == 'low'):>6}")

    # Absent feature breakdown
    print(f"\n{'=' * 60}")
    print("ABSENT FEATURE BREAKDOWN (top 30)")
    print(f"{'=' * 60}")
    for detail, count in detail_counts.most_common(30):
        print(f"  {count:>5}  {detail}")

    # High relevance failures detail
    print(f"\n{'=' * 60}")
    print(f"HIGH-RELEVANCE FAILURES ({len(high_relevance_failures)} tests)")
    print(f"{'=' * 60}")
    for f in high_relevance_failures[:100]:
        print(f"\n[{f['bucket']}] {f['test']}")
        if f['detail']:
            print(f"  detail: {f['detail']}")
        print(f"  {f['error'][:150]}")

    if len(high_relevance_failures) > 100:
        print(f"\n... and {len(high_relevance_failures) - 100} more")

    # Write high-relevance failures to a file for issue creation
    out_path = Path(xml_path).parent / "litelm_high_relevance.txt"
    with open(out_path, "w") as fh:
        for f in high_relevance_failures:
            fh.write(f"{f['bucket']}\t{f['test']}\t{f['detail']}\t{f['error']}\n")
    print(f"\nHigh-relevance failures written to {out_path}")


if __name__ == "__main__":
    main()
