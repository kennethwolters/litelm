"""
Test that search_tool_name is properly filtered out from search requests.

The search_tool_name parameter is used internally by LiteLLM to identify
which search tool configuration to use, but should not be sent to external
search provider APIs.
"""
import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

from litelm.types.utils import all_litelm_params
from litelm.utils import filter_out_litelm_params


def test_search_tool_name_in_all_litelm_params():
    """
    Test that search_tool_name is in all_litelm_params.
    
    If missing, it gets passed to provider APIs causing errors.
    """
    assert "search_tool_name" in all_litelm_params


def test_filter_out_search_tool_name():
    """
    Test that filter_out_litelm_params correctly filters search_tool_name.
    """
    kwargs = {
        "query": "latest ai developments",
        "max_results": 5,
        "scrapeOptions": {"formats": ["markdown"]},
        "search_tool_name": "firecrawl-search",
        "metadata": {"user": "test"},
        "litelm_call_id": "test-123"
    }
    
    filtered = filter_out_litelm_params(kwargs=kwargs)
    
    assert "search_tool_name" not in filtered
    assert "metadata" not in filtered
    assert "litelm_call_id" not in filtered
    
    assert "query" in filtered
    assert "max_results" in filtered
    assert "scrapeOptions" in filtered
    assert filtered["query"] == "latest ai developments"
    assert filtered["max_results"] == 5

