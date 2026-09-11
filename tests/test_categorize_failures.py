"""Tests for ported-test relevance classification."""

import pytest

from scripts.categorize_failures import relevance


@pytest.mark.parametrize(
    "classname",
    [
        "tests.ported.test_litellm.llms.anthropic.test_anthropic_common_utils",
        "tests.ported.test_litellm.test_openai_embedding_encoding_format_default",
        "tests.ported.test_litellm.test_shared_session_integration",
        "tests.ported.test_litellm.test_system_message_format_bug",
        "tests.ported.local_testing.test_acompletion_fallbacks",
        "tests.ported.test_litellm.llms.openai_like.test_meta_provider",
        "tests.ported.test_litellm.llms.soniox.test_soniox_provider_registration",
        "tests.ported.test_litellm.test_drop_params_env_var",
        "tests.ported.test_litellm.llms.openai_like.test_cognition_provider",
        "tests.ported.test_litellm.llms.openai_like.test_libertai_provider",
        "tests.ported.test_litellm.llms.openai_like.test_scx_ai_provider",
        "tests.ported.test_litellm.llms.parallel_ai.test_parallel_ai_search",
    ],
)
def test_known_out_of_scope_ported_failures_are_low_relevance(classname):
    assert relevance("assertion_error", "test_case", classname) == "low"


def test_unknown_core_failure_remains_high_relevance():
    assert relevance("assertion_error", "test_completion_contract", "tests.ported.test_completion") == "high"
