"""
Tests for LiteLLM Responses bridge provider.

Inherits from BaseInteractionsTest to run the same test suite against
the litelm_responses bridge provider, which calls litelm.responses() internally.
"""

import os

from tests.test_litelm.interactions.base_interactions_test import (
    BaseInteractionsTest,
)


class TestLiteLLMResponsesBridge(BaseInteractionsTest):
    """Test LiteLLM Responses bridge using the base test suite."""
    
    def get_model(self) -> str:
        """Return the model string for the bridge provider.
        
        The bridge provider uses litelm.responses() internally, so we can
        use any model that litelm.responses() supports (e.g., gpt-4o).
        """
        return "gpt-4o"
    
    def get_api_key(self) -> str:
        """Return the OpenAI API key from environment."""
        return os.getenv("OPENAI_API_KEY", "")

