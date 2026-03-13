import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


from base_rerank_unit_tests import BaseLLMRerankTest
import litelm


class TestJinaAI(BaseLLMRerankTest):
    def get_custom_llm_provider(self) -> litelm.LlmProviders:
        return litelm.LlmProviders.JINA_AI

    def get_base_rerank_call_args(self) -> dict:
        return {
            "model": "jina_ai/jina-reranker-v2-base-multilingual",
        }


def test_jina_ai_embedding():
    litelm.embedding(
        model="jina_ai/jina-embeddings-v3",
        input=["a"],
        task="separation",
        dimensions=1024,
    )
