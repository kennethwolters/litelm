import os
import sys
import threading
from datetime import datetime

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

import pytest
from litelm.integrations.humanloop import HumanLoopPromptManager
from litelm.types.utils import StandardCallbackDynamicParams
from litelm.litelm_core_utils.litelm_logging import DynamicLoggingCache
from unittest.mock import Mock, patch


def test_compile_prompt():
    prompt_manager = HumanLoopPromptManager()
    prompt_template = [
        {
            "content": "You are {{person}}. Answer questions as this person. Do not break character.",
            "name": None,
            "tool_call_id": None,
            "role": "system",
            "tool_calls": None,
        }
    ]
    prompt_variables = {"person": "John"}
    compiled_prompt = prompt_manager._compile_prompt_helper(
        prompt_template, prompt_variables
    )
    assert (
        compiled_prompt[0]["content"]
        == "You are John. Answer questions as this person. Do not break character."
    )
