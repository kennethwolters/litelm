#### What this tests ####
#    This tests calling batch_completions by running 100 messages together

import sys, os
import traceback
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litelm


def test_update_model_cost():
    try:
        litelm.register_model(
            {
                "gpt-4": {
                    "max_tokens": 8192,
                    "input_cost_per_token": 0.00002,
                    "output_cost_per_token": 0.00006,
                    "litelm_provider": "openai",
                    "mode": "chat",
                },
            }
        )
        assert litelm.model_cost["gpt-4"]["input_cost_per_token"] == 0.00002
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")


# test_update_model_cost()


def test_update_model_cost_map_url():
    try:
        litelm.register_model(
            model_cost="https://raw.githubusercontent.com/BerriAI/litelm/main/model_prices_and_context_window.json"
        )
        assert litelm.model_cost["gpt-4"]["input_cost_per_token"] == 0.00003
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")


# test_update_model_cost_map_url()


def test_update_model_cost_via_completion():
    try:
        response = litelm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            input_cost_per_token=0.3,
            output_cost_per_token=0.4,
        )
        print(
            f"litelm.model_cost for gpt-3.5-turbo: {litelm.model_cost['gpt-3.5-turbo']}"
        )
        assert litelm.model_cost["gpt-3.5-turbo"]["input_cost_per_token"] == 0.3
        assert litelm.model_cost["gpt-3.5-turbo"]["output_cost_per_token"] == 0.4
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")


test_update_model_cost_via_completion()
