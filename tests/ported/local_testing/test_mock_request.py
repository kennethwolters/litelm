#### What this tests ####
#    This tests mock request calls to litelm

import os
import sys
import traceback

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litelm
import time


def test_mock_request():
    try:
        model = "gpt-3.5-turbo"
        messages = [{"role": "user", "content": "Hey, I'm a mock request"}]
        response = litelm.mock_completion(model=model, messages=messages, stream=False)
        print(response)
        print(type(response))
    except Exception:
        traceback.print_exc()


# test_mock_request()
def test_streaming_mock_request():
    try:
        model = "gpt-3.5-turbo"
        messages = [{"role": "user", "content": "Hey, I'm a mock request"}]
        response = litelm.mock_completion(model=model, messages=messages, stream=True)
        complete_response = ""
        for chunk in response:
            complete_response += chunk["choices"][0]["delta"]["content"] or ""
        if complete_response == "":
            raise Exception("Empty response received")
    except Exception:
        traceback.print_exc()


# test_streaming_mock_request()


@pytest.mark.asyncio()
async def test_async_mock_streaming_request():
    generator = await litelm.acompletion(
        messages=[{"role": "user", "content": "Why is LiteLLM amazing?"}],
        mock_response="LiteLLM is awesome",
        stream=True,
        model="gpt-3.5-turbo",
    )
    complete_response = ""
    async for chunk in generator:
        print(chunk)
        complete_response += chunk["choices"][0]["delta"]["content"] or ""

    assert (
        complete_response == "LiteLLM is awesome"
    ), f"Unexpected response got {complete_response}"


def test_mock_request_n_greater_than_1():
    try:
        model = "gpt-3.5-turbo"
        messages = [{"role": "user", "content": "Hey, I'm a mock request"}]
        response = litelm.mock_completion(model=model, messages=messages, n=5)
        print("response: ", response)

        assert len(response.choices) == 5
        for choice in response.choices:
            assert choice.message.content == "This is a mock request"

    except Exception:
        traceback.print_exc()


@pytest.mark.asyncio()
async def test_async_mock_streaming_request_n_greater_than_1():
    generator = await litelm.acompletion(
        messages=[{"role": "user", "content": "Why is LiteLLM amazing?"}],
        mock_response="LiteLLM is awesome",
        stream=True,
        model="gpt-3.5-turbo",
        n=5,
    )
    complete_response = ""
    async for chunk in generator:
        print(chunk)
        # complete_response += chunk["choices"][0]["delta"]["content"] or ""

    # assert (
    #     complete_response == "LiteLLM is awesome"
    # ), f"Unexpected response got {complete_response}"


def test_mock_request_with_mock_timeout():
    """
    Allow user to set 'mock_timeout = True', this allows for testing if fallbacks/retries are working on timeouts.
    """
    start_time = time.time()
    with pytest.raises(litelm.Timeout):
        response = litelm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, I'm a mock request"}],
            timeout=3,
            mock_timeout=True,
        )
    end_time = time.time()
    assert end_time - start_time >= 3, f"Time taken: {end_time - start_time}"


def test_router_mock_request_with_mock_timeout():
    """
    Allow user to set 'mock_timeout = True', this allows for testing if fallbacks/retries are working on timeouts.
    """
    start_time = time.time()
    router = litelm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litelm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ],
    )
    with pytest.raises(litelm.Timeout):
        response = router.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, I'm a mock request"}],
            timeout=3,
            mock_timeout=True,
        )
        print(response)
    end_time = time.time()
    assert end_time - start_time >= 3, f"Time taken: {end_time - start_time}"


def test_router_mock_request_with_mock_timeout_with_fallbacks():
    """
    Allow user to set 'mock_timeout = True', this allows for testing if fallbacks/retries are working on timeouts.
    """
    litelm.set_verbose = True
    start_time = time.time()
    router = litelm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litelm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
            {
                "model_name": "gpt-4.1-nano",
                "litelm_params": {
                    "model": "gpt-4.1-nano",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ],
        fallbacks=[{"gpt-3.5-turbo": ["gpt-4.1-nano"]}],
    )
    response = router.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hey, I'm a mock request"}],
        timeout=3,
        num_retries=1,
        mock_timeout=True,
    )
    print(response)
    end_time = time.time()
    assert end_time - start_time >= 3, f"Time taken: {end_time - start_time}"
    assert (
        "gpt-4.1-nano" in response.model
    ), "Model should be gpt-4.1-nano"
