import sys
import os
import io, asyncio

# import logging
# logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, os.path.abspath("../.."))

from litelm import completion
import litelm

litelm.num_retries = 3
litelm.success_callback = ["wandb"]
import time
import pytest


def test_wandb_logging_async():
    try:
        litelm.set_verbose = False

        async def _test_langfuse():
            from litelm import Router

            model_list = [
                {  # list of model deployments
                    "model_name": "gpt-3.5-turbo",
                    "litelm_params": {  # params for litelm completion/embedding call
                        "model": "gpt-3.5-turbo",
                        "api_key": os.getenv("OPENAI_API_KEY"),
                    },
                }
            ]

            router = Router(model_list=model_list)

            # openai.ChatCompletion.create replacement
            response = await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "this is a test with litelm router ?"}
                ],
            )
            print(response)

        response = asyncio.run(_test_langfuse())
        print(f"response: {response}")
    except litelm.Timeout as e:
        pass
    except Exception as e:
        pass


test_wandb_logging_async()


def test_wandb_logging():
    try:
        response = completion(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Hi 👋 - i'm claude"}],
            max_tokens=10,
            temperature=0.2,
        )
        print(response)
    except litelm.Timeout as e:
        pass
    except Exception as e:
        print(e)


# test_wandb_logging()
