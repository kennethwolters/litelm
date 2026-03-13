import pytest
from litelm import acompletion
from litelm import completion


def test_acompletion_params():
    import inspect
    from litelm.types.completion import CompletionRequest

    acompletion_params_odict = inspect.signature(acompletion).parameters
    completion_params_dict = inspect.signature(completion).parameters

    acompletion_params = {
        name: param.annotation for name, param in acompletion_params_odict.items()
    }
    completion_params = {
        name: param.annotation for name, param in completion_params_dict.items()
    }

    keys_acompletion = set(acompletion_params.keys())
    keys_completion = set(completion_params.keys())

    print(keys_acompletion)
    print("\n\n\n")
    print(keys_completion)

    print("diff=", keys_completion - keys_acompletion)

    # Assert that the parameters are the same
    if keys_acompletion != keys_completion:
        pytest.fail(
            "The parameters of the litelm.acompletion function and litelm.completion are not the same. "
            f"Completion has extra keys: {keys_completion - keys_acompletion}"
        )


# test_acompletion_params()


@pytest.mark.asyncio
async def test_langfuse_double_logging():
    import litelm

    litelm.set_verbose = True
    litelm.success_callback = ["langfuse"]
    litelm.failure_callback = ["langfuse"]  # logs errors to langfuse

    models = ["gpt-4o-mini", "claude-3-5-haiku-20241022"]

    messages = [
        {"role": "user", "content": "Hello, how are you?"},
    ]

    resp = await litelm.acompletion(
        model=models[0],
        messages=messages,
        temperature=0.0,
        fallbacks=models[1:],
        # metadata={"generation_name": "test-gen", "project": "litelm-test"},
    )
    return resp
