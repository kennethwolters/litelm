# What is this?
## Unit Tests for OpenAI Assistants API
import json
import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import logging

import pytest
from openai.types.beta.assistant import Assistant
from typing_extensions import override

import litelm
from litelm import create_thread, get_thread
from litelm.llms.openai.openai import (
    AssistantEventHandler,
    AsyncAssistantEventHandler,
    AsyncCursorPage,
    MessageData,
    OpenAIAssistantsAPI,
)
from litelm.llms.openai.openai import OpenAIMessage as Message
from litelm.llms.openai.openai import SyncCursorPage, Thread

"""
V0 Scope:

- Add Message -> `/v1/threads/{thread_id}/messages`
- Run Thread -> `/v1/threads/{thread_id}/run`
"""

def _add_azure_related_dynamic_params(data: dict) -> dict:
    data["api_version"] = "2024-02-15-preview"
    data["api_base"] = os.getenv("AZURE_API_BASE")
    data["api_key"] = os.getenv("AZURE_API_KEY")
    return data


@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.parametrize(
    "sync_mode",
    [True, False],
)
@pytest.mark.asyncio
async def test_get_assistants(provider, sync_mode):
    data = {
        "custom_llm_provider": provider,
    }
    if provider == "azure":
        data = _add_azure_related_dynamic_params(data)

    if sync_mode == True:
        assistants = litelm.get_assistants(**data)
        assert isinstance(assistants, SyncCursorPage)
    else:
        assistants = await litelm.aget_assistants(**data)
        assert isinstance(assistants, AsyncCursorPage)


@pytest.mark.parametrize("provider", ["azure", "openai"])
@pytest.mark.parametrize(
    "sync_mode",
    [True, False],
)
@pytest.mark.asyncio()
@pytest.mark.flaky(retries=3, delay=1)
async def test_create_delete_assistants(provider, sync_mode):
    litelm.ssl_verify = False
    litelm._turn_on_debug()
    data = {
        "custom_llm_provider": provider,
        "model": "gpt-4.1",
        "instructions": "You are a personal math tutor. When asked a question, write and run Python code to answer the question.",
        "name": "Math Tutor",
        "tools": [{"type": "code_interpreter"}],
    }
    if provider == "azure":
        data = _add_azure_related_dynamic_params(data)

    if sync_mode == True:
        assistant = litelm.create_assistants(**data)

        print("New assistants", assistant)
        assert isinstance(assistant, Assistant)
        assert (
            assistant.instructions
            == "You are a personal math tutor. When asked a question, write and run Python code to answer the question."
        )
        assert assistant.id is not None

        # delete the created assistant
        delete_data = {
            "custom_llm_provider": provider,
            "assistant_id": assistant.id,
        }
        if provider == "azure":
            delete_data = _add_azure_related_dynamic_params(delete_data)
        response = litelm.delete_assistant(**delete_data)
        print("Response deleting assistant", response)
        assert response.id == assistant.id
    else:
        assistant = await litelm.acreate_assistants(**data)
        print("New assistants", assistant)
        assert isinstance(assistant, Assistant)
        assert (
            assistant.instructions
            == "You are a personal math tutor. When asked a question, write and run Python code to answer the question."
        )
        assert assistant.id is not None

        # delete the created assistant
        delete_data = {
            "custom_llm_provider": provider,
            "assistant_id": assistant.id,
        }
        if provider == "azure":
            delete_data = _add_azure_related_dynamic_params(delete_data)
        response = await litelm.adelete_assistant(**delete_data)
        print("Response deleting assistant", response)
        assert response.id == assistant.id


@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_create_thread_litelm(sync_mode, provider) -> Thread:
    message: MessageData = {"role": "user", "content": "Hey, how's it going?"}  # type: ignore
    data = {
        "custom_llm_provider": provider,
        "message": [message],
    }
    if provider == "azure":
        data = _add_azure_related_dynamic_params(data)

    if sync_mode:
        new_thread = create_thread(**data)
    else:
        new_thread = await litelm.acreate_thread(**data)

    assert isinstance(
        new_thread, Thread
    ), f"type of thread={type(new_thread)}. Expected Thread-type"

    return new_thread


@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_get_thread_litelm(provider, sync_mode):
    new_thread = test_create_thread_litelm(sync_mode, provider)

    if asyncio.iscoroutine(new_thread):
        _new_thread = await new_thread
    else:
        _new_thread = new_thread

    data = {
        "custom_llm_provider": provider,
        "thread_id": _new_thread.id,
    }
    if provider == "azure":
        data = _add_azure_related_dynamic_params(data)

    if sync_mode:
        received_thread = get_thread(**data)
    else:
        received_thread = await litelm.aget_thread(**data)

    assert isinstance(
        received_thread, Thread
    ), f"type of thread={type(received_thread)}. Expected Thread-type"
    return new_thread


@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_add_message_litelm(sync_mode, provider):
    message: MessageData = {"role": "user", "content": "Hey, how's it going?"}  # type: ignore
    new_thread = test_create_thread_litelm(sync_mode, provider)

    if asyncio.iscoroutine(new_thread):
        _new_thread = await new_thread
    else:
        _new_thread = new_thread
    # add message to thread
    message: MessageData = {"role": "user", "content": "Hey, how's it going?"}  # type: ignore

    data = {"custom_llm_provider": provider, "thread_id": _new_thread.id, **message}
    if provider == "azure":
        data = _add_azure_related_dynamic_params(data)
    if sync_mode:
        added_message = litelm.add_message(**data)
    else:
        added_message = await litelm.a_add_message(**data)

    print(f"added message: {added_message}")

    assert isinstance(added_message, Message)


@pytest.mark.parametrize(
    "provider",
    [
        "azure",
        "openai",
    ],
)  #
@pytest.mark.parametrize(
    "sync_mode",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "is_streaming",
    [True, False],
)  #
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_aarun_thread_litelm(sync_mode, provider, is_streaming):
    """
    - Get Assistants
    - Create thread
    - Create run w/ Assistants + Thread
    """
    import openai



    try:
        get_assistants_data = {
            "custom_llm_provider": provider,
        }
        if provider == "azure":
            get_assistants_data = _add_azure_related_dynamic_params(get_assistants_data)
        if sync_mode:
            assistants = litelm.get_assistants(**get_assistants_data)
        else:
            assistants = await litelm.aget_assistants(**get_assistants_data)

        ## get the first assistant ###
        try:
            assistant_id = assistants.data[0].id
        except IndexError:
            pytest.skip("No assistants found")

        new_thread = test_create_thread_litelm(sync_mode=sync_mode, provider=provider)

        if asyncio.iscoroutine(new_thread):
            _new_thread = await new_thread
        else:
            _new_thread = new_thread

        thread_id = _new_thread.id

        # add message to thread
        message: MessageData = {"role": "user", "content": "Hey, how's it going?"}  # type: ignore

        data = {"custom_llm_provider": provider, "thread_id": _new_thread.id, **message}
        if provider == "azure":
            data = _add_azure_related_dynamic_params(data)

        if sync_mode:
            added_message = litelm.add_message(**data)

            if is_streaming:
                run = litelm.run_thread_stream(assistant_id=assistant_id, **data)
                with run as run:
                    assert isinstance(run, AssistantEventHandler)
                    print(run)
                    run.until_done()
            else:
                run = litelm.run_thread(
                    assistant_id=assistant_id, stream=is_streaming, **data
                )
                if run.status == "completed":
                    messages = litelm.get_messages(
                        thread_id=_new_thread.id, custom_llm_provider=provider
                    )
                    assert isinstance(messages.data[0], Message)
                else:
                    pytest.fail(
                        "An unexpected error occurred when running the thread, {}".format(
                            run
                        )
                    )

        else:
            added_message = await litelm.a_add_message(**data)

            if is_streaming:
                run = litelm.arun_thread_stream(assistant_id=assistant_id, **data)
                async with run as run:
                    print(f"run: {run}")
                    assert isinstance(
                        run,
                        AsyncAssistantEventHandler,
                    )
                    print(run)
                    await run.until_done()
            else:
                run = await litelm.arun_thread(
                    custom_llm_provider=provider,
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                )

                if run.status == "completed":
                    messages = await litelm.aget_messages(
                        thread_id=_new_thread.id, custom_llm_provider=provider
                    )
                    assert isinstance(messages.data[0], Message)
                else:
                    pytest.fail(
                        "An unexpected error occurred when running the thread, {}".format(
                            run
                        )
                    )
    except openai.APIError as e:
        pass
