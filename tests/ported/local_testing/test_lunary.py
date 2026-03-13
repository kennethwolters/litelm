import io
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import litelm
from litelm import completion

litelm.failure_callback = ["lunary"]
litelm.success_callback = ["lunary"]
litelm.set_verbose = True


def test_lunary_logging():
    try:
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what llm are u"}],
            max_tokens=10,
            temperature=0.2,
            user="test-user",
        )
        print(response)
    except Exception as e:
        print(e)


test_lunary_logging()


def test_lunary_template():
    import lunary

    try:
        template = lunary.render_template("test-template", {"question": "Hello!"})
        response = completion(**template)
        print(response)
    except Exception as e:
        print(e)


def test_lunary_logging_with_metadata():
    try:
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what llm are u"}],
            max_tokens=10,
            temperature=0.2,
            metadata={
                "run_name": "litelmRUN",
                "project_name": "litelm-completion",
                "tags": ["tag1", "tag2"],
            },
        )
        print(response)
    except Exception as e:
        print(e)


def test_lunary_with_tools():
    import litelm

    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco, Tokyo, and Paris?",
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    response = litelm.completion(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )

    response_message = response.choices[0].message
    print("\nLLM Response:\n", response.choices[0].message)


def test_lunary_logging_with_streaming_and_metadata():
    try:
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what llm are u"}],
            max_tokens=10,
            temperature=0.2,
            metadata={
                "run_name": "litelmRUN",
                "project_name": "litelm-completion",
            },
            stream=True,
        )
        for chunk in response:
            continue
    except Exception as e:
        print(e)
