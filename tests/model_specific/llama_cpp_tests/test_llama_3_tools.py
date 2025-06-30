from typing import Optional

import pytest

import guidance
from guidance import assistant, gen, system, user
from guidance.tools import Llama3FunctionToolCallHandler, Llama3IPythonToolCallHandler


@pytest.fixture(scope="module")
def llama3_point_2(selected_model, selected_model_name):
    if selected_model_name in ["llamacpp_llama3.2_3b_cpu"] and selected_model is not None:
        return selected_model
    else:
        pytest.skip("Requires Llama3.2")


def test_llama_cpp_python_tool(llama3_point_2: guidance.models.Model):
    lm = llama3_point_2
    lm._interpreter.tool_call_handler_cls = Llama3IPythonToolCallHandler
    called = False

    def trending_songs(n: int, genre: Optional[str] = None) -> list[str]:
        nonlocal called
        called = True
        return ["Song 1", "Song 2", "Song 3"]

    with system():
        lm += """\
Environment: ipython

You are a helpful assistant.
"""

    with user():
        lm += """\
Answer the user's question by making use of the following functions if needed.
If none of the function can be used, please say so.
Here is a list of functions in JSON format:
{
    "type": "function",
    "function": {
        "name": "trending_songs",
        "description": "Returns the trending songs on a Music site",
        "parameters": {
            "type": "object",
            "properties": [
                {
                    "n": {
                        "type": "integer",
                        "description": "The number of songs to return"
                    }
                },
                {
                    "genre": {
                        "type": "string",
                        "description": "The genre of the songs to return"
                    }
                }
            ],
            "required": ["n"]
        }
    }
}

Return function calls in JSON format"""

    with user():
        lm += """\
Use tools to get latest trending songs."""

    with assistant():
        lm += gen(tools=[trending_songs])
    assert called


def test_llama_cpp_function(llama3_point_2: guidance.models.Model):
    lm = llama3_point_2
    lm._interpreter.tool_call_handler_cls = Llama3FunctionToolCallHandler
    called = False

    def trending_songs(n: int, genre: Optional[str] = None) -> list[str]:
        nonlocal called
        called = True
        return ["Song 1", "Song 2", "Song 3"]

    with system():
        lm += """\
You are a helpful assistant.
"""

    with user():
        lm += """\
You have access to the following functions:

Use the function 'trending_songs' to 'Returns the trending songs on a Music site':
{"name": "trending_songs", "description": "Returns the trending songs on a Music site", "parameters": {"genre": {"description": "The genre of the songs to return", "param_type": "str", "required": false}, "n": {"description": "The number of songs to return", "param_type": "int", "required": true}}}

Think very carefully before calling functions.
If you choose to call a function ONLY reply in the following format with no prefix or suffix:

<function=example_function_name>{"example_name": "example_value"}</function>"""

    with user():
        lm += """\
Use tools to get latest trending songs."""

    with assistant():
        lm += gen(tools=[trending_songs])

    assert called
