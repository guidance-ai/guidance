from typing import Optional

import pytest

import guidance
from guidance import assistant, gen, system, user
from guidance.tools import Qwen3ToolCallHandler


@pytest.fixture(scope="module")
def qwen3(selected_model, selected_model_name):
    if selected_model_name in ["llamacpp_qwen3_8b_cpu"] and selected_model is not None:
        return selected_model
    else:
        pytest.skip("Requires Qwen3")


def test_qwen3_tool(qwen3: guidance.models.Model):
    lm = qwen3
    lm._interpreter.tool_call_handler_cls = Qwen3ToolCallHandler
    called = False

    def trending_songs(n: int, genre: Optional[str] = None) -> list[str]:
        nonlocal called
        called = True
        return ["Song 1", "Song 2", "Song 3"]

    with system():
        lm += """\
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"additionalProperties": false, "properties": {"genre": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Genre"}, "n": {"title": "N", "type": "integer"}}, "required": ["n", "genre"], "title": "trending_songs", "type": "object"}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

    with user():
        lm += """\
Use tools to get latest trending songs."""

    with assistant():
        lm += gen(tools=[trending_songs])
    assert called
