from json import dumps
from typing import Optional

import pytest
from jinja2.environment import Environment

import guidance
from guidance import assistant, gen, system, user
from guidance._ast import ToolDefinition
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
    called_args = []

    def trending_songs(n: int, genre: Optional[str] = None) -> list[str]:
        called_args.append({"n": n, "genre": genre})
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
        lm += """What are the latest trending songs?"""

    with assistant():
        lm += gen(tools=[trending_songs])

    with assistant():
        lm += gen(name="follow_up")

    assert called_args

    jinja_env = Environment().from_string(lm._interpreter.engine.model_obj.metadata["tokenizer.chat_template"])
    tc = {"name": "trending_songs", "arguments": called_args[0]}
    jinja2_render = jinja_env.render(
        messages=[
            {"role": "user", "content": "What are the latest trending songs?"},
            {"role": "assistant", "content": f"<tool_call>\n{dumps(tc)}\n</tool_call>"},
            {"role": "tool", "content": dumps(["Song 1", "Song 2", "Song 3"])},
            {"role": "assistant", "content": lm["follow_up"]},
        ],
        tools=[ToolDefinition.from_callable(trending_songs).args.model_json_schema()],
    )

    # Compare the tokenization of the strings, rather than the strings
    # themselves (e.g. `<|user|>` may tokenize the same as `<|user|>\n`)
    lm_tokens = lm._interpreter.engine.tokenizer.encode(str(lm).encode())
    jinja2_tokens = lm._interpreter.engine.tokenizer.encode(jinja2_render.encode())

    # Only check substring due to BOS/EOS tokens, unfinished closing tags
    diff = len(jinja2_tokens) - len(lm_tokens)
    assert diff >= 0
    for i in range(diff + 1):
        if jinja2_tokens[i : i + len(lm_tokens)] == lm_tokens:
            break
    else:
        raise AssertionError("lm mismatches jinja template", str(lm), str(jinja2_render))
