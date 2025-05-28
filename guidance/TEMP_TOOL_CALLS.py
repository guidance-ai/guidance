from guidance._ast import ToolDefinition, SpecialToken, RuleNode
from uuid import uuid4
from guidance.library import json, select, string
from guidance import guidance
from json import loads, dumps
import re


@guidance
def to_llama3_ipython(
    lm,
    tools: dict[str, ToolDefinition]
):
    capture_id = f"_tool_call_{uuid4().hex}"
    grm = (
        SpecialToken("python_tag")
        + select(
            [
                json(
                    schema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "const": name},
                            "parameters": defn.args.model_json_schema(),
                        },
                        "required": ["name", "parameters"],
                    }
                )
                for name, defn in tools.items()
            ],
            name=capture_id,
        )
        + SpecialToken("eom_id")
    )
    # Now do the work
    lm += grm
    call = loads(
        lm[capture_id]
    )
    tool = tools[call["name"]]
    args = tool.args.model_validate(call["parameters"]).model_dump()
    response = tool.callable(**args)
    lm += (
        SpecialToken("start_header_id")
        + "ipython"
        + SpecialToken("end_header_id")
        + "\n"
        + dumps(response)
    )
    return lm

@guidance
def to_llama3_function(
    lm,
    tools: dict[str, ToolDefinition]
):
    capture_id = f"_tool_call_{uuid4().hex}"
    grm = (
        RuleNode(
            name="trigger",
            lazy=True,
            value=string("<function=")
        )
        + select(
            [
                f"{name}>" + json(schema=defn.args.model_json_schema())
                for name, defn in tools.items()
            ],
            name=capture_id,
        )
        + "</function>"
        + SpecialToken("eom_id")
    )
    # Now do the work
    lm += grm
    name, args = re.match(
        r"^(.*?)>(\{.*\})$",
        lm[capture_id]
    ).groups()
    tool = tools[name]
    args = tool.args.model_validate_json(args).model_dump()
    response = tool.callable(**args)
    lm += (
        SpecialToken("start_header_id")
        + "ipython"
        + SpecialToken("end_header_id")
        + "\n"
        + dumps(response)
    )
    return lm

from guidance import *
from guidance.chat import Llama3ChatTemplate
model = models.LlamaCpp(chat_template=Llama3ChatTemplate, n_ctx=4096)

lm = model
with system():
    lm += """\
Environment: ipython

Cutting Knowledge Date: December 2023
Today Date: 21 September 2024

You are a helpful assistant."""

with user():
    lm += """\
 You have access to the following functions:

Use the function 'trending_songs' to 'Returns the trending songs on a Music site':
{"name": "trending_songs", "description": "Returns the trending songs on a Music site", "parameters": {"genre": {"description": "The genre of the songs to return", "param_type": "str", "required": false}, "n": {"description": "The number of songs to return", "param_type": "int", "required": true}}}

Think very carefully before calling functions.
If you choose to call a function ONLY reply in the following format with no prefix or suffix:

<function=example_function_name>{"example_name": "example_value"}</function>

Reminder:
- If looking for real time information use relevant functions before falling back to brave_search
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line<|eot_id|><|start_header_id|>user<|end_header_id|>

Use tools to get latest trending songs"""

def trending_songs(genre: str = None, n: int = 5):
    """Returns the trending songs on a Music site"""
    return ["1. BIRDS OF A FEATHER by Billie Eilish", "2. Espresso by Sabrina Carpenter", "3. Please Please Please by Sabrina Carpenter", "4. Not Like Us by Kendrick Lamar", "5. Gata Only by FloyyMenor, Cris Mj"][:n]

tools = {
    "trending_songs": ToolDefinition.from_callable(
        trending_songs,
    )
}

with assistant():
    lm += to_llama3_function(
        tools
    )

with assistant():
    lm += gen(max_tokens=100)
