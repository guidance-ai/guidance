from uuid import uuid4
from json import loads, dumps
from typing import Union
import re

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Json

from guidance._ast import ToolDefinition, SpecialToken, RuleNode, GrammarNode
from guidance.library import json, select, string, capture
from guidance import guidance


class RawToolCall(BaseModel):
    name: str
    args: Union[dict[str, Any], Json[dict[str, Any]]]


class ToolCallHandler(ABC):
    def __init__(self, tools: dict[str, ToolDefinition]):
        self.tools = tools

    @abstractmethod
    def build_grammar(self) -> GrammarNode:
        """
        Build the grammar for tool calls.
        Should return a GrammarNode that matches the tool call syntax.
        """
        pass

    @abstractmethod
    def parse_tool_call(self, text: str) -> RawToolCall:
        """
        Parse the tool call from the text.
        Should return a RawToolCall object with name and args.
        """
        pass

    @abstractmethod
    def format_return_value(self, value: Any) -> str:
        """
        Format the return value of the tool call.
        Should return a string representation of the value.
        """
        pass

    def invoke_tool(self, tool_call: RawToolCall) -> Any:
        """
        Invoke the tool with the parsed arguments.
        Returns the result of the tool call.
        """
        tool_name = tool_call.name
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found.")
        tool_def = self.tools[tool_name]
        args = tool_def.args.model_validate(tool_call.args).model_dump()
        return tool_def.callable(**args)


class Llama3FunctionToolCallHandler(ToolCallHandler):
    expr = re.compile(
        r"^<function=(?P<name>[^>]+)>(?P<args>\{(.|\n)*\})</function><\|eom_id\|>$"
    )

    def build_grammar(self) -> GrammarNode:
        return (
            RuleNode(name="trigger", lazy=True, value=string("<function="))
            + select(
                [
                    f"{name}>" + json(schema=defn.args.model_json_schema())
                    for name, defn in tools.items()
                ],
            )
            + "</function>"
            + SpecialToken("eom_id")
            + "\n"
        )

    def parse_tool_call(self, text: str) -> RawToolCall:
        match = self.expr.match(text)
        if not match:
            raise ValueError(f"Invalid tool call format: {text}")
        return RawToolCall.model_validate(match.groupdict())

    def format_return_value(self, value: Any) -> str:
        return "<|start_header_id|>ipython<|end_header_id|>\n\n" + dumps(value)


class Llama3IPythonToolCallHandler(ToolCallHandler):
    expr = re.compile(r"^<\|python_tag\|>(?P<call>\{(.|\n)*\})<\|eom_id\|>$")

    def build_grammar(self) -> GrammarNode:
        return (
            SpecialToken("python_tag")
            + json(
                schema={
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "const": name},
                                "parameters": defn.args.model_json_schema(),
                            },
                            "required": ["name", "parameters"],
                        }
                        for name, defn in self.tools.items()
                    ]
                }
            )
            + SpecialToken("eom_id")
            + "\n"
        )

    def parse_tool_call(self, text: str) -> RawToolCall:
        match = self.expr.match(text)
        if not match:
            raise ValueError(f"Invalid tool call format: {text}")
        call_data = loads(match.group("call"))
        return RawToolCall(name=call_data["name"], args=call_data["parameters"])

    def format_return_value(self, value: Any) -> str:
        return "<|start_header_id|>ipython<|end_header_id|>\n\n" + dumps(value)


@guidance
def handle_tool_call(
    lm,
    handler: ToolCallHandler,
):
    capture_id = f"_tool_call_{uuid4().hex}"
    grm = handler.build_grammar()
    lm += capture(grm, name=capture_id)
    tool_call_text = lm[capture_id]

    tool_call = handler.parse_tool_call(tool_call_text)
    response = handler.invoke_tool(tool_call)
    lm += handler.format_return_value(response)
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
    return [
        "1. BIRDS OF A FEATHER by Billie Eilish",
        "2. Espresso by Sabrina Carpenter",
        "3. Please Please Please by Sabrina Carpenter",
        "4. Not Like Us by Kendrick Lamar",
        "5. Gata Only by FloyyMenor, Cris Mj",
    ][:n]


tools = {
    "trending_songs": ToolDefinition.from_callable(
        trending_songs,
    )
}

with assistant():
    lm += handle_tool_call(
        # Llama3FunctionToolCallHandler(tools=tools)
        Llama3IPythonToolCallHandler(tools=tools)
    )

with assistant():
    lm += gen(max_tokens=100)
