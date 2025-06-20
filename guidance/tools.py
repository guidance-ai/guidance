from uuid import uuid4
from json import loads, dumps
from typing import Union
import re

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Json

from ._ast import ToolDefinition, SpecialToken, RuleNode, GrammarNode
from .library import json, select, string, capture
from ._guidance import guidance


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
        r"^<function=(?P<name>[^>]+)>(?P<args>\{(.|\n)*\})</function><\|eot_id\|>$"
    )

    def build_grammar(self) -> GrammarNode:
        # https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md#model-response-format-6
        return (
            RuleNode(name="trigger", lazy=True, value=string("<function="))
            + select(
                [
                    f"{name}>" + json(schema=defn.args.model_json_schema())
                    for name, defn in self.tools.items()
                ],
            )
            + "</function>"
            + SpecialToken("eot_id") # eom / eot depends on "environment"?
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
        # https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md#model-response-format-5
        return (
            SpecialToken("python_tag")
            + json(
                schema={
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "const": "function"},
                                "name": {"type": "string", "const": name},
                                "parameters": defn.args.model_json_schema(),
                            },
                            # type is optional?
                            "required": ["name", "parameters"],
                        }
                        for name, defn in self.tools.items()
                    ]
                }
            )
            + SpecialToken("eom_id") # eom / eot depends on "environment"?
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
