from json import loads, dumps
import ast
import inspect
from typing import Union, cast
import re

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Json

from ._ast import ToolDefinition, SpecialToken, RuleNode, GrammarNode
from .library import json, select, string, optional, zero_or_more
from .library._subgrammar import subgrammar, lexeme


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

class LegacyToolCallHandler(ToolCallHandler):
    expr = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*\(.*\)$")

    def build_grammar(self) -> GrammarNode:
        arg = lexeme(r"[^,=)]+")
        kwarg = arg + "=" + arg
        args = arg + zero_or_more("," + arg)
        kwargs = kwarg + zero_or_more("," + kwarg)
        return select([
            RuleNode(lazy=True, value=string(f"{name}("))
            + subgrammar(
                name="tool_args",
                body=optional(
                    select([
                        args,
                        kwargs,
                        args + "," + kwargs,
                    ])
                ),
                skip_regex=r"\s+"
            )
            + ")"
            for name in self.tools.keys()
        ])

    def parse_tool_call(self, text: str) -> RawToolCall:
        match = self.expr.match(text)
        if not match:
            raise ValueError(f"Invalid tool call format: {text}")

        # Parse the ast to get the args and kwargs
        module = ast.parse(match.group())
        if not len(module.body) == 1 or not isinstance(module.body[0], ast.Expr) or not isinstance(module.body[0].value, ast.Call):
            raise ValueError(f"Invalid tool call format: {text}")
        call = module.body[0].value
        if not isinstance(call.func, ast.Name):
            raise ValueError(f"Tool call must be a function call: {text}")
        args = [
            ast.literal_eval(arg) for arg in call.args
        ]
        kwargs = {
            kw.arg: ast.literal_eval(kw.value) for kw in call.keywords
        }

        # Inspect the tool signature so we can bind the args correctly
        try:
            tool = self.tools[call.func.id]
        except KeyError:
            raise ValueError(f"Unknown tool: {call.func.id}")
        signature = inspect.signature(tool.callable)
        bound_args = signature.bind(*args, **kwargs)

        return RawToolCall(name=call.func.id, args=cast(dict[str, Any], bound_args.arguments))

    def format_return_value(self, value: Any) -> str:
        return f" = {dumps(value)}"
