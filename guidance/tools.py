import re
from abc import ABC, abstractmethod
from json import dumps, loads
from typing import TYPE_CHECKING, Any, Union

from pydantic import BaseModel, Json

from ._ast import GrammarNode, RuleNode, ToolCallNode, ToolDefinition
from ._utils import text_to_grammar
from .library import json, regex, select

if TYPE_CHECKING:
    from .models._engine import Tokenizer


class RawToolCall(BaseModel):
    name: str
    args: Union[dict[str, Any], Json[dict[str, Any]]]


class ToolCallHandler(ABC):
    def __init__(self, tool_call_node: ToolCallNode):
        self.tool_call_node = tool_call_node

    @abstractmethod
    def trigger(self) -> str:
        """
        A substring or token that signals when to start applying a tool.
        For example "<function" or "<|python_tag|>".
        """
        pass

    @abstractmethod
    def begin(self, tool_name: str) -> str:
        """
        The beginning of the tool call. Must start with trigger. May depend on the tool name.
        For example "<function=foo>", '<|python_tag|>{"name":"foo","parameters":', or just "<|python_tag|>".
        """
        pass

    @abstractmethod
    def body(self, tool: ToolDefinition) -> GrammarNode:
        """
        The body of the tool call. Should return a GrammarNode that matches the tool call arguments.
        For example, if begin contains the tool name, it may be given by
        ```python
            guidance.json(schema=tool.args.model_json_schema())
        ```
        or if begin does not contain the tool name, it may be given by
        ```python
        guidance.json(
            schema={
                "type": "object",
                "properties": {
                    "name": tool.name,
                    "parameters": tool.args.model_json_schema(),
                },
                "required": ["name", "parameters"],
                "additionalProperties": False,
            }
        )
        ```
        """
        pass

    @abstractmethod
    def end(self) -> str:
        """
        The end of the tool call. Should return a string that ends the tool call.
        For example "</function><|eot_id|>\n" or "<|python_tag|><eom_id>".
        """
        pass

    @abstractmethod
    def parse_tool_calls(self, text: str) -> list[RawToolCall]:
        """
        Parse the tool calls from the text.
        Should return a list of RawToolCall objects with name and args.
        """
        pass

    @abstractmethod
    def format_return_value(self, value: Any) -> str:
        """
        Format the return value of the tool call.
        Should return a string representation of the value.
        """
        pass

    def build_grammar(self, tokenizer: "Tokenizer") -> GrammarNode:
        if self.tool_call_node.parallel_tool_calls:
            raise ValueError("Parallel tool calls are not supported by this handler.")
        trg = self.trigger()
        special = any(tokenizer.is_special_token(t) for t in tokenizer.encode(trg.encode()))
        trg_grm_inner = text_to_grammar(tokenizer, trg)
        options: list[GrammarNode] = []
        if self.tool_call_node.tool_choice != "required":
            text = regex(
                self.tool_call_node.plaintext_regex or r"(.|\n)*",
            )
            options.append(text)
            trg_grm_inner = text + trg_grm_inner

        trg_grm = RuleNode(name="trigger", lazy=not special, value=trg_grm_inner)
        for tool_name, tool_def in self.tool_call_node.tools.items():
            begin = self.begin(tool_name)
            if not begin.startswith(trg):
                raise ValueError(f"Tool call begin '{begin}' must start with trigger '{trg}'")
            grm = (
                trg_grm
                + text_to_grammar(tokenizer, begin[len(trg) :])
                + self.body(tool_def)
                + text_to_grammar(tokenizer, self.end())
            )
            options.append(grm)
        return select(options)

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
    # https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md#model-response-format-6
    expr = re.compile(r"<function=(?P<name>[^>]+)>(?P<args>\{(.|\n)*\})</function>")

    def trigger(self) -> str:
        return "<function="

    def begin(self, tool_name: str) -> str:
        return f"<function={tool_name}>"

    def body(self, tool: ToolDefinition) -> GrammarNode:
        return json(schema=tool.args.model_json_schema())

    def end(self) -> str:
        # eom / eot depends on "environment"?
        return "</function><|eot_id|>\n"

    def parse_tool_calls(self, text: str) -> list[RawToolCall]:
        matches = self.expr.finditer(text)
        tool_calls = []
        for match in matches:
            tool_calls.append(RawToolCall.model_validate(match.groupdict()))
        return tool_calls

    def format_return_value(self, value: Any) -> str:
        return "<|start_header_id|>ipython<|end_header_id|>\n\n" + dumps(value)


class Llama3IPythonToolCallHandler(ToolCallHandler):
    # https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md#model-response-format-5
    expr = re.compile(r"^<\|python_tag\|>(?P<call>\{(.|\n)*\})<\|eom_id\|>$")

    def trigger(self) -> str:
        return "<|python_tag|>"

    def begin(self, tool_name: str) -> str:
        return "<|python_tag|>"

    def body(self, tool: ToolDefinition) -> GrammarNode:
        return json(
            schema={
                "type": "object",
                "properties": {
                    "type": {"type": "string", "const": "function"},
                    "name": {"type": "string", "const": tool.name},
                    "parameters": tool.args.model_json_schema(),
                },
                # type is optional?
                "required": ["name", "parameters"],
                "additionalProperties": False,
            }
        )

    def end(self) -> str:
        return "<|eom_id|>\n"

    def parse_tool_calls(self, text: str) -> list[RawToolCall]:
        matches = self.expr.finditer(text)
        tool_calls = []
        for match in matches:
            call_data = loads(match.group("call"))
            tool_calls.append(RawToolCall(name=call_data["name"], args=call_data["parameters"]))
        return tool_calls

    def format_return_value(self, value: Any) -> str:
        return "<|start_header_id|>ipython<|end_header_id|>\n\n" + dumps(value)


class Qwen3ToolCallHandler(ToolCallHandler):
    expr = re.compile(r"<tool_call>\n(?P<call>\{(.|\n)*\})\n</tool_call>")

    def trigger(self):
        return "<tool_call>"

    def begin(self, tool_name: str) -> str:
        return "<tool_call>\n"

    def body(self, tool: ToolDefinition) -> GrammarNode:
        return json(
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "const": tool.name},
                    "arguments": tool.args.model_json_schema(),
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        )

    def end(self) -> str:
        return "\n</tool_call><|im_end|>\n"

    def parse_tool_calls(self, text: str) -> list[RawToolCall]:
        matches = self.expr.finditer(text)
        tool_calls = []
        for match in matches:
            call_data = loads(match.group("call"))
            tool_calls.append(RawToolCall(name=call_data["name"], args=call_data["arguments"]))
        return tool_calls

    def format_return_value(self, value: Any) -> str:
        return f"<|im_start|>user\n<tool_response>\n{dumps(value)}\n</tool_response>"
