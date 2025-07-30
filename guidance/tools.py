import re
from abc import ABC, abstractmethod
from json import dumps
from typing import Any, Union

from pydantic import BaseModel, Json

from ._ast import GrammarNode, RuleNode, ToolCallNode, ToolDefinition
from ._utils import text_to_grammar
from .library import json, regex, select
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

    def build_grammar(self, tokenizer: Tokenizer) -> GrammarNode:
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
    expr = re.compile(r"^<function=(?P<name>[^>]+)>(?P<args>\{(.|\n)*\})</function><\|eot_id\|>$")

    def trigger(self) -> str:
        return "<function="

    def begin(self, tool_name: str) -> str:
        return f"<function={tool_name}>"

    def body(self, tool: ToolDefinition) -> GrammarNode:
        return json(schema=tool.args.model_json_schema())

    def end(self) -> str:
        # eom / eot depends on "environment"?
        return "</function><|eot_id|>\n"

    def parse_tool_call(self, text: str) -> RawToolCall:
        match = self.expr.match(text)
        if not match:
            raise ValueError(f"Invalid tool call format: {text}")
        return RawToolCall.model_validate(match.groupdict())

    def format_return_value(self, value: Any) -> str:
        return "<|start_header_id|>ipython<|end_header_id|>\n\n" + dumps(value)


# class Llama3IPythonToolCallHandler(ToolCallHandler):
#     expr = re.compile(r"^<\|python_tag\|>(?P<call>\{(.|\n)*\})<\|eom_id\|>$")

#     def trigger(self) -> SpecialToken:
#         return special_token("<|python_tag|>")

#     def build_grammar(self) -> GrammarNode:
#         # https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md#model-response-format-5
#         return (
#             special_token("<python_tag>")
#             + json(
#                 schema={
#                     "oneOf": [
#                         {
#                             "type": "object",
#                             "properties": {
#                                 "type": {"type": "string", "const": "function"},
#                                 "name": {"type": "string", "const": name},
#                                 "parameters": defn.args.model_json_schema(),
#                             },
#                             # type is optional?
#                             "required": ["name", "parameters"],
#                         }
#                         for name, defn in self.tools.items()
#                     ]
#                 }
#             )
#             + special_token("<eom_id>")  # eom / eot depends on "environment"?
#             + "\n"
#         )

#     def parse_tool_call(self, text: str) -> RawToolCall:
#         match = self.expr.match(text)
#         if not match:
#             raise ValueError(f"Invalid tool call format: {text}")
#         call_data = loads(match.group("call"))
#         return RawToolCall(name=call_data["name"], args=call_data["parameters"])

#     def format_return_value(self, value: Any) -> str:
#         return "<|start_header_id|>ipython<|end_header_id|>\n\n" + dumps(value)


# class Qwen3ToolCallHandler(ToolCallHandler):
#     expr = re.compile(r"^<tool_call>\n(?P<call>\{(.|\n)*\})\n</tool_call><\|im_end\|>$")

#     def build_grammar(self) -> GrammarNode:
#         # https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json#L230
#         return (
#             "<tool_call>\n"  # note: not special
#             + json(
#                 schema={
#                     "oneOf": [
#                         {
#                             "type": "object",
#                             "properties": {
#                                 "name": {"type": "string", "const": name},
#                                 "arguments": defn.args.model_json_schema(),
#                             },
#                             "required": ["name", "arguments"],
#                         }
#                         for name, defn in self.tools.items()
#                     ]
#                 }
#             )
#             + "\n</tool_call>"
#             + special_token("<im_end>")
#             + "\n"
#         )

#     def parse_tool_call(self, text: str) -> RawToolCall:
#         match = self.expr.match(text)
#         if not match:
#             raise ValueError(f"Invalid tool call format: {text}")
#         call_data = loads(match.group("call"))
#         return RawToolCall(name=call_data["name"], args=call_data["arguments"])

#     def format_return_value(self, value: Any) -> str:
#         return f"<|im_start|>user\n<tool_response>\n{dumps(value)}\n</tool_response>"


# class LegacyToolCallHandler(ToolCallHandler):
#     expr = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*\(.*\)$")

#     def build_grammar(self) -> GrammarNode:
#         arg = lexeme(r"[^,=)]+")
#         kwarg = arg + "=" + arg
#         args = arg + zero_or_more("," + arg)
#         kwargs = kwarg + zero_or_more("," + kwarg)
#         return select(
#             [
#                 RuleNode(lazy=True, value=string(f"{name}("))
#                 + subgrammar(
#                     name="tool_args",
#                     body=optional(
#                         select(
#                             [
#                                 args,
#                                 kwargs,
#                                 args + "," + kwargs,
#                             ]
#                         )
#                     ),
#                     skip_regex=r"\s+",
#                 )
#                 + ")"
#                 for name in self.tools.keys()
#             ]
#         )

#     def parse_tool_call(self, text: str) -> RawToolCall:
#         match = self.expr.match(text)
#         if not match:
#             raise ValueError(f"Invalid tool call format: {text}")

#         # Parse the ast to get the args and kwargs
#         module = ast.parse(match.group())
#         if (
#             not len(module.body) == 1
#             or not isinstance(module.body[0], ast.Expr)
#             or not isinstance(module.body[0].value, ast.Call)
#         ):
#             raise ValueError(f"Invalid tool call format: {text}")
#         call = module.body[0].value
#         if not isinstance(call.func, ast.Name):
#             raise ValueError(f"Tool call must be a function call: {text}")
#         args = [ast.literal_eval(arg) for arg in call.args]
#         kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}

#         # Inspect the tool signature so we can bind the args correctly
#         try:
#             tool = self.tools[call.func.id]
#         except KeyError:
#             raise ValueError(f"Unknown tool: {call.func.id}")
#         signature = inspect.signature(tool.callable)
#         bound_args = signature.bind(*args, **kwargs)

#         return RawToolCall(name=call.func.id, args=cast(dict[str, Any], bound_args.arguments))

#     def format_return_value(self, value: Any) -> str:
#         return f" = {dumps(value)}"
