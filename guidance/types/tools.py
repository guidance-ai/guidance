import builtins
import inspect
import sys
import textwrap
import traceback
from types import TracebackType
from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, create_model, field_serializer

if TYPE_CHECKING:
    from .._ast import GrammarNode


class GrammarFormat(BaseModel):
    type: Literal["grammar"] = "grammar"
    syntax: Literal["lark", "regex"]
    definition: str


# Placeholder TypeAlias for possible future Union
CustomFormat = GrammarFormat


class CustomTool(BaseModel):
    type: Literal["custom"] = "custom"
    format: CustomFormat


class FunctionTool(BaseModel):
    type: Literal["function"] = "function"
    parameters: Union[builtins.type[BaseModel], dict[str, Any]]

    @classmethod
    def from_callable(cls, callable: Callable) -> "FunctionTool":
        from guidance._guidance import GuidanceFunction

        if isinstance(callable, GuidanceFunction):
            raise TypeError(
                "An @guidance-wrapped function was passed to Tool.from_callable. The function must be called and return a valid grammar, which should be passed to Tool.from_grammar."
            )

        signature = inspect.signature(callable)
        parameters = {}
        for name, param in signature.parameters.items():
            if param.kind not in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }:
                raise ValueError(f"Unsupported parameter kind: {param.kind.description}")
            parameters[name] = param.annotation if param.annotation is not inspect.Parameter.empty else Any

        return FunctionTool(
            parameters=create_model(callable.__name__, __config__=ConfigDict(extra="forbid"), **parameters),
        )

    def get_schema(self) -> dict[str, Any]:
        """
        Returns the JSON schema for the function's parameters.
        If the parameters are a Pydantic model, it will return the model's schema.
        If they are a dict, it will return the dict as is.
        """
        return self.serialize_parameters(self.parameters)

    @field_serializer("parameters", mode="plain")
    def serialize_parameters(self, parameters: Union[builtins.type[BaseModel], dict[str, Any]]) -> dict[str, Any]:
        if isinstance(parameters, type) and issubclass(parameters, BaseModel):
            return parameters.model_json_schema()
        elif isinstance(parameters, dict):
            return parameters
        else:
            raise TypeError(f"Unsupported parameters type: {type(parameters)}. Expected a Pydantic model or a dict.")


ToolType = Annotated[Union[FunctionTool, CustomTool], Field(discriminator="type")]


class Tool(BaseModel):
    name: str
    description: str
    tool: ToolType
    callable: Callable
    exc_formatter: Optional[Callable[[type[BaseException], BaseException, TracebackType], str]]

    def call(self, *args, **kwargs) -> Any:
        try:
            return self.callable(*args, **kwargs)
        except BaseException:  # noqa: BLE001
            if self.exc_formatter is None:
                return "".join(traceback.format_exception(*sys.exc_info()))
            return self.exc_formatter(*sys.exc_info())

    @classmethod
    def from_callable(
        cls,
        callable: Callable,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Union[builtins.type[BaseModel], dict[str, Any]]] = None,
    ) -> "Tool":
        if parameters is not None:
            tool = FunctionTool(parameters=parameters)
        else:
            tool = FunctionTool.from_callable(callable)

        return Tool(
            name=name or callable.__name__,
            description=description or textwrap.dedent((callable.__doc__ or "").strip()),
            tool=tool,
            callable=callable,
        )

    @classmethod
    def from_regex(
        cls,
        pattern: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        callable: Callable,
    ) -> "Tool":
        return Tool(
            name=name or callable.__name__,
            description=description or (callable.__doc__ or "").strip(),
            tool=CustomTool(
                format=GrammarFormat(
                    syntax="regex",
                    definition=pattern,
                ),
            ),
            callable=callable,
        )

    @classmethod
    def from_lark(
        cls,
        lark: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        callable: Callable,
    ) -> "Tool":
        return Tool(
            name=name or callable.__name__,
            description=description or (callable.__doc__ or "").strip(),
            tool=CustomTool(
                format=GrammarFormat(
                    syntax="lark",
                    definition=lark,
                )
            ),
            callable=callable,
        )

    @classmethod
    def from_grammar(
        cls,
        grammar: "GrammarNode",
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        callable: Callable,
    ) -> "Tool":
        from guidance._guidance import GuidanceFunction

        if isinstance(grammar, GuidanceFunction):
            raise TypeError(
                "An @guidance-wrapped function was passed to Tool.from_grammar. The function must be called and return a valid grammar."
            )

        return cls.from_lark(lark=grammar.ll_grammar(), name=name, description=description, callable=callable)

    def to_openai_style(self) -> dict[str, Any]:
        if isinstance(self.tool, FunctionTool):
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.tool.get_schema(),
                    "strict": True,
                },
            }
        elif isinstance(self.tool, CustomTool):
            return {
                "type": "custom",
                "custom": {
                    "name": self.name,
                    "description": self.description,
                    "format": {
                        "type": "grammar",
                        "grammar": {
                            "syntax": self.tool.format.syntax,
                            "definition": self.tool.format.definition,
                        },
                    },
                },
            }
        else:
            raise TypeError(f"Unsupported tool type: {type(self.tool)}. Expected FunctionTool or CustomTool.")

    def with_name(self, name: str) -> "Tool":
        if self.name == name:
            return self
        new_self = self.model_copy()
        new_self.name = name
        return new_self
