import builtins
import inspect
from typing import Annotated, Any, Callable, Literal, Optional, TypeAlias, Union

from pydantic import BaseModel, Field

from .._ast import GrammarNode
from .._guidance import GuidanceFunction


class LarkGrammar(BaseModel):
    syntax: Literal["lark"] = "lark"
    lark: str


class RegexGrammar(BaseModel):
    syntax: Literal["regex"] = "regex"
    pattern: str


GrammarDefinition = Annotated[Union[LarkGrammar, RegexGrammar], Field(discriminator="syntax")]


class GrammarFormat(BaseModel):
    type: Literal["grammar"] = "grammar"
    definition: GrammarDefinition


# Placeholder for possible future Union
CustomFormat: TypeAlias = GrammarFormat


class CustomTool(BaseModel):
    type: Literal["custom"] = "custom"
    format: CustomFormat


class FunctionTool(BaseModel):
    type: Literal["function"] = "function"
    parameters: Union[builtins.type[BaseModel], dict[str, Any]]

    @classmethod
    def from_callable(cls, callable: Callable) -> "FunctionTool":
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
            parameters=parameters,
        )

    def schema_dump(self) -> dict[str, Any]:
        """
        Dump the schema of the tool.
        If the schema is a Pydantic model, it will return the model's schema.
        If it's a dict, it will return the dict as is.
        """
        if isinstance(self.parameters, type) and issubclass(self.parameters, BaseModel):
            # If the schema is a Pydantic model, return the model's schema
            return self.parameters.model_json_schema()
        elif isinstance(self.parameters, dict):
            # If the schema is a dict, we assume it's a JSON schema and return the dict as is
            return self.parameters
        else:
            raise TypeError(f"Unsupported schema type: {type(self.parameters)}. Expected a Pydantic model or a dict.")


ToolType = Annotated[Union[FunctionTool, CustomTool], Field(discriminator="type")]


class Tool(BaseModel):
    name: str
    description: str
    tool: ToolType
    callable: Callable

    @classmethod
    def from_callable(
        cls,
        callable: Callable,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "Tool":
        return Tool(
            name=name or callable.__name__,
            description=description or (callable.__doc__ or "").strip(),
            tool=FunctionTool.from_callable(callable),
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
                format=RegexGrammar(pattern=pattern),
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
                format=LarkGrammar(lark=lark),
            ),
            callable=callable,
        )

    @classmethod
    def from_grammar(
        cls,
        grammar: GrammarNode,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        callable: Callable,
    ) -> "Tool":
        if isinstance(grammar, GuidanceFunction):
            raise TypeError(
                "An @guidance-wrapped function was passed to Tool.from_grammar. The function must be called and return a valid grammar."
            )

        return cls.from_lark(lark=grammar.ll_grammar(), name=name, description=description, callable=callable)
