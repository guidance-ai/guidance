import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Union

from pydantic import BaseModel, ConfigDict, create_model


@dataclass
class Tool:
    callable: Callable
    name: str
    description: str
    schema: Union[type[BaseModel], dict[str, Any]]

    @classmethod
    def from_callable(cls, callable: Callable) -> "Tool":
        signature = inspect.signature(callable)
        args = {}
        for name, param in signature.parameters.items():
            if param.kind not in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }:
                raise ValueError(f"Unsupported parameter kind: {param.kind.description}")
            args[name] = param.annotation if param.annotation is not inspect.Parameter.empty else Any

        return Tool(
            callable=callable,
            name=callable.__name__,
            description=(callable.__doc__ or "").strip(),
            schema=create_model(
                callable.__name__,
                __config__=ConfigDict(extra="forbid"),
                **{name: (annotation, ...) for name, annotation in args.items()},
            ),
        )

    def validate_args(self, args: Union[str, dict[str, Any]]) -> dict[str, Any]:
        """
        Validate the arguments against the tool's schema.
        If the schema is a Pydantic model, it will validate and return the model's dict.
        If it's a dict, it will return the args as is.
        """
        if isinstance(args, str):
            loaded_args = json.loads(args)
        else:
            loaded_args = args
        if isinstance(self.schema, type) and issubclass(self.schema, BaseModel):
            # If the schema is a Pydantic model, validate the args
            return self.schema.model_validate(loaded_args).model_dump()
        elif isinstance(self.schema, dict):
            # If the schema is a dict, we assume it's a JSON schema and return the args as is
            # TODO: use a JSON schema validator?
            return loaded_args
        else:
            raise TypeError(f"Unsupported schema type: {type(self.schema)}. Expected a Pydantic model or a dict.")

    def schema_dump(self) -> dict[str, Any]:
        """
        Dump the schema of the tool.
        If the schema is a Pydantic model, it will return the model's schema.
        If it's a dict, it will return the dict as is.
        """
        if isinstance(self.schema, type) and issubclass(self.schema, BaseModel):
            # If the schema is a Pydantic model, return the model's schema
            return self.schema.model_json_schema()
        elif isinstance(self.schema, dict):
            # If the schema is a dict, we assume it's a JSON schema and return the dict as is
            return self.schema
        else:
            raise TypeError(f"Unsupported schema type: {type(self.schema)}. Expected a Pydantic model or a dict.")
