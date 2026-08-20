import inspect
from typing import Any, Union

import pydantic


class GenerateJsonSchemaSafe(pydantic.json_schema.GenerateJsonSchema):
    """
    Subclass pydantic's GenerateJsonSchema to catch pydantic schemas that will not
    translate properly to json schemas used for generation.

    In particular, JSON schemas do not offer a way to specify "key type",
    so we need to raise an exception if users attempt to specify non-string
    keys through pydantic. Otherwise, they may get unexpected output from
    model generation.
    """

    def generate_inner(self, schema):
        if schema["type"] == "dict":
            # A dict with no key-type annotation (bare `dict`, `Dict[Any, Any]`, etc.)
            # has a `keys_schema` of `{"type": "any"}`. Since JSON object keys are
            # always strings, this maps to a valid, unconstrained JSON object and
            # should not be rejected -- only genuinely non-string key types
            # (e.g. `int`, `float`) should be.
            key_type = schema.get("keys_schema", {}).get("type", "any")
            if key_type not in ("str", "any"):
                raise TypeError(f"JSON does not support non-string keys, got type {key_type}")
        return super().generate_inner(schema)


def pydantic_to_json_schema(schema: Union[type["pydantic.BaseModel"], "pydantic.TypeAdapter[Any]"]) -> dict[str, Any]:
    if inspect.isclass(schema) and issubclass(schema, pydantic.BaseModel):
        return schema.model_json_schema(schema_generator=GenerateJsonSchemaSafe)
    if isinstance(schema, pydantic.TypeAdapter):
        return schema.json_schema(schema_generator=GenerateJsonSchemaSafe)
    raise TypeError(f"Cannot generate json schema from type {type(schema)}")
