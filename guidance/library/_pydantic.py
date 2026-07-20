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
            # A dict with no key type (bare ``dict``, ``Dict[Any, ...]``) has an
            # ``"any"`` keys_schema. In JSON that maps to an unconstrained object
            # (JSON object keys are always strings), which is a valid, supported
            # schema, so it must not be rejected as "non-string keys". Only key
            # types that genuinely can't be represented as JSON strings (``int``,
            # ``float``, ...) are rejected.
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
