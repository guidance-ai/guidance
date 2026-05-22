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

    Additionally, when a model uses the default ``extra="ignore"`` configuration,
    this generator adds ``additionalProperties: false`` to the JSON schema so that
    LLMs do not waste tokens generating fields that would be silently discarded.
    Models with ``extra="allow"`` retain flexible schemas, while ``extra="forbid"``
    is already handled by pydantic itself.
    """

    def generate_inner(self, schema):
        if schema["type"] == "dict":
            key_type = schema["keys_schema"]["type"]
            if key_type != "str":
                raise TypeError(f"JSON does not support non-string keys, got type {key_type}")
        return super().generate_inner(schema)

    def model_schema(self, schema):
        result = super().model_schema(schema)
        config = schema.get("config", {})
        # extra_fields_behavior is "allow", "forbid", or "ignore" when explicitly configured,
        # and None when the model uses the default (which is "ignore").
        # Pydantic already sets additionalProperties: false for extra="forbid".
        # We mirror that for extra="ignore" (both explicit and default) so the LLM does
        # not waste tokens generating fields that would otherwise be silently discarded.
        extra_fields_behavior = config.get("extra_fields_behavior") if config else None
        if extra_fields_behavior in (None, "ignore") and "additionalProperties" not in result:
            result["additionalProperties"] = False
        return result


def pydantic_to_json_schema(schema: Union[type["pydantic.BaseModel"], "pydantic.TypeAdapter[Any]"]) -> dict[str, Any]:
    if inspect.isclass(schema) and issubclass(schema, pydantic.BaseModel):
        return schema.model_json_schema(schema_generator=GenerateJsonSchemaSafe)
    if isinstance(schema, pydantic.TypeAdapter):
        return schema.json_schema(schema_generator=GenerateJsonSchemaSafe)
    raise TypeError(f"Cannot generate json schema from type {type(schema)}")
