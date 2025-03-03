import inspect
from typing import Any, Dict, Type, Union

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
            key_type = schema["keys_schema"]["type"]
            if key_type != "str":
                raise TypeError(
                    f"JSON does not support non-string keys, got type {key_type}"
                )
        return super().generate_inner(schema)


def pydantic_to_json_schema(
    schema: Union[Type["pydantic.BaseModel"], "pydantic.TypeAdapter"]
) -> Dict[str, Any]:
    if inspect.isclass(schema) and issubclass(schema, pydantic.BaseModel):
        return schema.model_json_schema(schema_generator=GenerateJsonSchemaSafe)
    if isinstance(schema, pydantic.TypeAdapter):
        return schema.json_schema(schema_generator=GenerateJsonSchemaSafe)
    raise TypeError(f"Cannot generate json schema from type {type(schema)}")
