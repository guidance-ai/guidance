from typing import Type, Union, Any
from pydantic import BaseModel, TypeAdapter
from pydantic.json_schema import GenerateJsonSchema

import guidance
from ._json import json as gen_json


class GenerateJsonSchemaSafe(GenerateJsonSchema):
    """
    Subclass pydantic's GenerateJsonSchema to catch pydantic schemas that will not
    translate properly to json schemas used for generation
    """

    def generate_inner(self, schema):
        if schema["type"] == "dict":
            key_type = schema["keys_schema"]["type"]
            if key_type != "string":
                raise TypeError(
                    f"JSON does not support non-string keys, got type {key_type}"
                )
        return super().generate_inner(schema)


def generate_json_schema(
    pydantic_model: Union[Type[BaseModel], Type[TypeAdapter]]
) -> dict[str, Any]:
    if isinstance(pydantic_model, BaseModel):
        return pydantic_model.model_json_schema(schema_generator=GenerateJsonSchemaSafe)
    if isinstance(pydantic_model, TypeAdapter):
        return pydantic_model.json_schema(schema_generator=GenerateJsonSchemaSafe)
    raise TypeError(f"Cannot generate json schema from type {type(pydantic_model)}")


@guidance(stateless=True)
def pydantic(lm, pydantic_model: Union[Type[BaseModel], Type[TypeAdapter]]):
    # TODO: add a Literal["json", "python"] "mode" argument
    # to support various genneration modes?
    schema = generate_json_schema(pydantic_model)
    return lm + gen_json(schema)
