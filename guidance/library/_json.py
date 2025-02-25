import warnings
from json import dumps as json_dumps
from json import loads as json_loads
from typing import Any, Mapping, Optional, Type, Union, cast

import pydantic
from llguidance import JsonCompiler

from .._ast import JsonNode
from .._grammar import capture, token_limit, with_temperature
from ._pydantic import pydantic_to_json_schema

JSONSchema = Mapping[str, Any]


def json(
    name: Optional[str] = None,
    *,
    schema: Union[
        None,
        str,
        JSONSchema,
        Type[pydantic.BaseModel],
        pydantic.TypeAdapter,
    ] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    separators: Optional[tuple[str, str]] = None,
    whitespace_flexible: bool = False,
):
    """Generate valid JSON according to the supplied JSON schema or `pydantic` model.

    Not all parts of `JSON schema <https://json-schema.org/>`_ are supported. Indeed some parts
    (such as bounds on numbers) cannot really be supported in the context of LLM generation.

    Using a JSON schema:

        >>> schema = ''{ "type": "object", "properties": { "a" : {"type": "integer"} } }'
        >>> schema_obj = json.loads(schema)
        >>> lm += json(name="generated_object", schema=schema_obj)
        >>> print(json.loads(lm["generated_object"]))
        { 'a' : 2 }

    Using a ``pydantic.BaseModel``:

        >>> class Schema(BaseModel):
        ...     b: bool
        >>> lm += json(name="generated_object", schema=Schema)
        >>> print(json.loads(lm["generated_object"]))
        { 'b' : False }

    Using a ``pydantic.TypeAdapter``:

        >>> schema = TypeAdapter(list[int])
        >>> lm += json(name="generated_object", schema=schema)
        >>> print(json.loads(lm["generated_object"]))
        [1, 2, 3]

    Parameters
    ----------

    name : str or None
        If this is not None then the the results of the generation will be saved as a variable on
        the Model object (so you can access the result as ``lm["var_name"]``).

    schema : Union[None, Mapping[str, Any], Type[pydantic.BaseModel], pydantic.TypeAdapter]
        One of:
            - None, in which case any valid JSON will be generated
            - A string representing a JSON schema which will be parsed using ``json.loads()``
            - A JSON schema object. This is a JSON schema string which has been passed to ``json.loads()``
            - A subclass of ``pydantic.BaseModel``
            - An instance of ``pydantic.TypeAdapter``
    """
    if schema is None:
        # Default schema is empty, "anything goes" schema
        # TODO: consider default being `{"type": "object"}`
        schema = {}
    elif isinstance(schema, (Mapping, bool, str)):
        if isinstance(schema, str):
            schema = cast(JSONSchema, json_loads(schema))
    elif isinstance(schema, pydantic.TypeAdapter) or (
        isinstance(schema, type) and issubclass(schema, pydantic.BaseModel)
    ):
        schema = pydantic_to_json_schema(schema)
    else:
        raise TypeError(f"Unsupported schema type: {type(schema)}")

    coerce_one_of = False
    # TODO: decide whether or not to keep this -- it lets us double check that llguidance can handle the schema (which isn't necessarily)
    # what we want, as llguidance may or may not be the backend we are using. That being said, it's sort of nice to get an exception when
    # you call `json` instead of waiting for generation to fail.
    VALIDATE = True
    if VALIDATE:
        schema_string = json_dumps(schema)
        try:
            compiler = JsonCompiler(
                separators=separators,
                whitespace_flexible=whitespace_flexible,
                coerce_one_of=False,
            )
            compiler.compile(schema_string)
        except ValueError as e:
            if (
                e.args[0]
                == "oneOf constraints are not supported. Enable 'coerce_one_of' option to approximate oneOf with anyOf"
            ):
                warnings.warn(
                    "oneOf not fully supported, falling back to anyOf. This may cause validation errors in some cases."
                )
                compiler = JsonCompiler(
                    separators=separators,
                    whitespace_flexible=whitespace_flexible,
                    coerce_one_of=True,
                )
                compiler.compile(schema_string)
                coerce_one_of = True
            else:
                raise

    # TODO: separators, whitespace_flexible
    node = JsonNode(
        name=name or "json",
        schema=schema,
        # TODO:
        # separators = separators,
        # whitespace_flexible = whitespace_flexible,
        # coerce_one_of = coerce_one_of,
    )
    if temperature is not None:
        node = with_temperature(node, temperature)
    if max_tokens is not None:
        node = token_limit(node, max_tokens)
    if name is not None:
        node = capture(node, name)
    return node
