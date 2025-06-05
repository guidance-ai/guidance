from json import loads as json_loads
from typing import Any, Mapping, Optional, Type, Union

import pydantic

from .._ast import JsonNode, LLGJsonCompileOptions
from .._grammar import capture, token_limit, with_temperature
from ._pydantic import pydantic_to_json_schema

JSONSchema = Union[bool, Mapping[str, Any]]


def json(
    name: Optional[str] = None,
    *,
    schema: Union[
        None,
        str,
        JSONSchema,
        Type[pydantic.BaseModel],
        pydantic.TypeAdapter[Any],
    ] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    separators: Optional[tuple[str, str]] = None,
    whitespace_flexible: bool = False,
    whitespace_pattern: Optional[str] = None,
    lenient: bool = False,
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
    if schema is False:
        raise ValueError("Unsatisfiable schema: schema is false")
    elif schema is True:
        # Any valid JSON is acceptable
        schema = {}
    elif isinstance(schema, pydantic.TypeAdapter) or (
        isinstance(schema, type) and issubclass(schema, pydantic.BaseModel)
    ):
        schema = pydantic_to_json_schema(schema)
    elif isinstance(schema, str):
        from_str = json_loads(schema)
        if not isinstance(from_str, dict):
            raise ValueError("JSON schema string must be a JSON object (i.e. a dictionary)")
        schema = from_str

    if whitespace_pattern is not None:
        whitespace_flexible = True

    if separators is None:
        if whitespace_flexible or whitespace_pattern:
            separators = (",", ":")
        else:
            separators = (", ", ": ")

    if len(separators) != 2:
        raise ValueError("separators must be a tuple of (item_separator, key_separator)")
    item_separator, key_separator = separators

    llg_options = LLGJsonCompileOptions(
        whitespace_flexible=whitespace_flexible,
        whitespace_pattern=whitespace_pattern,
        item_separator=item_separator,
        key_separator=key_separator,
        coerce_one_of=True,
        lenient=lenient,
    )

    if isinstance(schema, Mapping):
        schema = dict(schema)
    elif schema is not None:
        raise TypeError(
            f"Invalid schema type: {type(schema)}. Expected None, a boolean, a JSON schema object, a pydantic model, or a pydantic TypeAdapter."
        )

    node = JsonNode(
        schema=schema,
        llg_options=llg_options,
    )

    VALIDATE = True
    if VALIDATE:
        # TODO: decide whether or not to keep this -- it lets us double check that llguidance can handle the schema which isn't necessarily
        # what we want, as llguidance may or may not be the backend we are using. That being said, it's sort of nice to get an exception when
        # you call `json` instead of waiting for generation to fail.
        node._llguidance_validate()

    if temperature is not None:
        node = with_temperature(node, temperature)
    if max_tokens is not None:
        node = token_limit(node, max_tokens)
    if name is not None:
        node = capture(node, name)
    return node
