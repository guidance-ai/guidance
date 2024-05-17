from json import dumps as json_dumps
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Union,
    Type,
    TYPE_CHECKING,
)

try:
    import jsonschema
    import pydantic
except ImportError:
    if TYPE_CHECKING:
        raise

from .._guidance import guidance
from ..library import char_range, one_or_more, optional, zero_or_more

from .._grammar import GrammarFunction, select, capture, with_temperature
from ._pydantic import pydantic_to_json_schema


def _to_compact_json(target: Any) -> str:
    # See 'Compact Encoding':
    # https://docs.python.org/3/library/json.html
    # Since this is ultimately about the generated
    # output, we don't need to worry about pretty printing
    # and whitespace
    return json_dumps(target, separators=(",", ":"))


_DEFS_KEYS = ["$defs", "definitions"]


@guidance(stateless=True)
def _gen_json_int(lm):
    pos_nonzero = char_range("1", "9") + zero_or_more(char_range("0", "9"))
    return lm + optional("-") + select(["0", pos_nonzero])


@guidance(stateless=True)
def _gen_json_number(lm):
    mantissa_int = _gen_json_int()
    mantissa_frac = "." + one_or_more(char_range("0", "9"))
    exponent = "e" + select(["", "+", "-"]) + one_or_more(char_range("0", "9"))

    return lm + mantissa_int + optional(mantissa_frac) + optional(exponent)


@guidance(stateless=True)
def _gen_json_string(lm):
    string_chars = select(
        [
            char_range("a", "z"),
            char_range("A", "Z"),
            char_range("0", "9"),
            *[c for c in "-_' ,.!?/[]{}():;"],
            "\\n",
            "\\t",
            "\\\\",
        ],
        recurse=True,
    )
    return lm + '"' + string_chars + '"'


@guidance(stateless=True)
def _gen_json_object(
    lm,
    *,
    properties: Union[Mapping[str, Any], None],
    additional_properties: Union[Mapping[str, Any], None],
    definitions: Mapping[str, Callable[[], GrammarFunction]],
):
    lm += "{"
    if properties:
        lm += _process_properties(properties=properties, definitions=definitions)
    if properties and additional_properties:
        lm += optional(
            ","
            + _process_additional_properties(
                additional_properties=additional_properties, definitions=definitions
            )
        )
    elif additional_properties:
        lm += optional(
            _process_additional_properties(
                additional_properties=additional_properties, definitions=definitions
            )
        )
    lm += "}"
    return lm


@guidance(stateless=True)
def _process_properties(
    lm,
    *,
    properties: Mapping[str, Any],
    definitions: Mapping[str, Callable[[], GrammarFunction]],
):
    properties_added = 0
    for name, property_schema in properties.items():
        lm += '"' + name + '"'

        lm += ":"
        lm += _gen_json(
            json_schema=property_schema,
            definitions=definitions,
        )
        properties_added += 1
        if properties_added < len(properties):
            lm += ","
    return lm


@guidance(stateless=True)
def _process_additional_properties(
    lm,
    *,
    additional_properties: Mapping[str, Any],
    definitions: Mapping[str, Callable[[], GrammarFunction]],
):
    item = (
        _gen_json_string()
        + ":"
        + _gen_json(json_schema=additional_properties, definitions=definitions)
    )
    return lm + zero_or_more(item + ",") + item


@guidance(stateless=True)
def _gen_json_array(
    lm,
    *,
    prefix_items_schema: Optional[Sequence[Mapping[str, Any]]],
    item_schema: Optional[Mapping[str, Any]],
    min_items: int,
    max_items: Optional[int],
    definitions: Mapping[str, Callable[[], GrammarFunction]],
):
    if prefix_items_schema is None:
        prefix_items_schema = []

    if len(prefix_items_schema) < min_items and item_schema is None:
        raise ValueError(
            "No items schema provided, but prefixItems has too few elements "
            f"({len(prefix_items_schema)}) to satisfy minItems ({min_items})"
        )

    if max_items is not None and max_items < min_items:
        raise ValueError(
            f"maxItems ({max_items}) can't be less than minItems ({min_items})"
        )

    required_items = []
    optional_items = []

    # If max_items is None, we can add an infinite tail of items later
    n_to_add = (
        max(len(prefix_items_schema), min_items) if max_items is None else max_items
    )
    for i in range(n_to_add):
        if i < len(prefix_items_schema):
            schema = prefix_items_schema[i]
        elif item_schema is not None:
            schema = item_schema
        else:
            assert i >= min_items
            break

        item = _gen_json(json_schema=schema, definitions=definitions)

        if i < min_items:
            required_items.append(item)
        else:
            optional_items.append(item)

    if max_items is None and item_schema is not None:
        # Add an infinite tail of items
        item = _gen_json(json_schema=item_schema, definitions=definitions)
        optional_items.append(item + zero_or_more("," + item))

    lm += "["

    if required_items:
        first, *rest = required_items
        lm += first
        for item in rest:
            lm += "," + item

    if optional_items:
        # This is a bit subtle and would not be required if not for prefixItems -- the previous
        # must be present before the next one may be added, meaning we have nested optionals:
        # (first optional(,second optional(,third (optional(,...)))))
        first, *rest = optional_items
        tail = ""
        for item in reversed(rest):
            tail = optional("," + item + tail)
        tail = first + tail

        if required_items:
            lm += optional("," + tail)
        else:
            lm += optional(tail)

    lm += "]"
    return lm


@guidance(stateless=True)
def _process_anyOf(
    lm,
    *,
    anyof_list: Sequence[Mapping[str, Any]],
    definitions: Mapping[str, Callable[[], GrammarFunction]],
):
    options = [
        _gen_json(json_schema=item, definitions=definitions) for item in anyof_list
    ]
    return lm + select(options)


@guidance(stateless=True)
def _process_enum(lm, *, options: Sequence[Mapping[str, Any]]):
    # options will come in as python objects, so we need to convert to (compact) JSON
    all_opts = []
    for opt in options:
        all_opts.append(_to_compact_json(opt))
    return lm + select(options=all_opts)


@guidance(stateless=True)
def _gen_json(
    lm,
    json_schema: Mapping[str, Any],
    definitions: Mapping[str, Callable[[], GrammarFunction]],
):
    ANYOF_STRING = "anyOf"
    if ANYOF_STRING in json_schema:
        return lm + _process_anyOf(
            anyof_list=json_schema[ANYOF_STRING], definitions=definitions
        )

    ALLOF_STRING = "allOf"
    if ALLOF_STRING in json_schema:
        allof_list = json_schema[ALLOF_STRING]
        if len(allof_list) != 1:
            raise ValueError("Only support allOf with exactly one item")
        return lm + _gen_json(allof_list[0], definitions)

    REF_STRING = "$ref"
    if REF_STRING in json_schema:
        return lm + _get_definition(
            reference=json_schema[REF_STRING], definitions=definitions
        )

    ENUM_STRING = "enum"
    if ENUM_STRING in json_schema:
        return lm + _process_enum(options=json_schema["enum"])

    TYPE_STRING = "type"
    if TYPE_STRING in json_schema:
        target_type = json_schema["type"]
        if target_type == "null":
            return lm + "null"
        if target_type == "boolean":
            return lm + select(["true", "false"])
        if target_type == "integer":
            return lm + _gen_json_int()
        if target_type == "number":
            return lm + _gen_json_number()
        if target_type == "string":
            return lm + _gen_json_string()
        if target_type == "array":
            return lm + _gen_json_array(
                prefix_items_schema=json_schema.get("prefixItems"),
                item_schema=json_schema.get("items"),
                min_items=json_schema.get("minItems", 0),
                max_items=json_schema.get("maxItems"),
                definitions=definitions,
            )
        if target_type == "object":
            return lm + _gen_json_object(
                properties=json_schema.get("properties"),
                additional_properties=json_schema.get("additionalProperties"),
                definitions=definitions,
            )
        raise ValueError(f"Unsupported type in schema: {target_type}")

    raise ValueError(f"Can't process JSON node: {json_schema}")


@guidance(stateless=True)
def json(
    lm,
    name: Optional[str] = None,
    *,
    schema: Union[
        Mapping[str, Any],
        Type["pydantic.BaseModel"],
        "pydantic.TypeAdapter",
    ],
    temperature: float = 0.0,
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

    schema : Union[Mapping[str, Any], Type[pydantic.BaseModel], pydantic.TypeAdapter]
        One of:
            - A JSON schema object. This is a JSON schema string which has been passed to ``json.loads()``
            - A subclass of ``pydantic.BaseModel``
            - An instance of ``pydantic.TypeAdapter``
    """
    if isinstance(schema, Mapping):
        # Raises jsonschema.exceptions.SchemaError or ValueError
        # if schema is not valid
        jsonschema.validators.Draft202012Validator.check_schema(schema)
    else:
        schema = pydantic_to_json_schema(schema)

    definitions: Mapping[str, Callable[[], GrammarFunction]] = {}
    for dk in _DEFS_KEYS:
        if dk in schema:
            assert len(definitions) == 0, "Found duplicate definitions"
            definitions = _build_definitions(schema[dk])

    return lm + capture(
        with_temperature(_gen_json(schema, definitions), temperature=temperature),
        name=name,
    )


def _build_definitions(
    raw_definitions: Mapping[str, Any]
) -> Mapping[str, Callable[[], GrammarFunction]]:
    definitions: Dict[str, Callable[[], GrammarFunction]] = {}

    def build_definition(
        json_schema: Mapping[str, Any]
    ) -> Callable[[], GrammarFunction]:
        @guidance(stateless=True, dedent=False)
        def closure(lm):
            return lm + _gen_json(json_schema=json_schema, definitions=definitions)

        return closure

    definitions = {
        ref: build_definition(schema) for ref, schema in raw_definitions.items()
    }
    return definitions


@guidance(stateless=True)
def _get_definition(
    lm,
    *,
    reference: str,
    definitions: Mapping[str, Callable[[], GrammarFunction]],
):
    assert definitions is not None
    target_definition = None
    for dk in _DEFS_KEYS:
        ref_start = f"#/{dk}/"
        if reference.startswith(ref_start):
            target_name = reference[len(ref_start) :]
            target_definition = definitions[target_name]

    assert target_definition is not None
    return lm + target_definition()
