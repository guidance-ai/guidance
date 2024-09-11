from json import dumps as json_dumps
from enum import Enum
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
import warnings

try:
    import jsonschema
    import pydantic
except ImportError:
    if TYPE_CHECKING:
        raise

from .._guidance import guidance
from ..library import char_range, gen, one_or_more, optional, sequence

from .._grammar import GrammarFunction, select, capture, with_temperature
from ._pydantic import pydantic_to_json_schema
from ._subgrammar import lexeme, subgrammar

JSONSchema = Union[bool, Mapping[str, Any]]

def _to_compact_json(target: Any) -> str:
    # See 'Compact Encoding':
    # https://docs.python.org/3/library/json.html
    # Since this is ultimately about the generated
    # output, we don't need to worry about pretty printing
    # and whitespace
    return json_dumps(target, separators=(",", ":"))

class JSONType(str, Enum):
    NULL = "null"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NUMBER = "number"
    STRING = "string"
    ARRAY = "array"
    OBJECT = "object"

class Keyword(str, Enum):
    ANYOF = "anyOf"
    ALLOF = "allOf" # Note: Partial support. Only supports exactly one item.
    ONEOF = "oneOf" # Note: Partial support. This is converted to anyOf.
    REF = "$ref"
    CONST = "const"
    ENUM = "enum"
    TYPE = "type"

class StringKeywords(str, Enum):
    PATTERN = "pattern"
    MIN_LENGTH = "minLength"
    MAX_LENGTH = "maxLength"

class ArrayKeywords(str, Enum):
    PREFIX_ITEMS = "prefixItems"
    ITEMS = "items"
    MIN_ITEMS = "minItems"
    MAX_ITEMS = "maxItems"

class ObjectKeywords(str, Enum):
    PROPERTIES = "properties"
    ADDITIONAL_PROPERTIES = "additionalProperties"
    REQUIRED = "required"

TYPE_SPECIFIC_KEYWORDS = {
    JSONType.STRING: StringKeywords,
    JSONType.ARRAY: ArrayKeywords,
    JSONType.OBJECT: ObjectKeywords,
}

DEFS_KEYS = {"$defs", "definitions"}

IGNORED_KEYS = {
    "$schema",
    "$id",
    "id",
    "$comment",
    "title",
    "description",
    "default",
    "examples",
}

# discriminator is part of OpenAPI 3.1, not JSON Schema itself
# https://json-schema.org/blog/posts/validating-openapi-and-json-schema
# TODO: While ignoring this key shouldn't lead to invalid outputs, forcing
# the model to choose the value of the marked field before other fields
# are generated (statefully or statelessly) would reduce grammar ambiguity
# and possibly improve quality.
IGNORED_KEYS.add("discriminator")

WHITESPACE = {b" ", b"\t", b"\n", b"\r"}

def validate_json_node_keys(node: Mapping[str, Any]):
    keys = set(node.keys())
    valid_keys = set(Keyword) | IGNORED_KEYS | DEFS_KEYS
    if Keyword.TYPE in node and (tp:=node[Keyword.TYPE]) in TYPE_SPECIFIC_KEYWORDS:
        valid_keys |= set(TYPE_SPECIFIC_KEYWORDS[tp])
    invalid_keys = keys - valid_keys
    if invalid_keys:
        raise ValueError(
            f"JSON schema had keys that could not be processed: {invalid_keys}" f"\nSchema: {node}"
        )


@guidance(stateless=True)
def _gen_json_int(lm):
    return lm + lexeme(r"-?(?:0|[1-9][0-9]*)", contextual=True)


@guidance(stateless=True)
def _gen_json_number(lm):
    return lm + select([
        _gen_json_int(),
        lexeme(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)", contextual=True),
        lexeme(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)", contextual=True),
    ])


@guidance(stateless=True)
def _gen_json_string(
    lm,
    min_length: int = 0,
    max_length: Union[int, None] = None,
    regex: Union[str, None] = None,
):
    if regex is None:
        range_expr = f"{{{min_length},{max_length}}}" if max_length is not None else f"{{{min_length},}}"
        regex = f"(?s:.{range_expr})"
    else:
        if min_length > 0 or max_length is not None:
            msg = (
                "If a pattern is specified for a JSON "
                "string, minLength and maxLength must be "
                "left unspecified."
            )
            raise ValueError(msg)
    return lm + lexeme(regex, contextual=True, json_string=True)


@guidance(stateless=True)
def _gen_json_object(
    lm,
    *,
    properties: Mapping[str, JSONSchema],
    additional_properties: JSONSchema,
    required: Sequence[str],
    definitions: Mapping[str, Callable[[], GrammarFunction]],
):
    if any(k not in properties for k in required):
        raise ValueError(f"Required properties not in properties: {set(required) - set(properties)}")

    grammars = tuple(f'"{name}":' + _gen_json(json_schema=schema, definitions=definitions) for name, schema in properties.items())
    required_items = tuple(name in required for name in properties)

    if additional_properties is not False:
        if additional_properties is True:
            # True means that anything goes
            additional_properties = {}
        additional_item_grammar =  _gen_json_string() + ':' + _gen_json(json_schema=additional_properties, definitions=definitions)
        additional_items_grammar = sequence(additional_item_grammar + ',') + additional_item_grammar
        grammars += (additional_items_grammar,)
        required_items += (False,)

    return lm + "{" + _gen_list(
        elements = grammars,
        required = required_items,
    ) + "}"

@guidance(stateless=True, cache=True)
def _gen_list(lm, *, elements: tuple[GrammarFunction, ...], required: tuple[bool, ...], prefixed: bool = False):
    if not elements:
        return lm

    elem, elements = elements[0], elements[1:]
    is_required, required = required[0], required[1:]

    if prefixed:
        if is_required:
            # If we know we have preceeding elements, we can safely just add a (',' + e)
            return lm + (',' + elem + _gen_list(elements=elements, required=required, prefixed=True))
        # If we know we have preceeding elements, we can safely just add an optional(',' + e)
        return lm + (optional(',' + elem) + _gen_list(elements=elements, required=required, prefixed=True))
    if is_required:
        # No preceding elements, and our element is required, so we just add the element
        return lm + (elem + _gen_list(elements=elements, required=required, prefixed=True))

    # No preceding elements, and our element is optional, so we add a select between the two options.
    # The first option is the recursive call with no preceding elements, the second is the recursive call
    # with the current element as a prefix.
    return lm + select([
        _gen_list(elements=elements, required=required, prefixed=False),
        elem + _gen_list(elements=elements, required=required, prefixed=True)
    ])


@guidance(stateless=True)
def _gen_json_array(
    lm,
    *,
    prefix_items_schema: Sequence[JSONSchema],
    item_schema: JSONSchema,
    min_items: int,
    max_items: Optional[int],
    definitions: Mapping[str, Callable[[], GrammarFunction]],
):
    if len(prefix_items_schema) < min_items and item_schema is False:
        raise ValueError(
            f"PrefixItems has too few elements ({len(prefix_items_schema)}) to"
            f" satisfy minItems ({min_items}) but no extra items were allowed"
        )

    if max_items is not None and max_items < min_items:
        raise ValueError(f"maxItems ({max_items}) can't be less than minItems ({min_items})")

    required_items = []
    optional_items = []

    # If max_items is None, we can add an infinite tail of items later
    n_to_add = max(len(prefix_items_schema), min_items) if max_items is None else max_items
    for i in range(n_to_add):
        if i < len(prefix_items_schema):
            schema = prefix_items_schema[i]
        elif item_schema is not False:
            schema = item_schema
        else:
            assert i >= min_items
            break

        item = _gen_json(json_schema=schema, definitions=definitions)

        if i < min_items:
            required_items.append(item)
        else:
            optional_items.append(item)

    if max_items is None and item_schema is not False:
        # Add an infinite tail of items
        item = _gen_json(json_schema=item_schema, definitions=definitions)
        optional_items.append(item + sequence("," + item))

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
        tail: Union[str, GrammarFunction] = ""
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
    anyof_list: Sequence[JSONSchema],
    definitions: Mapping[str, Callable[[], GrammarFunction]],
):
    options = [_gen_json(json_schema=item, definitions=definitions) for item in anyof_list]
    return lm + select(options)


@guidance(stateless=True)
def _process_enum(lm, *, options: Sequence[Mapping[str, Any]]):
    # TODO: can we support a whitespace-flexible version of this?
    all_opts = []
    for opt in options:
        all_opts.append(_to_compact_json(opt))
    return lm + select(options=all_opts)


@guidance(stateless=True)
def _gen_json_any(lm):
    return lm + select(
        [
            _gen_json(json_schema={"type": "null"}, definitions={}),
            _gen_json(json_schema={"type": "boolean"}, definitions={}),
            _gen_json(json_schema={"type": "integer"}, definitions={}),
            _gen_json(json_schema={"type": "number"}, definitions={}),
            _gen_json(json_schema={"type": "string"}, definitions={}),
            # Recursive cases
            _gen_json(
                json_schema={
                    "type": "array",
                    "items": True,
                },
                definitions={},
            ),
            _gen_json(
                json_schema={
                    "type": "object",
                    "additionalProperties": True,
                },
                definitions={},
            ),
        ]
    )


@guidance(stateless=True)
def _gen_json(
    lm,
    json_schema: JSONSchema,
    definitions: Mapping[str, Callable[[], GrammarFunction]],
):
    if json_schema is True:
        json_schema = {}
    elif json_schema is False:
        raise ValueError("No valid JSON can be generated from a schema of `False`")

    validate_json_node_keys(json_schema)

    if Keyword.ANYOF in json_schema:
        return lm + _process_anyOf(anyof_list=json_schema[Keyword.ANYOF], definitions=definitions)

    if Keyword.ALLOF in json_schema:
        allof_list = json_schema[Keyword.ALLOF]
        if len(allof_list) != 1:
            raise ValueError("Only support allOf with exactly one item")
        return lm + _gen_json(allof_list[0], definitions)

    if Keyword.ONEOF in json_schema:
        oneof_list = json_schema[Keyword.ONEOF]
        if len(oneof_list) == 1:
            return lm + _gen_json(oneof_list[0], definitions)
        warnings.warn("oneOf not fully supported, falling back to anyOf. This may cause validation errors in some cases.")
        return lm + _process_anyOf(anyof_list=oneof_list, definitions=definitions)

    if Keyword.REF in json_schema:
        return lm + _get_definition(reference=json_schema[Keyword.REF], definitions=definitions)

    if Keyword.CONST in json_schema:
        # TODO: can we support a whitespace-flexible version of this?
        return lm + _to_compact_json(json_schema[Keyword.CONST])

    if Keyword.ENUM in json_schema:
        return lm + _process_enum(options=json_schema[Keyword.ENUM])

    if Keyword.TYPE in json_schema:
        target_type = json_schema[Keyword.TYPE]
        if target_type == JSONType.NULL:
            return lm + "null"
        if target_type == JSONType.BOOLEAN:
            return lm + select(["true", "false"])
        if target_type == JSONType.INTEGER:
            return lm + _gen_json_int()
        if target_type == JSONType.NUMBER:
            return lm + _gen_json_number()
        if target_type == JSONType.STRING:
            return lm + _gen_json_string(
                regex=json_schema.get(StringKeywords.PATTERN, None),
                min_length=json_schema.get(StringKeywords.MIN_LENGTH, 0),
                max_length=json_schema.get(StringKeywords.MAX_LENGTH, None),
            )
        if target_type == JSONType.ARRAY:
            return lm + _gen_json_array(
                prefix_items_schema=json_schema.get(ArrayKeywords.PREFIX_ITEMS, []),
                item_schema=json_schema.get(ArrayKeywords.ITEMS, True),
                min_items=json_schema.get(ArrayKeywords.MIN_ITEMS, 0),
                max_items=json_schema.get(ArrayKeywords.MAX_ITEMS, None),
                definitions=definitions,
            )
        if target_type == JSONType.OBJECT:
            return lm + _gen_json_object(
                properties=json_schema.get(ObjectKeywords.PROPERTIES, {}),
                additional_properties=json_schema.get(ObjectKeywords.ADDITIONAL_PROPERTIES, True),
                required=json_schema.get(ObjectKeywords.REQUIRED, set()),
                definitions=definitions,
            )
        raise ValueError(f"Unsupported type in schema: {target_type}")

    return lm + _gen_json_any()


@guidance(stateless=True)
def json(
    lm,
    name: Optional[str] = None,
    *,
    schema: Union[
        None,
        JSONSchema,
        Type["pydantic.BaseModel"],
        "pydantic.TypeAdapter",
    ] = None,
    compact: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 100000000,
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
            - A JSON schema object. This is a JSON schema string which has been passed to ``json.loads()``
            - A subclass of ``pydantic.BaseModel``
            - An instance of ``pydantic.TypeAdapter``

    compact : bool
        If True, the generated JSON will be forced to be compact (no whitespace).
        If False, output will be whitespace-flexible (i.e. decided by the model).
    """
    if schema is None:
        # Default schema is empty, "anything goes" schema
        # TODO: consider default being `{"type": "object"}`
        schema = {}
    elif isinstance(schema, (Mapping, bool)):
        # Raises jsonschema.exceptions.SchemaError or ValueError
        # if schema is not valid
        jsonschema.validators.Draft202012Validator.check_schema(schema)
    elif isinstance(schema, pydantic.TypeAdapter) or (isinstance(schema, type) and issubclass(schema, pydantic.BaseModel)):
        schema = pydantic_to_json_schema(schema)
    else:
        raise TypeError(f"Unsupported schema type: {type(schema)}")

    definitions: Mapping[str, Callable[[], GrammarFunction]] = {}
    if isinstance(schema, Mapping):
        for dk in DEFS_KEYS:
            if dk in schema:
                assert len(definitions) == 0, "Found duplicate definitions"
                definitions = _build_definitions(schema[dk])

    return lm + with_temperature(
        subgrammar(
            name,
            body=_gen_json(json_schema=schema, definitions=definitions),
            skip_regex=(
                None if compact
                else r"[\x20\x0A\x0D\x09]+"
            ),
            no_initial_skip=True,
            max_tokens=max_tokens,
        ),
        temperature=temperature,
    )


def _build_definitions(
    raw_definitions: Mapping[str, JSONSchema]
) -> Mapping[str, Callable[[], GrammarFunction]]:
    definitions: Dict[str, Callable[[], GrammarFunction]] = {}

    def build_definition(json_schema: JSONSchema) -> Callable[[], GrammarFunction]:
        @guidance(stateless=True, dedent=False, cache=True)
        def closure(lm):
            return lm + _gen_json(json_schema=json_schema, definitions=definitions)

        return closure

    definitions = {ref: build_definition(schema) for ref, schema in raw_definitions.items()}
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
    for dk in DEFS_KEYS:
        ref_start = f"#/{dk}/"
        if reference.startswith(ref_start):
            target_name = reference[len(ref_start) :]
            target_definition = definitions[target_name]

    assert target_definition is not None
    return lm + target_definition()
