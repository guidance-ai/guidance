from typing import Any, Callable, Mapping, Optional, Sequence, Union
from json import dumps as json_dumps

import guidance
from guidance.library import char_range, one_or_more, optional, zero_or_more

from .._grammar import GrammarFunction, select

def _to_compact_json(target: Any) -> str:
    # See 'Compact Encoding':
    # https://docs.python.org/3/library/json.html
    # Since this is ultimately about the generated
    # output, we don't need to worry about pretty printing
    # and whitespace
    return json_dumps(target, separators=(",", ":"))


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
    definitions: Mapping[str, Callable[[], GrammarFunction]]
):
    lm += "{"
    if properties:
        lm += _process_properties(
            properties=properties,
            definitions=definitions
        )
    if properties and additional_properties:
        lm += optional(
            ','
            + _process_additional_properties(
                additional_properties=additional_properties,
                definitions=definitions
            )
        )
    elif additional_properties:
        lm += optional(
            _process_additional_properties(
                additional_properties=additional_properties,
                definitions=definitions
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
    item = _gen_json_string() + ':' + _gen_json(json_schema=additional_properties, definitions=definitions)
    return lm + zero_or_more(item + ",") + item


@guidance(stateless=True)
def _gen_json_array(
    lm,
    *,
    item_schema: Mapping[str, Any],
    definitions: Mapping[str, Callable[[], GrammarFunction]],
):
    lm += "["
    lm += optional(
        zero_or_more(_gen_json(json_schema=item_schema, definitions=definitions) + ",")
        + _gen_json(json_schema=item_schema, definitions=definitions)
    )
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
        all_opts.append(
            _to_compact_json(opt)
        )
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
            anyof_list=json_schema[ANYOF_STRING],
            definitions=definitions
        )

    REF_STRING = "$ref"
    if REF_STRING in json_schema:
        return lm + _get_definition(
            reference=json_schema[REF_STRING],
            definitions=definitions
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
                item_schema=json_schema["items"],
                definitions=definitions
            )
        if target_type == "object":
            return lm + _gen_json_object(
                properties=json_schema.get("properties"),
                additional_properties=json_schema.get("additionalProperties"),
                definitions=definitions
            )
        raise ValueError(f"Unsupported type in schema: {target_type}")

    raise ValueError(f"Can't process JSON node: {json_schema}")

@guidance(stateless=True)
def json(lm, json_schema: Mapping[str, Any], name: Optional[str] = None):
    _DEFS_KEY = "$defs"
    definitions = {}
    if _DEFS_KEY in json_schema:
        definitions = _build_definitions(json_schema[_DEFS_KEY])

    return lm + guidance.capture(_gen_json(json_schema, definitions), name=name)


def _build_definitions(
    raw_definitions: Mapping[str, Any]
) -> Mapping[str, Callable[[], GrammarFunction]]:
    definitions = {}

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
    REF_START = "#/$defs/"
    assert reference.startswith(
        REF_START
    ), f"Reference {reference} must start with {REF_START}"

    target_name = reference[len(REF_START) :]
    definition = definitions[target_name]
    return lm + definition()
