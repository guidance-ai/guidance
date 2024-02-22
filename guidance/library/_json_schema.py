from typing import Any, Mapping, MutableMapping, Sequence, Union

import guidance
from guidance.library import char_range, one_or_more, optional, zero_or_more

from .._grammar import select


def _get_definition(
    reference: str, definitions: Mapping[str, Any]
) -> Mapping[str, any]:
    assert definitions is not None
    REF_START = "#/$defs/"
    assert reference.startswith(
        REF_START
    ), f"Reference {reference} must start with {REF_START}"

    target_name = reference[len(REF_START) :]
    return definitions[target_name]


@guidance(stateless=True)
def _gen_json_int(lm):
    return lm + optional("-") + one_or_more(char_range("0", "9"))


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
    properties: Mapping[str, Any],
    json_schema_refs: MutableMapping[str, Any],
):
    lm += "{"
    properties_added = 0
    for name, property_schema in properties.items():
        lm += '"' + name + '"'

        lm += ":"
        lm += gen_json(
            name=None,
            json_schema=property_schema,
            json_schema_refs=json_schema_refs,
        )
        properties_added += 1
        if properties_added < len(properties):
            lm += ","
    lm += "}"
    return lm


@guidance(stateless=True)
def _gen_json_array(
    lm,
    *,
    item_schema: Mapping[str, Any],
    json_schema_refs: MutableMapping[str, Any],
):
    lm += "["
    lm += optional(
        zero_or_more(
            gen_json(json_schema=item_schema, json_schema_refs=json_schema_refs) + ","
        )
        + gen_json(json_schema=item_schema, json_schema_refs=json_schema_refs)
    )
    lm += "]"
    return lm


@guidance(stateless=True)
def _process_anyOf(
    lm,
    *,
    anyof_list: Sequence[MutableMapping[str, Any]],
    json_schema_refs: MutableMapping[str, Any],
):
    options = [
        gen_json(json_schema=item, json_schema_refs=json_schema_refs)
        for item in anyof_list
    ]
    return lm + select(options)


@guidance(stateless=True)
def gen_json(
    lm,
    name: Union[str, None] = None,
    *,
    json_schema: Mapping[str, Any],
    json_schema_refs: MutableMapping[str, Any] = dict(),
):
    _DEFS_KEY = "$defs"
    if _DEFS_KEY in json_schema:
        json_schema_refs.update(json_schema[_DEFS_KEY])

    ANYOF_STRING = "anyOf"
    if ANYOF_STRING in json_schema:
        return guidance.capture(
            _process_anyOf(
                anyof_list=json_schema[ANYOF_STRING], json_schema_refs=json_schema_refs
            ),
            name=name,
        )

    REF_STRING = "$ref"
    object_schema = None
    if REF_STRING in json_schema:
        target_type = "object"
        object_schema = _get_definition(json_schema[REF_STRING], json_schema_refs)
    else:
        target_type = json_schema["type"]

    result = None
    if target_type == "null":
        result = "null"
    elif target_type == "boolean":
        result = select(["true", "false"])
    elif target_type == "integer":
        result = _gen_json_int()
    elif target_type == "number":
        result = _gen_json_number()
    elif target_type == "string":
        result = _gen_json_string()
    elif target_type == "array":
        result = _gen_json_array(
            item_schema=json_schema["items"], json_schema_refs=json_schema_refs
        )
    elif target_type == "object":
        if object_schema is None:
            object_properties = json_schema["properties"]
        else:
            object_properties = object_schema["properties"]
        result = _gen_json_object(
            properties=object_properties, json_schema_refs=json_schema_refs
        )
    else:
        raise ValueError(f"Unsupported type in schema: {json_schema['type']}")

    return lm + guidance.capture(result, name=name)
