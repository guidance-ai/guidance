import collections.abc

from typing import Union

import guidance
from .._grammar import select

from ._char_range import char_range
from ._one_or_more import one_or_more
from ._optional import optional
from ._zero_or_more import zero_or_more


@guidance(stateless=True)
def _gen_json_int(lm):
    return lm + optional("-") + one_or_more(char_range("0", "9"))


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
    properties: collections.abc.Mapping[str, any],
    json_schema_refs: collections.abc.MutableMapping[str, any],
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
    item_schema: collections.abc.Mapping[str, any],
    json_schema_refs: collections.abc.MutableMapping[str, any],
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
def gen_json(
    lm,
    name: Union[str, None] = None,
    *,
    json_schema: collections.abc.Mapping[str, any],
    json_schema_refs: collections.abc.MutableMapping[str, any] = dict(),
):
    _DEFS_KEY = "$defs"
    if _DEFS_KEY in json_schema:
        json_schema_refs.update(json_schema[_DEFS_KEY])

    if json_schema["type"] == "null":
        return lm + "null"
    elif json_schema["type"] == "boolean":
        return lm + select(["true", "false"])
    elif json_schema["type"] == "integer":
        return lm + _gen_json_int()
    elif json_schema["type"] == "string":
        return lm + _gen_json_string()
    elif json_schema["type"] == "array":
        return lm + _gen_json_array(
            item_schema=json_schema["items"], json_schema_refs=json_schema_refs
        )
    elif json_schema["type"] == "object":
        return lm + _gen_json_object(
            properties=json_schema["properties"], json_schema_refs=json_schema_refs
        )
    else:
        raise ValueError(f"Unsupported type in schema: {json_schema['type']}")
