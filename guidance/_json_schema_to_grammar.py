import json
from typing import Dict, Union

from ._grammar import Byte, GrammarFunction, Join, Select, select
from .library._char_range import char_range

_QUOTE = Byte(b'"')
_SAFE_STRING = select(
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
_OPEN_BRACE = Byte(b"{")
_CLOSE_BRACE = Byte(b"}")
_OPEN_BRACKET = Byte(b"[")
_CLOSE_BRACKET = Byte(b"]")
_COMMA = Byte(b",")
_COLON = Byte(b":")


def _make_optional(f: GrammarFunction) -> GrammarFunction:
    return select(["", f])


def _process_int() -> GrammarFunction:
    return Join([select(["-", ""]), select([char_range("0", "9")], recurse=True)])


def _process_number() -> GrammarFunction:
    mantissa_int = _process_int()
    mantissa_frac = _make_optional(
        Join([Byte(b"."), select([char_range("0", "9")], recurse=True)])
    )
    exponent = _make_optional(
        Join(
            [
                "e",
                # Since the exponent can contain a '+', can't just reuse
                # _process_int() here
                select(["", "-", "+"]),
                select([char_range("0", "9")], recurse=True),
            ]
        )
    )
    return Join(
        [
            mantissa_int,
            mantissa_frac,
            exponent,
        ],
    )


def _process_object(
    schema_properties: Dict[str, any], definitions: Union[Dict[str, any], None]
) -> GrammarFunction:
    properties = []
    for name, nxt_node in schema_properties.items():
        nxt = Join(
            [
                Join([_QUOTE, name, _QUOTE]),
                _COLON,
                _process_node(nxt_node, definitions),
            ]
        )
        properties.append(nxt)
        if len(properties) < len(schema_properties):
            properties.append(_COMMA)
    return Join([_OPEN_BRACE, *properties, _CLOSE_BRACE])


def _process_array(
    item_node: Dict[str, any], definitions: Union[Dict[str, any], None]
) -> GrammarFunction:
    return Join(
        [
            _OPEN_BRACKET,
            _make_optional(
                # One or more items
                Join(
                    [
                        select(
                            ["", Join([_process_node(item_node, definitions), _COMMA])],
                            recurse=True,
                        ),
                        _process_node(item_node, definitions),
                    ]
                )
            ),
            _CLOSE_BRACKET,
        ]
    )


def _get_definition(reference: str, definitions: Dict[str, any]) -> Dict[str, any]:
    assert definitions is not None
    REF_START = "#/$defs/"
    assert reference.startswith(
        REF_START
    ), f"Reference {reference} must start with {REF_START}"

    target_name = reference[len(REF_START) :]
    return definitions[target_name]


def _process_node(
    node: Dict[str, any], definitions: Union[Dict[str, any], None]
) -> GrammarFunction:
    if node["type"] == "null":
        # Not completely sure about this
        return Select(["null"])
    elif node["type"] == "string":
        return Join([_QUOTE, _SAFE_STRING, _QUOTE])
    elif node["type"] == "boolean":
        return select(["true", "false"])
    elif node["type"] == "integer":
        return _process_int()
    elif node["type"] == "number":
        return _process_number()
    elif node["type"] == "object":
        return _process_object(node["properties"], definitions)
    elif node["type"] == "array":
        if "type" in node["items"]:
            item_node = dict(type=node["items"]["type"])
            if item_node["type"] == "object":
                item_node["properties"] = node["items"]["properties"]
        else:
            item_node = _get_definition(node["items"]["$ref"], definitions)
        return _process_array(item_node, definitions)
    else:
        raise ValueError(f"Unsupported type in schema: {node['type']}")


def json_schema_to_grammar(schema: str) -> GrammarFunction:
    schema_obj = json.loads(schema)

    _DEFS_KEY = "$defs"

    definitions = None
    if _DEFS_KEY in schema_obj:
        definitions = schema_obj[_DEFS_KEY]
        del schema_obj[_DEFS_KEY]

    return _process_node(schema_obj, definitions)
