import json
from typing import Dict

from ._grammar import Byte, GrammarFunction, Join, select
from .library._char_range import char_range

_QUOTE = Byte(b'"')
_SAFE_STRING = select(
    [
        char_range("a", "z"),
        char_range("A", "Z"),
        char_range("0", "9"),
        "_",
        "-",
        "'",
        " ",
        ",",
    ],
    recurse=True,
)
_OPEN_BRACE = Byte(b"{")
_CLOSE_BRACE = Byte(b"}")
_OPEN_BRACKET = Byte(b"[")
_CLOSE_BRACKET = Byte(b"]")
_COMMA = Byte(b",")
_COLON = Byte(b":")


def _process_node(node: Dict[str, any]) -> GrammarFunction:
    if node["type"] == "string":
        return Join([_QUOTE, _SAFE_STRING, _QUOTE])
    elif node["type"] == "boolean":
        return select(["true", "false"])
    elif node["type"] == "integer":
        return Join([select(["-", ""]), select([char_range("0", "9")], recurse=True)])
    elif node["type"] == "number":
        return Join(
            [
                # Leading sign
                select(["", "-"]),
                # Integer part of mantissa
                select([char_range("0", "9")], recurse=True),
                # Optional fractional part of mantissa
                select(
                    [
                        "",
                        Join(
                            [Byte(b"."), select([char_range("0", "9")], recurse=True)]
                        ),
                    ]
                ),
                # Optional exponent
                select(
                    [
                        "",
                        Join(
                            [
                                "e",
                                select(["", "-", "+"]),
                                select([char_range("0", "9")], recurse=True),
                            ]
                        ),
                    ]
                ),
            ],
        )
    elif node["type"] == "object":
        properties = []
        for name, nxt_node in node["properties"].items():
            nxt = Join(
                [
                    Join([_QUOTE, name, _QUOTE]),
                    _COLON,
                    _process_node(nxt_node),
                ]
            )
            properties.append(nxt)
            if len(properties) < len(node["properties"]):
                properties.append(_COMMA)
        return Join([_OPEN_BRACE, *properties, _CLOSE_BRACE])
    elif node["type"] == "array":
        item_node = dict(type=node["items"]["type"])
        if item_node["type"] == "object":
            item_node["properties"] = node["items"]["properties"]
        return Join(
            [
                _OPEN_BRACKET,
                # This may not be quite correct. The last item in a list must _not_
                # be followed by a comma
                select(["", Join([_process_node(item_node), _COMMA])], recurse=True),
                select(["", _process_node(item_node)]),
                _CLOSE_BRACKET,
            ]
        )
    else:
        raise ValueError(f"Unsupported type in schema: {node['type']}")


def json_schema_to_grammar(schema: str) -> GrammarFunction:
    schema_obj = json.loads(schema)

    return _process_node(schema_obj)
