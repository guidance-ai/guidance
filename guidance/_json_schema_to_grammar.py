import json

from ._grammar import Byte, Join, Select, GrammarFunction

from .library._char_range import char_range

_QUOTE = Byte(b'"')
_SAFE_STRING = Select(
    [
        char_range("a", "z"),
        char_range("A", "Z"),
        char_range("0", "9"),
        "_",
        "-",
        "'",
        " ",
    ],
    recursive=True,
)


def _process_node(node: dict[str, any]) -> GrammarFunction:
    if node["type"] == "string":
        return Join([_QUOTE, _SAFE_STRING, _QUOTE])
    else:
        raise ValueError(f"Unsupposed type in schema: {node['type']}")


def json_schema_to_grammar(schema: str) -> GrammarFunction:
    schema_obj = json.loads(schema)

    return _process_node(schema_obj)
