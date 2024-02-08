from guidance._parser import EarleyCommitParser
from guidance._grammar import Byte

from guidance._json_schema_to_grammar import json_schema_to_grammar


def test_string_schema():
    schema = """{ "type": "string" }"""

    grammar = json_schema_to_grammar(schema)
    parser = EarleyCommitParser(grammar)

    simple_json_string = '"my_string"'

    simple_json_string_bytes = bytes(simple_json_string, encoding="utf8")

    for c in simple_json_string:
        print(f"Working on: {c}")
        next_byte = Byte(bytes(c, encoding="utf8"))
        assert parser.valid_next_bytes() == set([next_byte])
        parser.consume_byte(next_byte)
