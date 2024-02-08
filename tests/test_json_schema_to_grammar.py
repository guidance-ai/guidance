import json

from jsonschema import validate

from guidance._parser import EarleyCommitParser
from guidance._grammar import Byte

from guidance._json_schema_to_grammar import json_schema_to_grammar


def test_string_schema():
    schema = """{ "type": "string" }"""

    simple_json_string = '"my_string"'

    # First sanity check what we're setting up
    schema_obj = json.loads(schema)
    validate(instance=simple_json_string, schema=schema_obj)

    # Now set up the actual conversion
    grammar = json_schema_to_grammar(schema)
    parser = EarleyCommitParser(grammar)

    for c in simple_json_string:
        print(f"Working on: {c}")
        next_byte = Byte(bytes(c, encoding="utf8"))
        parser.consume_byte(next_byte)
