import json

import pytest

from jsonschema import validate

from guidance._parser import EarleyCommitParser
from guidance._grammar import Byte

from guidance._json_schema_to_grammar import json_schema_to_grammar


@pytest.mark.parametrize(
    "simple_json_string",
    [
        '"with_underscore"',
        '"ALLCAPS"',
        '"with a space"',
        '"MiXeD cAsInG"',
        '"with-hyphen"',
        '"Mix case_underscore-hyphens',
    ],
)
def test_string_schema(simple_json_string):
    schema = """{ "type": "string" }"""

    # First sanity check what we're setting up
    schema_obj = json.loads(schema)
    validate(instance=simple_json_string, schema=schema_obj)

    # Now set up the actual conversion
    grammar = json_schema_to_grammar(schema)
    parser = EarleyCommitParser(grammar)

    for c in simple_json_string:
        print(f"Working on: {c}")
        print(f"Valid next bytes: {parser.valid_next_bytes()}")
        next_byte = bytes(c, encoding="utf8")
        print(f"Consuming: {next_byte}")
        parser.consume_byte(next_byte)
