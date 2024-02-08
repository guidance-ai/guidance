import json

import pytest

from jsonschema import validate

from guidance._parser import EarleyCommitParser
from guidance._grammar import Byte

from guidance._json_schema_to_grammar import json_schema_to_grammar


def to_compact_json(target: any) -> str:
    # See 'Compact Encoding':
    # https://docs.python.org/3/library/json.html
    # Since this is ultimately about the generated
    # output, we don't need to worry about pretty printing
    # and whitespace
    return json.dumps(target, separators=(",", ":"))


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


@pytest.mark.parametrize(
    "json_int",
    [to_compact_json(x) for x in [0, 1, 100, 9876543210, 99, 737, 858, -1, -10, -20]],
)
def test_integer_schema(json_int):
    schema = """{ "type": "integer" }"""

    # First sanity check what we're setting up
    schema_obj = json.loads(schema)
    validate(instance=json.loads(json_int), schema=schema_obj)

    # Now set up the actual conversion
    grammar = json_schema_to_grammar(schema)
    parser = EarleyCommitParser(grammar)
    for c in json_int:
        print(f"Working on: {c}")
        print(f"Valid next bytes: {parser.valid_next_bytes()}")
        next_byte = bytes(c, encoding="utf8")
        print(f"Consuming: {next_byte}")
        parser.consume_byte(next_byte)


def test_simple_object():
    schema = """{
        "type": "object",
        "properties": {
            "name" : {
                "type": "string"
            },
            "productId": {
                "description": "The unique identifier for a product",
                "type": "integer"
            }
        }
    }
"""
    target_obj = dict(name="my product", productId=123)

    # First sanity check what we're setting up
    schema_obj = json.loads(schema)
    validate(instance=target_obj, schema=schema_obj)

    grammar = json_schema_to_grammar(schema)
    parser = EarleyCommitParser(grammar)

    target_string = to_compact_json(target_obj)
    print(f"target_string: {target_string}")

    for c in target_string:
        print(f"Working on: {c}")
        print(f"Valid next bytes: {parser.valid_next_bytes()}")
        next_byte = bytes(c, encoding="utf8")
        print(f"Consuming: {next_byte}")
        parser.consume_byte(next_byte)
