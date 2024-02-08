import json

import pytest
from jsonschema import validate

from guidance._grammar import GrammarFunction
from guidance._json_schema_to_grammar import json_schema_to_grammar
from guidance._parser import EarleyCommitParser


def to_compact_json(target: any) -> str:
    # See 'Compact Encoding':
    # https://docs.python.org/3/library/json.html
    # Since this is ultimately about the generated
    # output, we don't need to worry about pretty printing
    # and whitespace
    return json.dumps(target, separators=(",", ":"))


def check_string_with_grammar(input_string: str, grammar: GrammarFunction):
    parser = EarleyCommitParser(grammar)

    print(f"Checking {input_string}")
    for c in input_string:
        print(f"Working on: {c}")
        print(f"Valid next bytes: {parser.valid_next_bytes()}")
        next_byte = bytes(c, encoding="utf8")
        print(f"Consuming: {next_byte}")
        parser.consume_byte(next_byte)


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
    check_string_with_grammar(json_int, grammar)


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

    target_string = to_compact_json(target_obj)
    check_string_with_grammar(target_string, grammar)


def test_nested_object():
    schema = """{
        "type": "object",
        "properties": {
            "name" : {
                "type": "string"
            },
            "info": {
                "type": "object",
                "properties" : {
                    "a" : {
                        "type" : "integer"
                    },
                    "b" : {
                        "type" : "integer"
                    }
                }
            }
        }
    }
"""
    target_obj = dict(name="my product", info=dict(a=1, b=2))

    # First sanity check what we're setting up
    schema_obj = json.loads(schema)
    validate(instance=target_obj, schema=schema_obj)
