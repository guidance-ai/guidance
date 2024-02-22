import json

import pytest
from jsonschema import validate

from guidance import models

from guidance.library import gen_json


def to_compact_json(target: any) -> str:
    # See 'Compact Encoding':
    # https://docs.python.org/3/library/json.html
    # Since this is ultimately about the generated
    # output, we don't need to worry about pretty printing
    # and whitespace
    return json.dumps(target, separators=(",", ":"))


def _generate_and_check(target_obj: any, schema_obj):
    prepared_string = f"<s>{to_compact_json(target_obj)}"
    lm = models.Mock(prepared_string.encode())

    # Run with the mock model
    lm += gen_json(json_schema=schema_obj)

    # Make sure the round trip works
    assert json.loads(str(lm)) == target_obj


def test_null():
    schema = """{"type": "null" }"""

    target_obj = None

    # First sanity check what we're setting up
    schema_obj = json.loads(schema)
    validate(instance=target_obj, schema=schema_obj)

    _generate_and_check(target_obj, schema_obj)


@pytest.mark.parametrize(
    "my_int",
    [0, 1, 100, 9876543210, 99, 737, 858, -1, -10, -20],
)
def test_integer_schema(my_int):
    schema = """{ "type": "integer" }"""

    # First sanity check what we're setting up
    schema_obj = json.loads(schema)
    validate(instance=my_int, schema=schema_obj)

    # The actual check
    _generate_and_check(my_int, schema_obj)


@pytest.mark.parametrize(
    "my_string",
    [
        "with_underscore",
        "ALLCAPS",
        "with a space",
        "MiXeD cAsInG",
        "with-hyphen",
        "Mix case_underscore-hyphens",
        "with a comma, in the string",
        "A full stop.",
        """How about
            a
            multiline string?""",
        "A \t tab \t between \t words",
        r"End with backslash \ ",
        "Have a forward / slash",
        "Include [the] odd {brace} and (parentheses)",
        "Some more symbols: ; are useful!",
    ],
)
def test_string_schema(my_string: str):
    schema = """{ "type": "string" }"""

    # First sanity check what we're setting up
    schema_obj = json.loads(schema)
    validate(instance=my_string, schema=schema_obj)

    # The actual check
    _generate_and_check(my_string, schema_obj)


def test_object_with_single_property():
    schema = """{
        "type": "object",
        "properties": {
            "a" : {"type": "integer"}
        }
    }
"""
    target_obj = dict(a=1)

    # First sanity check what we're setting up
    schema_obj = json.loads(schema)
    validate(instance=target_obj, schema=schema_obj)

    # The actual check
    _generate_and_check(target_obj, schema_obj)


def test_object_with_many_properties():
    schema = """{
        "type": "object",
        "properties": {
            "a" : {"type": "integer"},
            "b" : {"type": "integer"},
            "c" : {"type": "integer"},
            "d" : {"type": "integer"},
            "e" : {"type": "integer"},
            "f" : {"type": "integer"},
            "g" : {"type": "integer"},
            "h" : {"type": "integer"}
        }
    }
"""
    target_obj = dict(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8)

    # First sanity check what we're setting up
    schema_obj = json.loads(schema)
    validate(instance=target_obj, schema=schema_obj)

    # The actual check
    _generate_and_check(target_obj, schema_obj)


@pytest.mark.parametrize("target_obj", [[], [0], [34, 56], [1, 2, 3], [9, 8, 7, 6]])
def test_integer_list(target_obj):
    schema = """{
    "type" : "array",
    "items" : {
            "type" : "integer"
        }
    }
"""

    # First sanity check what we're setting up
    schema_obj = json.loads(schema)
    validate(instance=target_obj, schema=schema_obj)

    # The actual check
    _generate_and_check(target_obj, schema_obj)