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
