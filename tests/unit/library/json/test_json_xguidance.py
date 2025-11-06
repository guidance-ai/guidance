from json import dumps as json_dumps, loads as json_loads

import pytest
from jsonschema import ValidationError, validate

from guidance import json as gen_json

from .utils import check_match_failure


class TestXGuidance:
    @pytest.mark.parametrize("desired_indent", [None, 2, 4])
    @pytest.mark.parametrize("separators", [None, (",", ":"), (", ", ": "), (",     ", ":      ")])
    def test_whitespace_flexible(self, desired_indent, separators):
        schema = """{
            "type": "object",
            "properties": {
                "a" : {"type": "integer"},
                "b" : {"type": "integer"}
            },
            "required": ["a"],
            "additionalProperties": false,
            "x-guidance" : {
              "whitespace_flexible": true
           }
        }
        """
        target_obj = {"a": 1, "b": 2}

        # First sanity check what we're setting up
        schema_obj = json_loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # Now, check an indented string
        indented_string = json_dumps(target_obj, indent=desired_indent, separators=separators)
        gen_json_grammar = gen_json(schema=schema)

        match = gen_json_grammar.match(indented_string, raise_exceptions=True)
        assert match is not None, "Grammar failed to match indented JSON string"

    def test_whitespace_inflexible_indented(self):
        schema = """{
            "type": "object",
            "properties": {
                "a" : {"type": "integer"},
                "b" : {"type": "integer"}
            },
            "required": ["a"],
            "additionalProperties": false,
            "x-guidance" : {
              "whitespace_flexible": false
           }
        }
        """
        target_obj = {"a": 1, "b": 2}

        # First sanity check what we're setting up
        schema_obj = json_loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # Now, check an indented string, which should fail
        # due to the whitespace_flexible: false setting
        check_match_failure(
            bad_string=json_dumps(target_obj, indent=4),
            good_bytes=b"{",
            failure_byte=b"\n",
            schema_obj=schema,
        )

    def test_item_separator(self):
        schema = """{
            "type": "object",
            "properties": {
                "a" : {"type": "integer"},
                "b" : {"type": "integer"}
            },
            "required": ["a"],
            "additionalProperties": false,
            "x-guidance": {
              "whitespace_flexible": false,
              "item_separator": ","
           }
        }
        """
        target_obj = {"a": 1, "b": 2}

        # First sanity check what we're setting up
        schema_obj = json_loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # Now, check an indented string, which should fail
        # due to the whitespace_flexible: false setting
        check_match_failure(
            bad_string=json_dumps(target_obj, separators=(",", ":")),
            good_bytes=b'{"a":1,',
            failure_byte=b"b",
            schema_obj=schema,
        )
