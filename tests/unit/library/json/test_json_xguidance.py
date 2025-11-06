from json import dumps as json_dumps, loads as json_loads

import pytest
from jsonschema import ValidationError, validate

from guidance import json as gen_json

from .utils import check_match_failure


class TestXGuidance:
    def test_smoke(self):
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
