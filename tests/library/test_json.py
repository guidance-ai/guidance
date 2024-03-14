import json
from typing import Any

import pytest
from jsonschema import validate

from guidance import models
from guidance._parser import ParserException
from guidance.library import json as gen_json
from guidance.library._json import _to_compact_json


def _generate_and_check(target_obj: Any, schema_obj):
    # Sanity check what we're being asked
    validate(instance=target_obj, schema=schema_obj)

    prepared_string = f"<s>{_to_compact_json(target_obj)}"
    lm = models.Mock(prepared_string.encode())

    # Run with the mock model
    CAPTURE_KEY = "my_capture"
    lm += gen_json(name=CAPTURE_KEY, json_schema=schema_obj)

    # Make sure the round trip works
    assert json.loads(lm[CAPTURE_KEY]) == target_obj


def _check_match_failure(bad_string, failure_byte, schema_obj):
    grammar = gen_json(schema_obj)
    with pytest.raises(ParserException) as pe:
        grammar.match(bad_string, raise_exceptions=True)
    assert pe.value.current_byte == failure_byte


def test_null():
    schema = """{"type": "null" }"""

    target_obj = None

    # First sanity check what we're setting up
    schema_obj = json.loads(schema)
    validate(instance=target_obj, schema=schema_obj)

    _generate_and_check(target_obj, schema_obj)


@pytest.mark.parametrize("target_obj", [True, False])
def test_boolean(target_obj):
    schema = """{"type": "boolean" }"""

    # First sanity check what we're setting up
    schema_obj = json.loads(schema)
    validate(instance=target_obj, schema=schema_obj)

    _generate_and_check(target_obj, schema_obj)


class TestInteger:
    schema = """{ "type": "integer" }"""

    @pytest.mark.parametrize(
        "my_int",
        [0, 1, 100, 9876543210, 99, 737, 858, -1, -10, -20],
    )
    def test_integer_schema(self, my_int):
        # First sanity check what we're setting up
        schema_obj = json.loads(TestInteger.schema)
        validate(instance=my_int, schema=schema_obj)

        # The actual check
        _generate_and_check(my_int, schema_obj)

    @pytest.mark.parametrize(
        ["bad_string", "failure_byte"],
        [
            ("9999a7777", b"a"),
            ("123, []", b","),
            ("a321", b"a"),
            ("123789.456", b"."),
            ("[]", b"["),
            ('{"a":4}', b"{"),
        ],
    )
    def test_bad_integer(self, bad_string, failure_byte):
        schema_obj = json.loads(TestInteger.schema)
        _check_match_failure(bad_string, failure_byte, schema_obj)


class TestNumber:
    schema = """{"type": "number" }"""

    @pytest.mark.parametrize(
        "target_obj",
        # It appears that Inf and NaN are not actually part of the JSON spec
        [
            0,
            0.0,
            1.0,
            -1.0,
            1,
            -1,
            134,
            -234762,
            0.1,
            -0.1,
            -10.33,
            452.342,
            1.23e23,
            -1.2e-22,
            123.6,
        ],
    )
    def test_number(self, target_obj):
        # First sanity check what we're setting up
        schema_obj = json.loads(TestNumber.schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        _generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        ["bad_string", "failure_byte"],
        [
            ("9999a7777", b"a"),
            ("123.6, []", b","),
            ("a321", b"a"),
            ("[]", b"["),
            ('{"a":4}', b"{"),
        ],
    )
    def test_bad_number(self, bad_string, failure_byte):
        schema_obj = json.loads(TestNumber.schema)
        _check_match_failure(bad_string, failure_byte, schema_obj)


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


class TestSimpleObject:
    # These are objects without cross references
    def test_object_with_single_property(self):
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

    def test_object_with_many_properties(self):
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

    def test_directly_nested_object(self):
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

        # The actual check
        _generate_and_check(target_obj, schema_obj)

    def test_object_containing_list(self):
        schema = """{
        "type": "object",
        "properties" : {
                "a" : { "type" : "string" },
                "b list" : {
                    "type": "array",
                    "items" : {"type": "integer" }
                }
            }
        }
    """

        target_obj = {
            "a": "some lengthy string of characters",
            "b list": [1, 2, 3, 2312, 123],
        }

        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        _generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        ["bad_string", "failure_byte"],
        [
            ("9999a7777", b"9"),
            ('{"a":1255.4567}', b"."),
            ('{"a":"123"}', b'"'),
        ],
    )
    def test_bad_object(self, bad_string, failure_byte):
        schema = """{
            "type": "object",
            "properties": {
                "a" : {"type": "integer"}
            }
        }
    """
        schema_obj = json.loads(schema)
        _check_match_failure(bad_string, failure_byte, schema_obj)


class TestSimpleArray:
    # These are array without references
    @pytest.mark.parametrize("target_obj", [[], [0], [34, 56], [1, 2, 3], [9, 8, 7, 6]])
    def test_integer_list(self, target_obj):
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

    @pytest.mark.parametrize("target_obj", [[], ["a"], ["b c", "d, e"]])
    def test_string_list(self, target_obj):
        schema = """{
        "type" : "array",
        "items" : {
                "type" : "string"
            }
        }
    """

        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        _generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "target_obj",
        [[], [dict(a=1)], [dict(a=2), dict(a=3)], [dict(a=4), dict(a=5), dict(a=6)]],
    )
    def test_object_list(self, target_obj):
        schema = """{
        "type" : "array",
        "items" : {
                "type" : "object",
                "properties" : {
                    "a" : {
                        "type": "integer"
                    }
                }
            }
        }
    """

        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        _generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        ["bad_string", "failure_byte"],
        [
            ("9999a7777", b"9"),
            ("[321.654]", b"."),
            ('["123"]', b'"'),
        ],
    )
    def test_bad_object(self, bad_string, failure_byte):
        schema = """{
        "type" : "array",
        "items" : {
                "type" : "integer"
            }
        }"""
        schema_obj = json.loads(schema)
        _check_match_failure(bad_string, failure_byte, schema_obj)


class TestWithReferences:
    @pytest.mark.parametrize(
        "target_obj",
        [
            dict(all_cats=[]),
            dict(all_cats=[dict(name="Kasha")]),
            dict(all_cats=[dict(name="Dawon"), dict(name="Barong")]),
        ],
    )
    def test_simple_ref(self, target_obj):
        schema = """{
        "$defs": {
            "Cat": {
            "properties": {
                "name": {
                "title": "Name",
                "type": "string"
                }
            },
            "required": [
                "name"
            ],
            "title": "Cat",
            "type": "object"
            }
        },
        "properties": {
            "all_cats": {
            "items": {
                "$ref": "#/$defs/Cat"
            },
            "title": "All Cats",
            "type": "array"
            }
        },
        "required": [
            "all_cats"
        ],
        "title": "CatList",
        "type": "object"
        }"""

        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        _generate_and_check(target_obj, schema_obj)

    def test_nested_ref(self):
        schema = """{
        "$defs": {
            "A": {
            "properties": {
                "name": {
                "title": "Name",
                "type": "string"
                }
            },
            "required": [
                "name"
            ],
            "title": "A",
            "type": "object"
            },
            "B": {
            "properties": {
                "other_str": {
                "title": "Other Str",
                "type": "string"
                },
                "my_A": {
                "$ref": "#/$defs/A"
                }
            },
            "required": [
                "other_str",
                "my_A"
            ],
            "title": "B",
            "type": "object"
            }
        },
        "properties": {
            "my_B": {
            "$ref": "#/$defs/B"
            }
        },
        "required": [
            "my_B"
        ],
        "title": "C",
        "type": "object"
        }"""

        target_obj = dict(my_B=dict(other_str="some string", my_A=dict(name="my name")))

        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        _generate_and_check(target_obj, schema_obj)


class TestAnyOf:
    @pytest.mark.parametrize("target_obj", [123, True])
    def test_anyOf_simple(self, target_obj):
        schema = """{
        "anyOf": [
                    {
                        "type": "integer"
                    },
                    {
                        "type": "boolean"
                    }
                ]
    }
    """
        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        _generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "target_obj",
        [
            dict(my_val=dict(my_int=1)),
            dict(my_val=dict(my_str="Some long string or other")),
        ],
    )
    def test_anyOf_objects(self, target_obj):
        schema = """{
    "$defs": {
        "A": {
        "properties": {
            "my_str": {
            "default": "me",
            "title": "My Str",
            "type": "string"
            }
        },
        "title": "A",
        "type": "object"
        },
        "B": {
        "properties": {
            "my_int": {
            "default": 1,
            "title": "My Int",
            "type": "integer"
            }
        },
        "title": "B",
        "type": "object"
        }
    },
    "properties": {
        "my_val": {
        "anyOf": [
            {
            "$ref": "#/$defs/A"
            },
            {
            "$ref": "#/$defs/B"
            }
        ],
        "title": "My Val"
        }
    },
    "title": "C",
    "type": "object"
    }
    """
        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        _generate_and_check(target_obj, schema_obj)


class TestEnum:
    simple_schema = """{
        "enum": [1,"2",false]
    }
    """

    prefix_schema = """{
        "enum": ["aa", "bb", "cc"]
    }"""

    @pytest.mark.parametrize("target_obj", [1, "2", False])
    def test_enum(self, target_obj):
        # First sanity check what we're setting up
        schema_obj = json.loads(self.simple_schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        _generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_obj, failure_byte",
        [
            ("1", b"1"),
            (2, b"2"),
            (True, b"t"),
        ],
    )
    def test_bad_enum(self, bad_obj, failure_byte):
        schema_obj = json.loads(self.simple_schema)
        bad_str = _to_compact_json(bad_obj)
        _check_match_failure(bad_str, failure_byte, schema_obj)

    @pytest.mark.parametrize(
        "bad_obj, failure_byte",
        [
            ("ab", b"b"),
            ("bc", b"c"),
            ("ca", b"a"),
        ],
    )
    def test_bad_prefix_enum(self, bad_obj, failure_byte):
        schema_obj = json.loads(self.prefix_schema)
        bad_str = _to_compact_json(bad_obj)
        _check_match_failure(bad_str, failure_byte, schema_obj)


class TestAdditionalProperties:

    simple_schema = """{
    "type": "object",
    "additionalProperties": {
            "type" : "integer"
        }
    }
    """

    anyOf_schema = """{
    "type": "object",
    "additionalProperties": {
            "anyOf": [
                {"type" : "string"},
                {"type": "integer"}
            ]
        }
    }
    """

    combined_schema = """{
    "type": "object",
    "properties": {
            "mystr": {"type": "string"}
        },
    "additionalProperties": {
            "type": "integer"
        }
    }
    """

    @pytest.mark.parametrize("target_obj", [{}, {"a": 1}, {"a": 1, "b": 2}])
    def test_simple_additional_properties(self, target_obj):
        # First sanity check what we're setting up
        schema_obj = json.loads(self.simple_schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        _generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_obj, failure_byte",
        [
            ({"a": "1"}, b'"'),
            ({"a": 1, "b": 1.5}, b"."),
        ],
    )
    def test_simple_bad_type(self, bad_obj, failure_byte):
        schema_obj = json.loads(self.simple_schema)
        bad_string = _to_compact_json(bad_obj)
        _check_match_failure(bad_string, failure_byte, schema_obj)

    @pytest.mark.parametrize(
        "target_obj", [{}, {"a": 1}, {"a": "2"}, {"a": 1, "b": "2"}]
    )
    def test_anyOf_additional_properties(self, target_obj):
        # First sanity check what we're setting up
        schema_obj = json.loads(self.anyOf_schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        _generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_obj, failure_byte",
        [({"a": 1.5}, b"."), ({"a": True}, b"t"), ({"a": 1, "b": False}, b"f")],
    )
    def test_anyOf_bad_type(self, bad_obj, failure_byte):
        schema_obj = json.loads(self.anyOf_schema)
        bad_string = _to_compact_json(bad_obj)
        _check_match_failure(bad_string, failure_byte, schema_obj)

    @pytest.mark.parametrize(
        "target_obj",
        [
            {"mystr": "hello"},
            {"mystr": "hello", "a": 1},
            {"mystr": "hello", "a": 1, "b": 2},
        ],
    )
    def test_properties_and_additional_properties(self, target_obj):
        # First sanity check what we're setting up
        schema_obj = json.loads(self.combined_schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        _generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_obj, failure_byte",
        [
            ({}, b"}"),
            ({"a": 1}, b"a"),
            ({"a": 1, "b": 2}, b"a"),
        ],
    )
    def test_combined_missing_properties(self, bad_obj, failure_byte):
        schema_obj = json.loads(self.combined_schema)
        bad_string = _to_compact_json(bad_obj)
        _check_match_failure(bad_string, failure_byte, schema_obj)

    @pytest.mark.parametrize(
        "bad_obj, failure_byte",
        [
            ({"mystr": 1}, b"1"),
            ({"mystr": 1, "a": 2}, b"1"),
            ({"mystr": "hello", "a": False}, b"f"),
        ],
    )
    def test_combined_bad_type(self, bad_obj, failure_byte):
        schema_obj = json.loads(self.combined_schema)
        bad_string = _to_compact_json(bad_obj)
        _check_match_failure(bad_string, failure_byte, schema_obj)


class TestRecursiveStructures:
    @pytest.mark.parametrize(
        "target_obj",
        [
            dict(my_list=None),
            dict(my_list=dict(my_str="a", next=None)),
            dict(my_list=dict(my_str="a", next=dict(my_str="b", next=None))),
        ],
    )
    def test_linked_list(self, target_obj):
        schema = """
{
    "$defs": {
        "A": {
            "properties": {
                "my_str": {
                    "default": "me",
                    "title": "My Str",
                    "type": "string"
                },
                "next": {
                    "anyOf": [
                        {
                            "$ref": "#/$defs/A"
                        },
                        {
                            "type": "null"
                        }
                    ]
                }
            },
            "type": "object"
        }
    },
    "type": "object",
    "properties": {
        "my_list": {
            "anyOf": [
                {
                    "$ref": "#/$defs/A"
                },
                {
                    "type": "null"
                }
            ]
        }
    }
}
        """
        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        _generate_and_check(target_obj, schema_obj)
