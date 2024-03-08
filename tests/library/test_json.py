import json

from typing import Any

import pytest
from jsonschema import validate

from guidance import models
from guidance.library import json as gen_json

from ..utils import to_compact_json


def _generate_and_check(target_obj: Any, schema_obj):
    # Sanity check what we're being asked
    validate(instance=target_obj, schema=schema_obj)

    prepared_string = f"<s>{to_compact_json(target_obj)}"
    lm = models.Mock(prepared_string.encode())

    # Run with the mock model
    CAPTURE_KEY = "my_capture"
    lm += gen_json(name=CAPTURE_KEY, json_schema=schema_obj)

    # Make sure the round trip works
    assert json.loads(lm[CAPTURE_KEY]) == target_obj


def _check_failed_generation(bad_string: str, expected_output: Any, schema_obj):
    """
    One can argue that this is slightly misnamed. The generation should never
    fail so long as the Mock model keeps producing output, and the output itself
    should always conform to the specified schema. However, the final output
    won't just reproduce the input because the input doesn't match the schema
    in the negative test cases
    """
    prepared_string = "<s>" + bad_string
    lm = models.Mock(prepared_string.encode())

    # Run with the mock model
    CAPTURE_KEY = "my_capture"
    lm += gen_json(name=CAPTURE_KEY, json_schema=schema_obj)
    print(f"{lm[CAPTURE_KEY]=}")

    generated_obj = json.loads(lm[CAPTURE_KEY])
    # Following is going to depend on implementation details
    # of the mock
    assert generated_obj == expected_output

    # Ensure what what was output does still match
    # the schema
    validate(instance=generated_obj, schema=schema_obj)


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

    """
    Note that '2' is the failure case in the following.

    I have not fully decoded why, but ultimately, the
    Mock model depends on MockEngine, and that generates
    randomised logits for the next token (the PRNG is
    seeded in the MockEngine constructor). Between the
    (pseudo-)randomised logits and the grammar, it
    appears that we manage to generate a '2' before
    hitting a character which won't work with the grammar,
    thereby terminating the generation
    """

    @pytest.mark.parametrize(
        ["bad_string", "expected_capture"],
        [
            ("9999a7777", 9999),  # 'a' is first failure
            ("123, []", 123),
            ("a321", 2),  # Failure case
            ("123789.456", 123789),  # '.' is first failure
            ("[]", 2),  # Failure case
            ('{"a":4}', 2),  # Failure case
        ],
    )
    def test_bad_integer(self, bad_string, expected_capture: int):
        schema_obj = json.loads(TestInteger.schema)
        _check_failed_generation(bad_string, expected_capture, schema_obj)


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

    # See above for explanation of the failure cases
    @pytest.mark.parametrize(
        ["bad_string", "expected_capture"],
        [
            ("9999a7777", 9999),  # 'a' is the first failure
            ("123.6, []", 123.6),  # ',' is the first failure
            ("a321", 2),  # Failure case
            ("[]", 2),  # Failure case
            ('{"a":4}', 2),  # Failure case
        ],
    )
    def test_bad_number(self, bad_string, expected_capture: int):
        schema_obj = json.loads(TestNumber.schema)
        _check_failed_generation(bad_string, expected_capture, schema_obj)


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

    """
    The following is even more complicated than the failure cases
    discussed above.

    The key point is that the schema forces the generation of
    '{"a":' and '}'. The only part where the mocked model will
    actually be called is to generate the integer value.
    And then values will be pulled from the string supplied to the
    mocked model until an invalid character is encountered
    """

    @pytest.mark.parametrize(
        ["bad_string", "expected_capture"],
        [
            ("9999a7777", {"a": 274153349361519053618738762}),
            ('{"a":1255.4567}', {"a": 125553349361519053618738762}),
            ('{"a":"123"}', {"a": 274153349361519053618738762}),
        ],
    )
    def test_bad_object(self, bad_string, expected_capture: int):
        schema = """{
            "type": "object",
            "properties": {
                "a" : {"type": "integer"}
            }
        }
    """
        schema_obj = json.loads(schema)
        _check_failed_generation(bad_string, expected_capture, schema_obj)


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

    """
    Again, in the following, the exact output is dependent
    on how many valid input characters the mock supplies vs
    random ones, coupld with the brackets being forced.
    """

    @pytest.mark.parametrize(
        ["bad_string", "expected_capture"],
        [
            ("9999a7777", []),
            ("[321.654]", [32115, 3493615190]),
            ('["123"]', []),
        ],
    )
    def test_bad_object(self, bad_string, expected_capture: int):
        schema = """{
        "type" : "array",
        "items" : {
                "type" : "integer"
            }
        }"""
        schema_obj = json.loads(schema)
        _check_failed_generation(bad_string, expected_capture, schema_obj)


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
