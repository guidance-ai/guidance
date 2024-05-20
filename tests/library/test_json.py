import json
from typing import Any, Union

import pytest
from jsonschema import validate

from guidance import json as gen_json
from guidance import models
from guidance._parser import ParserException
from guidance.library._json import _to_compact_json


def _generate_and_check(
    target_obj: Any, schema_obj, desired_temperature: Union[float, None] = None
):
    # Sanity check what we're being asked
    validate(instance=target_obj, schema=schema_obj)

    # The next part is to prevent intermittent test failures
    # when the temperature is non-zero
    # The Mock model generates random characters once the
    # supplied string has been exhausted. Sometimes
    # these can also be valid according to the grammar
    # (especially when generating numbers) which interferes
    # with our round trip check.
    # So append a 'stop' character which we don't
    # use in any of our tests

    STOP_CHAR = "\g"
    prepared_json = _to_compact_json(target_obj)
    assert STOP_CHAR not in prepared_json, "STOP_CHAR in string"

    prepared_string = f"<s>{prepared_json}{STOP_CHAR}"
    lm = models.Mock(prepared_string.encode())

    # Run with the mock model
    CAPTURE_KEY = "my_capture"
    if desired_temperature is not None:
        lm += gen_json(
            name=CAPTURE_KEY, schema=schema_obj, temperature=desired_temperature
        )
    else:
        lm += gen_json(name=CAPTURE_KEY, schema=schema_obj)

    # Make sure the round trip works
    assert json.loads(lm[CAPTURE_KEY]) == target_obj

    # Check on some temperatures
    if desired_temperature is not None:
        assert len(lm.engine.called_temperatures) > 0
        # Make sure that at least one temperature matches exactly
        temperature_matches = [
            x == desired_temperature for x in lm.engine.called_temperatures
        ]
        assert any(temperature_matches)
        # Check that all temperatures were 0 or the desired temperature
        # If there has been a forced byte, then get_logits() is
        # called with a temperature of zero
        assert all(
            [
                (x == desired_temperature or x == 0)
                for x in lm.engine.called_temperatures
            ]
        )


def _check_match_failure(bad_string, failure_byte, schema_obj):
    grammar = gen_json(schema=schema_obj)
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
@pytest.mark.parametrize("temperature", [None, 0.1, 1])
def test_boolean(target_obj, temperature):
    schema = """{"type": "boolean" }"""

    # First sanity check what we're setting up
    schema_obj = json.loads(schema)
    validate(instance=target_obj, schema=schema_obj)

    _generate_and_check(target_obj, schema_obj, desired_temperature=temperature)


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
    @pytest.mark.parametrize("temperature", [None, 0.1, 1])
    def test_number(self, target_obj, temperature):
        # First sanity check what we're setting up
        schema_obj = json.loads(TestNumber.schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        _generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

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
@pytest.mark.parametrize("temperature", [None, 0.1, 1])
def test_string_schema(my_string: str, temperature):
    schema = """{ "type": "string" }"""

    # First sanity check what we're setting up
    schema_obj = json.loads(schema)
    validate(instance=my_string, schema=schema_obj)

    # The actual check
    _generate_and_check(my_string, schema_obj, desired_temperature=temperature)


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

    @pytest.mark.parametrize("temperature", [None, 0.1, 1])
    def test_directly_nested_object(self, temperature):
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
        _generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

    @pytest.mark.parametrize("temperature", [None, 0.1, 1])
    def test_object_containing_list(self, temperature):
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
        _generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

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
    @pytest.mark.parametrize("temperature", [None, 0.1, 1])
    def test_integer_list(self, target_obj, temperature):
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
        _generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

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
    @pytest.mark.parametrize("temperature", [None, 0.1, 1])
    def test_object_list(self, target_obj, temperature):
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
        _generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

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


class TestArrayWithLengthConstraints:
    prefix_schema_obj = [{"type": "integer"}, {"type": "boolean"}]
    items_schema_obj = {"type": "string"}

    @pytest.mark.parametrize(
        "min_items, max_items, target_obj",
        [
            (0, 0, []),  # None allowed, none provided
            (0, 1, []),  # Some prefixItems allowed, none provided
            (0, 1, [42]),  # Some prefixItems allowed, one provided
            (1, 2, [42, True]),  # All prefix items, no extra items allowed or provided.
            (
                1,
                3,
                [42, True],
            ),  # All prefix items, some extra items allowed but not provided.
            (0, 3, [42, True, "hello"]),  # All prefix items and one extra item
            (
                3,
                4,
                [42, True, "hello"],
            ),  # All prefix items and one extra item but more allowed
            (
                5,
                5,
                [42, True, "hello", "world", "test"],
            ),  # Exactly meets minItems and maxItems.
            (
                0,
                10,
                [42, True] + ["extra"] * 8,
            ),  # Exactly meet large number of extra items
        ],
    )
    def test_good_with_prefix_and_items(self, min_items, max_items, target_obj):
        schema_obj = {
            "prefixItems": self.prefix_schema_obj,
            "items": self.items_schema_obj,
            "minItems": min_items,
            "maxItems": max_items,
            "type": "array",
        }
        _generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "min_items, max_items, target_obj",
        [
            (0, 0, []),  # None allowed, none provided
            (0, 2, []),  # Some allowed, none provided
            (1, 2, [42, True]),  # All prefix items, no extra allowed
            (2, 2, [42, True]),  # Exactly match min, max
            (1, 3, [42]),  # Single prefix item, extra allowed
            (1, 3, [42, True]),  # All prefix items, extra allowed
        ],
    )
    def test_good_with_prefix(self, min_items, max_items, target_obj):
        schema_obj = {
            "prefixItems": self.prefix_schema_obj,
            "minItems": min_items,
            "maxItems": max_items,
            "type": "array",
        }
        _generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "min_items, max_items, target_obj",
        [
            (0, 0, []),  # None allowed, none provided
            (0, 2, []),  # Some allowed, none provided
            (1, 2, ["hello"]),  # Single item, more allowed
            (1, 2, ["hello", "world"]),  # Meet max
            (3, 3, ["hello", "world", "extra"]),  # Exactly match min, max
            (0, 8, ["extra"] * 8),  # Large number of items
        ],
    )
    def test_good_with_items(self, min_items, max_items, target_obj):
        schema_obj = {
            "items": self.items_schema_obj,
            "minItems": min_items,
            "maxItems": max_items,
            "type": "array",
        }
        _generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "min_items, max_items, bad_obj, failure_byte",
        [
            (
                1,
                4,
                [42, "string_not_bool", "hello", "extra"],
                b'"',
            ),  # Second item does not match prefix schema
            (
                0,
                3,
                [42, True, 100],
                b"1",
            ),  # Last item does not match general item schema
            (
                3,
                5,
                [42, True, "valid", "extra1", "extra2", "too_many"],
                b",",
            ),  # Exceeds maxItems
            (2, 3, [42], b"]"),  # Not enough items
            (1, 1, [42, True], b","),  # Too many items for maxItems
            (
                0,
                0,
                [42, True, "str"],
                b"4",
            ),  # maxItems set to 0, but array is not empty
            (
                3,
                5,
                [42, True],
                b"]",
            ),  # Array has one fewer item than required by minItems
        ],
    )
    def test_bad_with_prefix_and_items(
        self, min_items, max_items, bad_obj, failure_byte
    ):
        schema_obj = {
            "prefixItems": self.prefix_schema_obj,
            "items": self.items_schema_obj,
            "minItems": min_items,
            "maxItems": max_items,
            "type": "array",
        }
        bad_string = _to_compact_json(bad_obj)
        _check_match_failure(bad_string, failure_byte, schema_obj)

    @pytest.mark.parametrize(
        "min_items, max_items, bad_obj, failure_byte",
        [
            (
                2,
                2,
                [42],
                b"]",
            ),  # Array too short to meet minItems, despite matching prefixItems
            (
                1,
                2,
                [42, "not_bool"],
                b'"',
            ),  # Second item violates prefixItems type requirement
            (
                0,
                1,
                [42, True],
                b",",
            ),  # Array exceeds maxItems with valid prefixItems types
            (
                1,
                5,
                [42, True, "extra"],
                b",",
            ),  # Item beyond prefixItems with no "items" schema
            (0, 0, [42], b"4"),  # maxItems set to 0, but array is not empty
        ],
    )
    def test_bad_with_prefix(self, min_items, max_items, bad_obj, failure_byte):
        schema_obj = {
            "prefixItems": self.prefix_schema_obj,
            "minItems": min_items,
            "maxItems": max_items,
            "type": "array",
        }
        bad_string = _to_compact_json(bad_obj)
        _check_match_failure(bad_string, failure_byte, schema_obj)

    @pytest.mark.parametrize(
        "min_items, max_items, bad_obj, failure_byte",
        [
            (1, 2, ["hello", "world", "extra"], b","),  # Too many items for maxItems
            (2, 3, ["hello"], b"]"),  # Not enough items
            (2, 3, ["hello", 42], b"4"),  # Badly typed second item
            (0, 0, ["hello"], b'"'),  # maxItems set to 0, but array is not empty
        ],
    )
    def test_bad_with_items(self, min_items, max_items, bad_obj, failure_byte):
        schema_obj = {
            "items": self.items_schema_obj,
            "minItems": min_items,
            "maxItems": max_items,
            "type": "array",
        }
        bad_string = _to_compact_json(bad_obj)
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

    @pytest.mark.parametrize(
        "target_obj",
        [
            dict(all_cats=[]),
            dict(all_cats=[dict(name="Kasha")]),
            dict(all_cats=[dict(name="Dawon"), dict(name="Barong")]),
        ],
    )
    def test_simple_ref_alt(self, target_obj):
        # Uses 'definitions' rather than '$defs'
        schema = """{
        "definitions": {
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
                "$ref": "#/definitions/Cat"
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

    @pytest.mark.parametrize("temperature", [None, 0.1, 1])
    def test_nested_ref(self, temperature):
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
        _generate_and_check(target_obj, schema_obj, desired_temperature=temperature)


class TestAnyOf:
    @pytest.mark.parametrize("target_obj", [123, True])
    @pytest.mark.parametrize("temperature", [None, 0.1, 1])
    def test_anyOf_simple(self, target_obj, temperature):
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
        _generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

    @pytest.mark.parametrize(
        "target_obj",
        [
            dict(my_val=dict(my_int=1)),
            dict(my_val=dict(my_str="Some long string or other")),
        ],
    )
    @pytest.mark.parametrize("temperature", [None, 0.1, 1])
    def test_anyOf_objects(self, target_obj, temperature):
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
        _generate_and_check(target_obj, schema_obj, desired_temperature=temperature)


class TestAllOf:
    @pytest.mark.parametrize(
        "my_int",
        [0, 1, 100, 9876543210, 99, 737, 858, -1, -10, -20],
    )
    def test_allOf_integer(self, my_int):
        schema = """{
        "allOf" : [{ "type": "integer" }]
        }
        """
        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=my_int, schema=schema_obj)

        # The actual check
        _generate_and_check(my_int, schema_obj)

    def test_allOf_ref(self):
        schema = """{
            "definitions": {
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
            "type": "object",
            "properties": {
                "my_cat": {
                    "allOf": [
                        {
                            "$ref": "#/definitions/Cat"
                        }
                    ]
                }
            }
        }
        """

        target_obj = dict(my_cat=dict(name="Sampson"))
        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        _generate_and_check(target_obj, schema_obj)

    def test_allOf_bad_schema(self):
        schema = """{
        "allOf" : [{ "type": "integer" }, { "type": "number" }]
        }
        """
        # First sanity check what we're setting up
        schema_obj = json.loads(schema)

        TARGET_VALUE = 20
        validate(instance=TARGET_VALUE, schema=schema_obj)

        prepared_string = f"<s>{_to_compact_json(TARGET_VALUE)}"
        lm = models.Mock(prepared_string.encode())

        # Run with the mock model
        CAPTURE_KEY = "my_capture"
        with pytest.raises(ValueError) as ve:
            lm += gen_json(name=CAPTURE_KEY, schema=schema_obj)
        assert ve.value.args[0] == "Only support allOf with exactly one item"


class TestEnum:
    simple_schema = """{
        "enum": [1,"2",false]
    }
    """

    prefix_schema = """{
        "enum": ["aa", "bb", "cc"]
    }"""

    @pytest.mark.parametrize("target_obj", [1, "2", False])
    @pytest.mark.parametrize("temperature", [None, 0.1, 1])
    def test_enum(self, target_obj, temperature):
        # First sanity check what we're setting up
        schema_obj = json.loads(self.simple_schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        _generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

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
    @pytest.mark.parametrize("temperature", [None, 0.1, 1])
    def test_simple_additional_properties(self, target_obj, temperature):
        # First sanity check what we're setting up
        schema_obj = json.loads(self.simple_schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        _generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

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
    @pytest.mark.parametrize("temperature", [None, 0.1, 1])
    def test_properties_and_additional_properties(self, target_obj, temperature):
        # First sanity check what we're setting up
        schema_obj = json.loads(self.combined_schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        _generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

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
