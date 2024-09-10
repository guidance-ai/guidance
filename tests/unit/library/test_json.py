import json
from functools import partial
from typing import Any, Dict, Set, Union, Optional

import pytest
from jsonschema import validate

from guidance import json as gen_json
from guidance import models

from guidance.library._json import _to_compact_json, WHITESPACE, IGNORED_KEYS

from ...utils import check_match_failure as _check_match_failure
from ...utils import check_run_with_temperature
from ...utils import generate_and_check as _generate_and_check


def generate_and_check(
    target_obj: Any, schema_obj, desired_temperature: Optional[float] = None
):
    # Sanity check what we're being asked
    validate(instance=target_obj, schema=schema_obj)
    prepared_json = _to_compact_json(target_obj)
    assert json.loads(prepared_json) == target_obj

    # Now test that the grammar can recognize and generate prepared_json
    # We partial in the grammar_callable
    if desired_temperature is not None:
        grammar_callable = partial(
            gen_json, schema=schema_obj, temperature=desired_temperature
        )
    else:
        grammar_callable = partial(gen_json, schema=schema_obj)

    lm = _generate_and_check(
        grammar_callable,
        test_string=prepared_json,
    )
    check_run_with_temperature(lm, desired_temperature)


def check_match_failure(
    *,
    bad_string: str,
    good_bytes: Optional[bytes] = None,
    failure_byte: Optional[bytes] = None,
    allowed_bytes: Optional[Set[bytes]] = None,
    schema_obj: Dict[str, Any],
    maybe_whitespace: Optional[bool] = None,
    compact: bool = True,
):
    if (allowed_bytes is not None) and (maybe_whitespace is None):
        raise ValueError("If allowed_bytes is provided, maybe_whitespace must also be provided")
    if allowed_bytes is not None and maybe_whitespace and not compact:
        allowed_bytes = allowed_bytes.union(WHITESPACE)

    grammar = gen_json(schema=schema_obj, compact=compact)

    _check_match_failure(
        bad_string=bad_string,
        good_bytes=good_bytes,
        failure_byte=failure_byte,
        allowed_bytes=allowed_bytes,
        grammar=grammar,
    )


# Common sets of allowed_bytes
INTEGER_LEADING = {b"-", b"0", *{bytes([i]) for i in range(ord("1"), ord("9") + 1)}}
INTEGER_FOLLOWING = {bytes([i]) for i in range(ord("0"), ord("9") + 1)}
A_to_Z = {bytes([i]) for i in range(ord("A"), ord("Z") + 1)}

def test_null():
    schema = """{"type": "null" }"""

    target_obj = None

    # First sanity check what we're setting up
    schema_obj = json.loads(schema)
    validate(instance=target_obj, schema=schema_obj)

    generate_and_check(target_obj, schema_obj)


@pytest.mark.parametrize("target_obj", [True, False])
@pytest.mark.parametrize("temperature", [None, 0.1, 1])
def test_boolean(target_obj, temperature):
    schema = """{"type": "boolean" }"""

    # First sanity check what we're setting up
    schema_obj = json.loads(schema)
    validate(instance=target_obj, schema=schema_obj)

    generate_and_check(target_obj, schema_obj, desired_temperature=temperature)


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
        generate_and_check(my_int, schema_obj)

    @pytest.mark.parametrize(
        "compact", [True, False]
    )
    @pytest.mark.parametrize(
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes", "maybe_whitespace"],
        [
            ("9999a7777", b"9999", b"a", INTEGER_FOLLOWING, True),
            ("123, []", b"123", b",", INTEGER_FOLLOWING, True),
            ("a321", b"", b"a", INTEGER_LEADING, False),
            ("123789.456", b"123789", b".", INTEGER_FOLLOWING, True),
            ("[]", b"", b"[", INTEGER_LEADING, False),
            ('{"a":4}', b"", b"{", INTEGER_LEADING, False),
        ],
    )
    def test_bad_integer(self, bad_string, good_bytes, failure_byte, allowed_bytes, maybe_whitespace, compact):
        schema_obj = json.loads(TestInteger.schema)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=maybe_whitespace,
            compact=compact,
        )


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
        generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

    @pytest.mark.parametrize(
        "compact", [True, False]
    )
    @pytest.mark.parametrize(
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes", "maybe_whitespace"],
        [
            ("9999a7777", b"9999", b"a", {b"e", b"E", b".", *INTEGER_FOLLOWING}, True),
            ("123.6, []", b"123.6", b",", {b"e", b"E", *INTEGER_FOLLOWING}, True),
            ("a321", b"", b"a", INTEGER_LEADING, False),
            ("[]", b"", b"[", INTEGER_LEADING, False),
            ('{"a":4}', b"", b"{", INTEGER_LEADING, False),
        ],
    )
    def test_bad_number(self, bad_string, good_bytes, failure_byte, allowed_bytes, maybe_whitespace, compact):
        schema_obj = json.loads(TestNumber.schema)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=maybe_whitespace,
            compact=compact,
        )


class TestString:
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
    def test_smoke(self, my_string: str, temperature):
        schema = """{ "type": "string" }"""

        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=my_string, schema=schema_obj)

        # The actual check
        generate_and_check(my_string, schema_obj, desired_temperature=temperature)

    @pytest.mark.parametrize("my_string", ["aA", "aB", "aK", "aZ"])
    def test_regex(self, my_string: str):
        schema = """{ "type": "string", "pattern": "a[A-Z]"}"""

        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=my_string, schema=schema_obj)

        # The actual check
        generate_and_check(my_string, schema_obj)

    def test_regex_no_min_max_length(self):
        schema = """{ "type": "string", "pattern": "a[A-Z]", "minLength": 1 }"""
        schema_obj = json.loads(schema)

        lm = models.Mock("".encode())

        expected = "If a pattern is specified for a JSON string, minLength and maxLength must be left unspecified."
        with pytest.raises(ValueError) as ve:
            lm += gen_json(schema=schema_obj)
        assert ve.value.args[0] == expected

    @pytest.mark.parametrize(
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes"],
        [
            ('"ab"', b'"a', b"b", A_to_Z),
            ('"a1"', b'"a', b"1", A_to_Z),
        ],
    )
    def test_regex_bad(self, bad_string: str, good_bytes, failure_byte, allowed_bytes):
        # Note that the strings being fed in include the double quotes required
        # to make them JSON strings
        schema = """{ "type": "string", "pattern": "a[A-Z]"}"""
        schema_obj = json.loads(schema)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=False,
            compact=True,
        )

    @pytest.mark.parametrize(
        "string", ["aA\u001f", '"""']
    )
    def test_regex_properly_escaped_good(self, string):
        schema_obj = {"type": "string", "pattern": r".{3}"}
        # First sanity check what we're setting up
        validate(instance=string, schema=schema_obj)
        # The actual check
        generate_and_check(string, schema_obj)

    @pytest.mark.parametrize(
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes"],
        [
            (
                '"\\u001f\\u001f\u001f',
                b'"\\u001f\\u001f', # able to match the first two stringified bytes
                '\u001f'.encode(), # fails on a literal \x1f byte
                None # hard to write a set of allowed bytes here
            ),
        ],
    )
    def test_regex_properly_escaped_bad(self, bad_string: str, good_bytes, failure_byte, allowed_bytes):
        # Note that the strings being fed in include the double quotes required
        # to make them JSON strings
        schema_obj = {"type": "string", "pattern": r".{3}"}
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=False,
            compact=True,
        )


    @pytest.mark.parametrize(
        "my_string", ["a", "bb", "ccc", "150", ",?", ".\t\n", "(){", "aA7", "\\9O"]
    )
    def test_min_and_maxLength(self, my_string: str):
        schema = """{ "type": "string", "minLength": 1, "maxLength": 3}"""

        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=my_string, schema=schema_obj)

        # The actual check
        generate_and_check(my_string, schema_obj)

    @pytest.mark.parametrize(
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes"],
        [
            ('""', b'"', b'"', None),
            ('"dddd"', b'"ddd', b"d", {b'"'}),
        ],
    )
    def test_min_and_maxLength_bad(self, bad_string: str, good_bytes, failure_byte, allowed_bytes):
        # Note that the strings being fed in include the double quotes required
        # to make them JSON strings
        schema = """{ "type": "string", "minLength": 1, "maxLength": 3}"""
        schema_obj = json.loads(schema)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=False,
            compact=True,
        )

    @pytest.mark.parametrize(
        "my_string",
        [
            "a",
            "bb",
            "ccc",
            "150",
            ",?",
            ".\t\n",
            "(){",
            "aA7",
            "\\9O",
            "This is a really long string" * 10,
        ],
    )
    def test_minLength(self, my_string: str):
        schema = """{ "type": "string", "minLength": 1}"""

        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=my_string, schema=schema_obj)

        # The actual check
        generate_and_check(my_string, schema_obj)

    @pytest.mark.parametrize(
        "my_string",
        [
            "",
            "a",
            "bb",
            "ccc",
            "150",
            ",?",
            ".\t\n",
            "(){",
            "aA7",
            "\\9O",
            "This is a really long string" * 10,
        ],
    )
    def test_minLength_zero(self, my_string: str):
        schema = """{ "type": "string", "minLength": 0}"""

        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=my_string, schema=schema_obj)

        # The actual check
        generate_and_check(my_string, schema_obj)

    @pytest.mark.parametrize(
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes"],
        [
            ('""', b'"', b'"', None),
            ('"a"', b'"a', b'"', None),
        ],
    )
    def test_minLength_bad(self, bad_string: str, good_bytes, failure_byte, allowed_bytes):
        # Note that the strings being fed in include the double quotes required
        # to make them JSON strings
        schema = """{ "type": "string", "minLength": 2}"""
        schema_obj = json.loads(schema)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=False,
            compact=True,
        )

    @pytest.mark.parametrize(
        "my_string",
        [
            "a",
            "bb",
            "ccc",
            "150",
            ",?",
            ".\t\n",
            "(){",
            "aA7",
            "\\9O",
            "",
        ],
    )
    def test_maxLength(self, my_string: str):
        schema = """{ "type": "string", "maxLength": 3}"""

        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=my_string, schema=schema_obj)

        # The actual check
        generate_and_check(my_string, schema_obj)

    def test_maxLength_zero(self):
        schema = """{ "type": "string", "maxLength": 0}"""
        my_string = ""

        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=my_string, schema=schema_obj)

        # The actual check
        generate_and_check(my_string, schema_obj)

    @pytest.mark.parametrize(
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes"],
        [
            ('"aaa"', b'"aa', b"a", {b'"'}),
            ('"1111"', b'"11', b"1", {b'"'}),
        ],
    )
    def test_maxLength_bad(self, bad_string: str, good_bytes, failure_byte, allowed_bytes):
        # Note that the strings being fed in include the double quotes required
        # to make them JSON strings
        schema = """{ "type": "string", "maxLength": 2}"""
        schema_obj = json.loads(schema)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=False,
            compact=True,
        )


class TestSimpleObject:
    # These are objects without cross references
    def test_object_with_single_property(self):
        schema = """{
            "type": "object",
            "properties": {
                "a" : {"type": "integer"}
            },
            "required": ["a"]
        }
    """
        target_obj = dict(a=1)

        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        generate_and_check(target_obj, schema_obj)

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
            },
            "required": ["a", "b", "c", "d", "e", "f", "g", "h"]
        }
    """
        target_obj = dict(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8)

        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        generate_and_check(target_obj, schema_obj)

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
            },
            "required": ["name", "info"]
        }
    """
        target_obj = dict(name="my product", info=dict(a=1, b=2))

        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

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
        generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

    @pytest.mark.parametrize(
        "compact", [True, False]
    )
    @pytest.mark.parametrize(
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes", "maybe_whitespace"],
        [
            ("9999a7777", b"", b"9", {b"{"}, False),
            ('{"a":1255.4567}', b'{"a":1255', b".", {b"}", *INTEGER_FOLLOWING}, True),
            ('{"a":"123"}', b'{"a":', b'"', INTEGER_LEADING, True),
        ],
    )
    def test_bad_object(self, bad_string, good_bytes, failure_byte, allowed_bytes, maybe_whitespace, compact):
        schema = """{
            "type": "object",
            "properties": {
                "a" : {"type": "integer"}
            },
            "required": ["a"],
            "additionalProperties": false
        }
    """
        schema_obj = json.loads(schema)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=maybe_whitespace,
            compact=compact,
        )


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
        generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

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
        generate_and_check(target_obj, schema_obj)

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
        generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

    @pytest.mark.parametrize(
        "compact", [True, False]
    )
    @pytest.mark.parametrize(
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes", "maybe_whitespace"],
        [
            ("9999a7777", b"", b"9", {b"["}, False),
            ("[321.654]", b"[321", b".", {b"]", b",", *INTEGER_FOLLOWING}, True),
            ('["123"]', b"[", b'"', {b"]", *INTEGER_LEADING}, True),
        ],
    )
    def test_bad_object(self, bad_string, good_bytes, failure_byte, allowed_bytes, maybe_whitespace, compact):
        schema = """{
        "type" : "array",
        "items" : {
                "type" : "integer"
            }
        }"""
        schema_obj = json.loads(schema)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=maybe_whitespace,
            compact=compact,
        )


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
        generate_and_check(target_obj, schema_obj)

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
        generate_and_check(target_obj, schema_obj)

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
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "compact", [True, False]
    )
    @pytest.mark.parametrize(
        "min_items, max_items, bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace",
        [
            (
                1,
                4,
                [42, "string_not_bool", "hello", "extra"],
                b"[42,",
                b'"',
                {b"t", b"f"},
                True,
            ),  # Second item does not match prefix schema
            (
                0,
                3,
                [42, True, 100],
                b"[42,true,",
                b"1",
                {b'"'},
                True,
            ),  # Last item does not match general item schema
            (
                3,
                5,
                [42, True, "valid", "extra1", "extra2", "too_many"],
                b'[42,true,"valid","extra1","extra2"',
                b",",
                {b"]"},
                True,
            ),  # Exceeds maxItems
            (
                2,
                3,
                [42],
                b"[42",
                b"]",
                {b",", *INTEGER_FOLLOWING},
                True,
            ),  # Not enough items
            (
                1,
                1,
                [42, True],
                b"[42",
                b",",
                {b"]", *INTEGER_FOLLOWING},
                True,
            ),  # Too many items for maxItems
            (
                0,
                0,
                [42, True, "str"],
                b"[",
                b"4",
                {b"]"},
                True,
            ),  # maxItems set to 0, but array is not empty
            (
                3,
                5,
                [42, True],
                b"[42,true",
                b"]",
                {b","},
                True,
            ),  # Array has one fewer item than required by minItems
        ],
    )
    def test_bad_with_prefix_and_items(
        self, min_items, max_items, bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace, compact
    ):
        schema_obj = {
            "prefixItems": self.prefix_schema_obj,
            "items": self.items_schema_obj,
            "minItems": min_items,
            "maxItems": max_items,
            "type": "array",
        }
        bad_string = _to_compact_json(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=maybe_whitespace,
            compact=compact,
        )

    @pytest.mark.parametrize(
        "compact", [True, False]
    )
    @pytest.mark.parametrize(
        "min_items, max_items, bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace",
        [
            (
                2,
                2,
                [42],
                b"[42",
                b"]",
                {b",", *INTEGER_FOLLOWING},
                True,
            ),  # Array too short to meet minItems, despite matching prefixItems
            (
                1,
                2,
                [42, "not_bool"],
                b"[42,",
                b'"',
                {b"t", b"f"},
                True,
            ),  # Second item violates prefixItems type requirement
            (
                0,
                1,
                [42, True],
                b"[42",
                b",",
                {b"]", *INTEGER_FOLLOWING},
                True,
            ),  # Array exceeds maxItems with valid prefixItems types
            (
                1,
                5,
                [42, True, "extra"],
                b"[42,true",
                b",",
                {b"]"},
                True,
            ),  # Item beyond prefixItems with no "items" schema
            (
                0,
                0,
                [42],
                b"[",
                b"4",
                {b"]"},
                True,
            ),  # maxItems set to 0, but array is not empty
        ],
    )
    def test_bad_with_prefix(
        self, min_items, max_items, bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace, compact
    ):
        schema_obj = {
            "prefixItems": self.prefix_schema_obj,
            "items": False,
            "minItems": min_items,
            "maxItems": max_items,
            "type": "array",
        }
        bad_string = _to_compact_json(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=maybe_whitespace,
            compact=compact,
        )

    @pytest.mark.parametrize(
        "compact", [True, False]
    )
    @pytest.mark.parametrize(
        "min_items, max_items, bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace",
        [
            (
                1,
                2,
                ["hello", "world", "extra"],
                b'["hello","world"',
                b",",
                {b"]"},
                True,
            ),  # Too many items for maxItems
            (
                2,
                3,
                ["hello"],
                b'["hello"',
                b"]",
                {b","},
                True,
            ),  # Not enough items
            (
                2,
                3,
                ["hello", 42],
                b'["hello",',
                b"4",
                {b'"'},
                True,
            ),  # Badly typed second item
            (
                0,
                0,
                ["hello"],
                b"[",
                b'"',
                {b"]"},
                True,
            ),  # maxItems set to 0, but array is not empty
        ],
    )
    def test_bad_with_items(
        self, min_items, max_items, bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace, compact
    ):
        schema_obj = {
            "items": self.items_schema_obj,
            "minItems": min_items,
            "maxItems": max_items,
            "type": "array",
        }
        bad_string = _to_compact_json(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=maybe_whitespace,
            compact=compact,
        )


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
        generate_and_check(target_obj, schema_obj)

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
        generate_and_check(target_obj, schema_obj)

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
        generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

    @pytest.mark.parametrize("temperature", [None, 0.1, 1])
    def test_multiple_refs_to_same_def(self, temperature):
        schema = """{
        "$defs": {
            "A": {
                "properties": {
                    "name": {
                        "type": "string"
                    }
                },
                "required": ["name"],
                "type": "object"
            }
        },
        "properties": {
            "A1": {
                "$ref": "#/$defs/A"
            },
            "A2": {
                "$ref": "#/$defs/A"
            }
        },
        "required": ["A1", "A2"],
        "type": "object"
        }"""

        target_obj = dict(A1=dict(name="Romulus"), A2=dict(name="Remus"))

        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        generate_and_check(target_obj, schema_obj, desired_temperature=temperature)


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
        generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

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
        "required": ["my_str"],
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
        "required": ["my_int"],
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
    "required": ["my_val"],
    "title": "C",
    "type": "object"
    }
    """
        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        generate_and_check(target_obj, schema_obj, desired_temperature=temperature)


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
        generate_and_check(my_int, schema_obj)

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
            },
            "required": ["my_cat"]
        }
        """

        target_obj = dict(my_cat=dict(name="Sampson"))
        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        generate_and_check(target_obj, schema_obj)

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

class TestOneOf:
    @pytest.mark.parametrize("target_obj", [123, 42])
    def test_oneOf_simple(self, target_obj):
        schema = """{
            "oneOf" : [{ "type": "integer" }]
        }
        """
        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        generate_and_check(target_obj, schema_obj)


    @pytest.mark.parametrize("target_obj", [123, True])
    def test_oneOf_compound(self, target_obj):
        schema = """{
        "oneOf" : [{ "type": "integer" }, { "type": "boolean" }]
        }
        """
        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check; we expect a warning here because oneOf is not fully supported
        with pytest.warns() as record:
            generate_and_check(target_obj, schema_obj)
        assert len(record) == 1
        assert record[0].message.args[0].startswith("oneOf not fully supported")


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
        generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

    @pytest.mark.parametrize(
        "compact", [True, False]
    )
    @pytest.mark.parametrize(
        "bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace",
        [
            ("1", b'"', b"1", {b"2"}, False),
            (2, b"", b"2", {b'"', b"1", b"f"}, False),
            (True, b"", b"t", {b'"', b"1", b"f"}, False),
        ],
    )
    def test_bad_enum(self, bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace, compact):
        schema_obj = json.loads(self.simple_schema)
        bad_string = _to_compact_json(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=maybe_whitespace,
            compact=compact,
        )

    @pytest.mark.parametrize(
        "compact", [True, False]
    )
    @pytest.mark.parametrize(
        "bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace",
        [
            ("ab", b'"a', b"b", {b"a"}, False),
            ("bc", b'"b', b"c", {b"b"}, False),
            ("ca", b'"c', b"a", {b"c"}, False),
        ],
    )
    def test_bad_prefix_enum(self, bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace, compact):
        schema_obj = json.loads(self.prefix_schema)
        bad_string = _to_compact_json(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=maybe_whitespace,
            compact=compact,
        )


class TestConst:
    def test_constant_int(self):
        # First sanity check what we're setting up
        schema_obj = {"const": 1}
        target_obj = 1
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        generate_and_check(target_obj, schema_obj)

    def test_constant_string(self):
        # First sanity check what we're setting up
        schema_obj = {"const": "hello"}
        target_obj = "hello"
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        generate_and_check(target_obj, schema_obj)

    def test_constant_array(self):
        # First sanity check what we're setting up
        schema_obj = {"const": [1, 2, 3]}
        target_obj = [1, 2, 3]
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        generate_and_check(target_obj, schema_obj)

    def test_constant_object(self):
        # First sanity check what we're setting up
        schema_obj = {"const": {"a": 1, "b": 2}}
        target_obj = {"a": 1, "b": 2}
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        generate_and_check(target_obj, schema_obj)

    def test_nested_constant(self):
        # First sanity check what we're setting up
        schema_obj = {"type": "object", "properties": {"a": {"const": 1}}, "required": ["a"]}
        target_obj = {"a": 1}
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        generate_and_check(target_obj, schema_obj)

    def test_constant_precedence(self):
        schema_obj = {"type": "integer", "const": 1}
        bad_string = _to_compact_json(2)

        check_match_failure(
            bad_string=bad_string,
            good_bytes=b"",
            failure_byte=b"2",
            allowed_bytes={b"1"},
            schema_obj=schema_obj,
            maybe_whitespace=False,
            compact=True,
        )


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
                {"type": "string"},
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
        },
    "required": ["mystr"]
    }
    """

    @pytest.mark.parametrize("target_obj", [{}, {"a": 1}, {"a": 1, "b": 2}])
    @pytest.mark.parametrize("temperature", [None, 0.1, 1])
    def test_simple_additional_properties(self, target_obj, temperature):
        # First sanity check what we're setting up
        schema_obj = json.loads(self.simple_schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

    @pytest.mark.parametrize(
        "compact", [True, False]
    )
    @pytest.mark.parametrize(
        "bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace",
        [
            ({"a": "1"}, b'{"a":', b'"', INTEGER_LEADING, True),
            (
                {"a": 1, "b": 1.5},
                b'{"a":1,"b":1',
                b".",
                {b",", b"}", *INTEGER_FOLLOWING},
                True,
            ),
        ],
    )
    def test_simple_bad_type(self, bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace, compact):
        schema_obj = json.loads(self.simple_schema)
        bad_string = _to_compact_json(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=maybe_whitespace,
            compact=compact,
        )

    @pytest.mark.parametrize(
        "target_obj", [{}, {"a": 1}, {"a": "2"}, {"a": 1, "b": "2"}]
    )
    def test_anyOf_additional_properties(self, target_obj):
        # First sanity check what we're setting up
        schema_obj = json.loads(self.anyOf_schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "compact", [True, False]
    )
    @pytest.mark.parametrize(
        "bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace",
        [
            ({"a": 1.5}, b'{"a":1', b".", {b",", b"}", *INTEGER_FOLLOWING}, True),
            ({"a": True}, b'{"a":', b"t", {b'"', *INTEGER_LEADING}, True),
            (
                {"a": 1, "b": False},
                b'{"a":1,"b":',
                b"f",
                {b'"', *INTEGER_LEADING},
                True,
            ),
        ],
    )
    def test_anyOf_bad_type(self, bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace, compact):
        schema_obj = json.loads(self.anyOf_schema)
        bad_string = _to_compact_json(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=maybe_whitespace,
            compact=compact,
        )

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
        generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

    @pytest.mark.parametrize(
        "compact", [True, False]
    )
    @pytest.mark.parametrize(
        "bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace",
        [
            ({}, b"{", b"}", {b'"'}, True),
            ({"a": 1}, b'{"', b"a", {b"m"}, False),
            ({"a": 1, "b": 2}, b'{"', b"a", {b"m"}, False),
        ],
    )
    def test_combined_missing_properties(
        self, bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace, compact
    ):
        schema_obj = json.loads(self.combined_schema)
        bad_string = _to_compact_json(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=maybe_whitespace,
            compact=compact,
        )

    @pytest.mark.parametrize(
        "compact", [True, False]
    )
    @pytest.mark.parametrize(
        "bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace",
        [
            ({"mystr": 1}, b'{"mystr":', b"1", {b'"'}, True),
            ({"mystr": 1, "a": 2}, b'{"mystr":', b"1", {b'"'}, True),
            (
                {"mystr": "hello", "a": False},
                b'{"mystr":"hello","a":',
                b"f",
                INTEGER_LEADING,
                True,
            ),
        ],
    )
    def test_combined_bad_type(self, bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace, compact):
        schema_obj = json.loads(self.combined_schema)
        bad_string = _to_compact_json(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=maybe_whitespace,
            compact=compact,
        )


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
            "required": ["my_str", "next"],
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
    },
    "required": ["my_list"]
}
        """
        # First sanity check what we're setting up
        schema_obj = json.loads(schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        generate_and_check(target_obj, schema_obj)


class TestEmptySchemas:
    empty_schema = "{}"
    nested_empty_schema_with_props = """{
    "properties" : {
        "a": {},
        "b": {"type": "number"}
    },
    "required" : ["a", "b"],
    "type" : "object"
    }"""

    @pytest.mark.parametrize(
        "target_obj",
        [
            1,
            "2",
            False,
            [1, 2, 3],
            {"a": 1},
            None,
            [{"a": 1}],
            {"a": [1, 2, 3]},
            {"a": {"b": 1}},
        ],
    )
    @pytest.mark.parametrize("temperature", [None, 0.1, 1])
    def test_empty_schema(self, target_obj, temperature):
        # First sanity check what we're setting up
        schema_obj = json.loads(self.empty_schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

    @pytest.mark.parametrize(
        "compact", [True, False]
    )
    @pytest.mark.parametrize(
        "bad_string, good_bytes, failure_byte, allowed_bytes, maybe_whitespace",
        [
            # {} is not carte blanche for malformed JSON
            ("{a:1}", b"{", b"a", {b'"', b"}"}, True),
            (
                "[1,2} ",
                b"[1,2",
                b"}",
                {b",", b"]", b"e", b"E", b".", *INTEGER_FOLLOWING},
                True,
            ),
            ("123a", b"123", b"a", {b"e", b"E", b".", *INTEGER_FOLLOWING}, True),
            (
                "]",
                b"",
                b"]",
                {
                    b"[",
                    b"{",
                    b'"',
                    b"t",
                    b"f",
                    b"n",
                    *INTEGER_LEADING,
                },
                False,
            ),
        ],
    )
    def test_bad_empty_schema(
        self, bad_string, good_bytes, failure_byte, allowed_bytes, maybe_whitespace, compact
    ):
        schema_obj = json.loads(self.empty_schema)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=maybe_whitespace,
            compact=compact,
        )

    @pytest.mark.parametrize(
        "schema_obj",
        [
            # Empty property
            {"type": "object", "properties": {"a": {}}, "required": ["a"]},
            # Empty reference
            {"type": "object", "properties": {"a": {"$ref": "#/$defs/A"}}, "$defs": {"A": {}}, "required": ["a"]},
        ],
    )
    @pytest.mark.parametrize(
        "target_obj",
        [
            {"a": 1},
            {"a": "2"},
            {"a": False},
            {"a": [1, 2, 3]},
            {"a": {"b": 1}},
            {"a": None},
            {"a": [{"b": 1}]},
            {"a": {"b": [1, 2, 3]}},
            {"a": {"b": {"c": 1}}},
        ],
    )
    @pytest.mark.parametrize("temperature", [None, 0.1, 1])
    def test_nested_empty_schema(self, schema_obj, target_obj, temperature):
        # First sanity check what we're setting up
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

    @pytest.mark.parametrize(
        "schema_obj",
        [
            # Empty property
            {"type": "object", "properties": {"a": {}}, "required": ["a"]},
            # Empty reference
            {"type": "object", "properties": {"a": {"$ref": "#/$defs/A"}}, "$defs": {"A": {}}, "required": ["a"]},
        ],
    )
    @pytest.mark.parametrize(
        "compact", [True, False]
    )
    @pytest.mark.parametrize(
        "bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace",
        [
            # Missing property -- presence of {} deeper in the schema isn't carte blanche
            ({"b": 42}, b'{"', b"b", {b"a"}, False),
        ],
    )
    def test_nested_empty_schema_bad(
        self, schema_obj, bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace, compact
    ):
        bad_string = _to_compact_json(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=maybe_whitespace,
            compact=compact,
        )

    @pytest.mark.parametrize(
        "target_obj",
        [
            {"a": 1, "b": 2},
            {"a": "2", "b": 1.998},
            {"a": False, "b": -3.14},
            {"a": [1, 2, 3], "b": 42},
            {"a": {"b": 1}, "b": 0.2},
            {"a": None, "b": 5e-4},
            {"a": [{"b": 1}], "b": -5e2},
            {"a": {"b": [1, 2, 3]}, "b": 1},
            {"a": {"b": {"c": 1}}, "b": -1},
        ],
    )
    @pytest.mark.parametrize("temperature", [None, 0.1, 1])
    def test_nested_empty_schema_with_props(self, target_obj, temperature):
        # First sanity check what we're setting up
        schema_obj = json.loads(self.nested_empty_schema_with_props)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        generate_and_check(target_obj, schema_obj, desired_temperature=temperature)

    @pytest.mark.parametrize(
        "compact", [True, False]
    )
    @pytest.mark.parametrize(
        "bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace",
        [
            # Missing property -- presence of {} deeper in the schema isn't carte blanche
            ({"b": 42}, b'{"', b"b", {b"a"}, False),
        ],
    )
    def test_nested_empty_schema_with_props_bad(
        self, bad_obj, good_bytes, failure_byte, allowed_bytes, maybe_whitespace, compact
    ):
        schema_obj = json.loads(self.nested_empty_schema_with_props)

        bad_string = _to_compact_json(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
            maybe_whitespace=maybe_whitespace,
            compact=compact,
        )

    @pytest.mark.parametrize(
        "schema_obj",
        [
            {"type": "array"},
            {"type": "array", "items": {}},
            {"type": "array", "items": True},
        ],
    )
    def test_items(self, schema_obj):
        generate_and_check(
            [1, 0.4, "hello", False, None, {"a": 42}, [1, 2, 3, "four"]], schema_obj
        )

    @pytest.mark.parametrize(
        "compact", [True, False]
    )
    def test_no_items(self, compact):
        schema_obj = {"type": "array", "items": False}
        check_match_failure(
            bad_string="[42]",
            good_bytes=b"[",
            failure_byte=b"4",
            allowed_bytes={b"]"},  # array must be empty
            schema_obj=schema_obj,
            maybe_whitespace=True,
            compact=compact,
        )

    @pytest.mark.parametrize(
        "schema_obj",
        [
            {"type": "object"},
            {"type": "object", "additionalProperties": {}},
            {"type": "object", "additionalProperties": True},
        ],
    )
    def test_additionalProperties(self, schema_obj):
        generate_and_check(
            {
                "a": 1,
                "b": 0.4,
                "c": "hello",
                "d": False,
                "e": None,
                "f": {"a": 42},
                "g": [1, 2, 3, "four"],
            },
            schema_obj,
        )

    @pytest.mark.parametrize(
        "compact", [True, False]
    )
    def test_no_additionalProperties(self, compact):
        schema_obj = {"type": "object", "additionalProperties": False}
        check_match_failure(
            bad_string='{"a": 42}',
            good_bytes=b"{",
            failure_byte=b'"',
            allowed_bytes={b"}"},  # object must be empty
            schema_obj=schema_obj,
            maybe_whitespace=True,
            compact=compact,
        )

def test_ignored_keys_allowed_as_properties():
    schema_obj = {
        "type": "object",
        "properties": {
            key: {"type": "string"} for key in IGNORED_KEYS
        },
        "required": list(IGNORED_KEYS),
    }
    target_obj = {key: "value" for key in IGNORED_KEYS}
    generate_and_check(target_obj, schema_obj)

class TestRequiredProperties:
    schema_obj = {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "number"},
            "c": {"type": "boolean"},
        },
        "additionalProperties": True
    }
    ALL_REQUIRED = ["a", "b", "c"]
    SOME_REQUIRED_SUBSETS = [[], ["a"], ["b"], ["c"], ["a", "b"], ["a", "c"], ["b", "c"], ["a", "b", "c"]]
    NONE_REQUIRED: list[str] = []

    @pytest.mark.parametrize(
        "extra_items",
        [
            {},
            {"d": "hello"},
            {"d": 42, "e": True},
        ]
    )
    def test_all_required_good(self, extra_items):
        schema_obj = {**self.schema_obj, "required": self.ALL_REQUIRED}
        target_obj = {"a": "hello", "b": 42, "c": True, **extra_items}
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_obj",
        [
            # Missing one
            ({"a": "hello", "b": 42}),
            ({"a": "hello", "c": True}),
            ({"b": 42, "c": True}),
            # Missing two
            ({"a": "hello"}),
            ({"b": 42}),
            ({"c": True}),
            # Missing all
            ({}),
        ]
    )
    def test_all_required_bad(self, bad_obj):
        schema_obj = {**self.schema_obj, "required": self.ALL_REQUIRED}
        check_match_failure(
            bad_string=_to_compact_json(bad_obj),
            schema_obj=schema_obj,
            compact=True
        )

    @pytest.mark.parametrize(
        "extra_items",
        [
            {},
            {"d": "hello"},
            {"d": 42, "e": True},
        ]
    )
    @pytest.mark.parametrize(
        "required",
        SOME_REQUIRED_SUBSETS,
    )
    def test_some_required_good(self, required, extra_items):
        base_obj = {"a": "hello", "b": 42, "c": True}
        schema_obj = {**self.schema_obj, "required": required}

        for subset in self.SOME_REQUIRED_SUBSETS:
            if set(required).issubset(subset):
                target_obj = {**{key: base_obj[key] for key in subset}, **extra_items}
                generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "required",
        SOME_REQUIRED_SUBSETS,
    )
    def test_some_required_bad(self, required):
        base_obj = {"a": "hello", "b": 42, "c": True}
        schema_obj = {**self.schema_obj, "required": required}

        for subset in self.SOME_REQUIRED_SUBSETS:
            if not set(required).issubset(subset):
                bad_obj = {key: base_obj[key] for key in subset}
                check_match_failure(
                    bad_string=_to_compact_json(bad_obj),
                    schema_obj=schema_obj,
                    compact=True
                )

    # No equivalent "bad" tests for none required, as the schema is satisfied by an empty object
    @pytest.mark.parametrize(
        "extra_items",
        [
            {},
            {"d": "hello"},
            {"d": 42, "e": True},
        ]
    )
    @pytest.mark.parametrize(
        "target_obj",
        [
            {},
            {"a": "hello"},
            {"b": 42},
            {"c": True},
            {"a": "hello", "b": 42},
            {"a": "hello", "c": True},
            {"b": 42, "c": True},
            {"a": "hello", "b": 42, "c": True},
        ]
    )
    def test_none_required(self, target_obj, extra_items):
        schema_obj = {**self.schema_obj, "required": self.NONE_REQUIRED}
        generate_and_check({**target_obj, **extra_items}, schema_obj)

class TestRequiredPropertiesScaling:
    @pytest.mark.parametrize(
        "num_properties",
        [1, 2, 3, 4, 5, 10, 20, 50, 100]
    )
    def test_many_optional_properties_doesnt_blow_up(self, num_properties):
        schema_obj = {
            "type": "object",
            "properties": {
                f"prop_{i}": {"type": "string"} for i in range(num_properties)
            },
            "required": [] # Empty should be worst-case scenario
        }
        from guidance.library._json import _gen_list
        _gen_list.__wrapped__.cache_clear()
        _ = gen_json(
            schema=schema_obj,
        )
        cache_info = _gen_list.__wrapped__.cache_info()

        # Theoretical number of cache misses under the current implementation
        expected_misses = 2*num_properties - 1
        MISSES_MAGIC_NUMBER = 5 # Where in the world is this coming from?
        assert 0 < cache_info.misses <= expected_misses + MISSES_MAGIC_NUMBER
        # NOTE: that if the cache maxsize is hit, the number of misses will be more than expected

        # Theoretical number of total calls under the current implementation
        expected_calls = num_properties*(num_properties - 1) // 2
        CALLS_MAGIC_NUMBER = 12 # Where in the world is this coming from?
        assert 0 < cache_info.hits + cache_info.misses <= expected_calls + CALLS_MAGIC_NUMBER

    @pytest.mark.parametrize(
        "num_properties",
        [1, 2, 3, 4, 5, 10, 20, 50, 100]
    )
    def test_all_required_properties_doesnt_blow_up(self, num_properties):
        schema_obj = {
            "type": "object",
            "properties": {
                f"prop_{i}": {"type": "string"} for i in range(num_properties)
            },
            "required": [f"prop_{i}" for i in range(num_properties)]
        }
        from guidance.library._json import _gen_list
        _gen_list.__wrapped__.cache_clear()
        _ = gen_json(
            schema=schema_obj,
        )
        cache_info = _gen_list.__wrapped__.cache_info()

        # Theoretical number of cache misses under the current implementation
        expected_misses = num_properties
        MISSES_MAGIC_NUMBER = 4
        assert 0 < cache_info.misses <= expected_misses + MISSES_MAGIC_NUMBER
        HITS_MAGIC_NUMBER = 1
        expected_hits = 0
        assert cache_info.hits <= expected_hits + HITS_MAGIC_NUMBER

class TestBooleanSchema:
    @pytest.mark.parametrize(
        "target_obj",
        [
            123,
            "hello",
            [1, 2, 3],
            {"a": 1},
            None,
            [{"a": 1}],
            {"a": [1, 2, 3]},
            {"a": {"b": 1}},
            False,
            True
        ],
    )
    def test_true_schema(self, target_obj):
        # should be the same as an empty schema
        schema_obj = True
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "schema_obj",
        [
            False,
            {"type": "object", "properties": {"a": False}, "required": ["a"]},
        ]
    )
    def test_false_schema(self, schema_obj):
        with pytest.raises(ValueError) as ve:
            gen_json(schema=schema_obj)
        assert ve.value.args[0] == "No valid JSON can be generated from a schema of `False`"
