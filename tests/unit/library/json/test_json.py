import json
import re
import warnings
from json import dumps as json_dumps

import pytest
from jsonschema import ValidationError, validate

from guidance import json as gen_json
from guidance import models

from .utils import check_match_failure, generate_and_check

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
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes"],
        [
            ("9999a7777", b"9999", b"a", INTEGER_FOLLOWING),
            ("123, []", b"123", b",", INTEGER_FOLLOWING),
            ("a321", b"", b"a", INTEGER_LEADING),
            ("123789.456", b"123789", b".", INTEGER_FOLLOWING),
            ("[]", b"", b"[", INTEGER_LEADING),
            ('{"a": 4}', b"", b"{", INTEGER_LEADING),
        ],
    )
    def test_bad_integer(self, bad_string, good_bytes, failure_byte, allowed_bytes):
        schema_obj = json.loads(TestInteger.schema)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
        )

    @pytest.mark.parametrize(
        "schema",
        [
            {"type": "integer", "minimum": 5, "maximum": 4},
            {"type": "integer", "minimum": 5, "exclusiveMaximum": 5},
            {"type": "integer", "exclusiveMinimum": 5, "maximum": 5},
        ],
    )
    def test_unsatisfiable_min_max(self, schema):
        with pytest.raises(ValueError) as ve:
            _ = gen_json(schema=schema)
        assert re.fullmatch(
            r"Unsatisfiable schema: (exclusiveMinimum|minimum) \(5\) is (greater than|equal to) (exclusiveMaximum|maximum) \((4|5)\)",
            ve.value.args[0],
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
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes"],
        [
            ("9999a7777", b"9999", b"a", {b"e", b"E", b".", *INTEGER_FOLLOWING}),
            ("123.6, []", b"123.6", b",", {b"e", b"E", *INTEGER_FOLLOWING}),
            ("a321", b"", b"a", INTEGER_LEADING),
            ("[]", b"", b"[", INTEGER_LEADING),
            ('{"a": 4}', b"", b"{", INTEGER_LEADING),
        ],
    )
    def test_bad_number(self, bad_string, good_bytes, failure_byte, allowed_bytes):
        schema_obj = json.loads(TestNumber.schema)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
        )

    @pytest.mark.parametrize(
        "schema",
        [
            {"type": "integer", "minimum": 5, "maximum": 4},
            {"type": "integer", "minimum": 5, "exclusiveMaximum": 5},
            {"type": "integer", "exclusiveMinimum": 5, "maximum": 5},
        ],
    )
    def test_unsatisfiable_min_max(self, schema):
        with pytest.raises(ValueError) as ve:
            _ = gen_json(schema=schema)
        assert ve.value.args[0].startswith("Unsatisfiable schema")
        assert re.fullmatch(
            r"Unsatisfiable schema: (exclusiveMinimum|minimum) \(5\) is (greater than|equal to) (exclusiveMaximum|maximum) \((4|5)\)",
            ve.value.args[0],
        )


class TestBoundedNumeric:
    @pytest.mark.parametrize(
        "instance, schema, should_pass",
        [
            # --- Integer type tests ---
            (5, {"type": "integer", "minimum": 5}, True),
            (-5, {"type": "integer", "minimum": -5}, True),
            pytest.param(
                *(5.0, {"type": "integer", "minimum": 5}, True),
                marks=pytest.mark.xfail(reason="JSON technically allows trailing zeroes, but we currently don't"),
            ),
            pytest.param(
                *(-5.0, {"type": "integer", "minimum": -5}, True),
                marks=pytest.mark.xfail(reason="JSON technically allows trailing zeroes, but we currently don't"),
            ),
            (5.1, {"type": "integer", "minimum": 5}, False),
            (-5.1, {"type": "integer", "minimum": -5}, False),
            (4, {"type": "integer", "minimum": 5}, False),
            (-6, {"type": "integer", "minimum": -5}, False),
            (6, {"type": "integer", "exclusiveMinimum": 5}, True),
            (-4, {"type": "integer", "exclusiveMinimum": -5}, True),
            (5, {"type": "integer", "exclusiveMinimum": 5}, False),
            (-5, {"type": "integer", "exclusiveMinimum": -5}, False),
            (5, {"type": "integer", "maximum": 5}, True),
            (-5, {"type": "integer", "maximum": -5}, True),
            (6, {"type": "integer", "maximum": 5}, False),
            (-4, {"type": "integer", "maximum": -5}, False),
            (4, {"type": "integer", "exclusiveMaximum": 5}, True),
            (-6, {"type": "integer", "exclusiveMaximum": -5}, True),
            (5, {"type": "integer", "exclusiveMaximum": 5}, False),
            (-5, {"type": "integer", "exclusiveMaximum": -5}, False),
            (5, {"type": "integer", "minimum": 5, "maximum": 10}, True),
            (-10, {"type": "integer", "minimum": -10, "maximum": -5}, True),
            (4, {"type": "integer", "minimum": 5, "maximum": 10}, False),
            (-11, {"type": "integer", "minimum": -10, "maximum": -5}, False),
            (10, {"type": "integer", "exclusiveMinimum": 5, "exclusiveMaximum": 10}, False),
            (-5, {"type": "integer", "exclusiveMinimum": -10, "exclusiveMaximum": -5}, False),
            (5, {"type": "integer", "exclusiveMinimum": 5, "exclusiveMaximum": 10}, False),
            (-10, {"type": "integer", "exclusiveMinimum": -10, "exclusiveMaximum": -5}, False),
            (7, {"type": "integer", "exclusiveMinimum": 5, "exclusiveMaximum": 10}, True),
            (-7, {"type": "integer", "exclusiveMinimum": -10, "exclusiveMaximum": -5}, True),
            # --- Number type tests ---
            (5, {"type": "number", "minimum": 5.0}, True),
            (-5, {"type": "number", "minimum": -5.0}, True),
            (5.0, {"type": "number", "minimum": 5.0}, True),
            (-5.0, {"type": "number", "minimum": -5.0}, True),
            (4.9, {"type": "number", "minimum": 5.0}, False),
            (-5.1, {"type": "number", "minimum": -5.0}, False),
            (5.1, {"type": "number", "exclusiveMinimum": 5.0}, True),
            (-4.9, {"type": "number", "exclusiveMinimum": -5.0}, True),
            (5, {"type": "number", "exclusiveMinimum": 5.0}, False),
            (-5, {"type": "number", "exclusiveMinimum": -5.0}, False),
            (5.0, {"type": "number", "exclusiveMinimum": 5.0}, False),
            (-5.0, {"type": "number", "exclusiveMinimum": -5.0}, False),
            (5, {"type": "number", "maximum": 5.0}, True),
            (-5, {"type": "number", "maximum": -5.0}, True),
            (5.0, {"type": "number", "maximum": 5.0}, True),
            (-5.0, {"type": "number", "maximum": -5.0}, True),
            (5.1, {"type": "number", "maximum": 5.0}, False),
            (-4.9, {"type": "number", "maximum": -5.0}, False),
            (4.9, {"type": "number", "exclusiveMaximum": 5.0}, True),
            (-5.1, {"type": "number", "exclusiveMaximum": -5.0}, True),
            (5, {"type": "number", "exclusiveMaximum": 5.0}, False),
            (-5, {"type": "number", "exclusiveMaximum": -5.0}, False),
            (5.0, {"type": "number", "exclusiveMaximum": 5.0}, False),
            (-5.0, {"type": "number", "exclusiveMaximum": -5.0}, False),
            (7.5, {"type": "number", "minimum": 5.0, "maximum": 10.0}, True),
            (-7.5, {"type": "number", "minimum": -10.0, "maximum": -5.0}, True),
            (4.9, {"type": "number", "minimum": 5.0, "maximum": 10.0}, False),
            (-10.1, {"type": "number", "minimum": -10.0, "maximum": -5.0}, False),
            (5.1, {"type": "number", "exclusiveMinimum": 5.0, "exclusiveMaximum": 10.0}, True),
            (-9.9, {"type": "number", "exclusiveMinimum": -10.0, "exclusiveMaximum": -5.0}, True),
            (5.0, {"type": "number", "exclusiveMinimum": 5.0, "exclusiveMaximum": 10.0}, False),
            (
                -10.0,
                {"type": "number", "exclusiveMinimum": -10.0, "exclusiveMaximum": -5.0},
                False,
            ),
            (9.9, {"type": "number", "exclusiveMinimum": 5.0, "exclusiveMaximum": 10.0}, True),
            (-5.1, {"type": "number", "exclusiveMinimum": -10.0, "exclusiveMaximum": -5.0}, True),
            # --- Edge cases ---
            (0, {"type": "integer", "minimum": 0}, True),
            (0, {"type": "number", "maximum": 0}, True),
            (-1, {"type": "integer", "minimum": 0}, False),
            (1, {"type": "integer", "minimum": 0}, True),
            (-5, {"type": "number", "maximum": 0}, True),
            (5, {"type": "number", "maximum": 0}, False),
            (5, {"type": "integer", "minimum": 5.5}, False),
            (-6, {"type": "integer", "minimum": -5.5}, False),
            (6, {"type": "integer", "minimum": 5.5}, True),
            (-5, {"type": "integer", "minimum": -5.5}, True),
            (5, {"type": "integer", "exclusiveMinimum": 5.5}, False),
            (-6, {"type": "integer", "exclusiveMinimum": -5.5}, False),
            (6, {"type": "integer", "exclusiveMinimum": 5.5}, True),
            (-5, {"type": "integer", "exclusiveMinimum": -5.5}, True),
            (5, {"type": "integer", "maximum": 5.5}, True),
            (-6, {"type": "integer", "maximum": -5.5}, True),
            (6, {"type": "integer", "maximum": 5.5}, False),
            (-5, {"type": "integer", "maximum": -5.5}, False),
            (5, {"type": "integer", "exclusiveMaximum": 5.5}, True),
            (-6, {"type": "integer", "exclusiveMaximum": -5.5}, True),
            (6, {"type": "integer", "exclusiveMaximum": 5.5}, False),
            (-5, {"type": "integer", "exclusiveMaximum": -5.5}, False),
            # --- Large numbers ---
            (1e10, {"type": "number", "minimum": 1e10}, True),
            (-1e10, {"type": "number", "minimum": -1e10}, True),
            (1e9, {"type": "number", "minimum": 1e10}, False),
            (-1e11, {"type": "number", "minimum": -1e10}, False),
            (1e10, {"type": "number", "maximum": 1e10}, True),
            (-1e10, {"type": "number", "maximum": -1e10}, True),
            (1e11, {"type": "number", "maximum": 1e10}, False),
            (-1e9, {"type": "number", "maximum": -1e10}, False),
            # --- Decimal precision ---
            (0.1001, {"type": "number", "minimum": 0.1, "maximum": 0.3}, True),
            (-0.1001, {"type": "number", "minimum": -0.3, "maximum": -0.1}, True),
            (0.2999, {"type": "number", "minimum": 0.1, "maximum": 0.3}, True),
            (-0.2999, {"type": "number", "minimum": -0.3, "maximum": -0.1}, True),
            (0.0999, {"type": "number", "minimum": 0.1, "maximum": 0.3}, False),
            (-0.0999, {"type": "number", "minimum": -0.3, "maximum": -0.1}, False),
            (0.3001, {"type": "number", "minimum": 0.1, "maximum": 0.3}, False),
            (-0.3001, {"type": "number", "minimum": -0.3, "maximum": -0.1}, False),
        ],
    )
    def test_numeric_validation(self, instance, schema, should_pass):
        # Sanity check
        if should_pass:
            validate(instance, schema=schema)
            generate_and_check(instance, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance, schema=schema)
            check_match_failure(bad_string=json_dumps(instance), schema_obj=schema)


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
        schema = """{ "type": "string", "pattern": "^a[A-Z]$"}"""
        schema_obj = json.loads(schema)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
        )

    @pytest.mark.parametrize("string", ["aA\u001f", '"""'])
    def test_regex_properly_escaped_good(self, string):
        schema_obj = {"type": "string", "pattern": r"^.{3}$"}
        # First sanity check what we're setting up
        validate(instance=string, schema=schema_obj)
        # The actual check
        generate_and_check(string, schema_obj)

    @pytest.mark.parametrize(
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes"],
        [
            (
                '"\\u001f\\u001f\u001f',
                b'"\\u001f\\u001f',  # able to match the first two stringified bytes
                "\u001f".encode(),  # fails on a literal \x1f byte
                None,  # hard to write a set of allowed bytes here
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
        )

    @pytest.mark.parametrize("my_string", ["a", "bb", "ccc", "150", ",?", ".\t\n", "(){", "aA7", "\\9O"])
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
        )

    def test_unsatisfiable_length(self):
        schema = {"type": "string", "minLength": 10, "maxLength": 5}
        with pytest.raises(ValueError) as ve:
            _ = gen_json(schema=schema)
        assert ve.value.args[0] == "Unsatisfiable schema: minLength (10) is greater than maxLength (5)"

    @pytest.mark.parametrize("length", range(2, 7))
    @pytest.mark.parametrize("character", ["a", "b"])
    def test_pattern_length_intersection(self, length, character):
        schema = {"type": "string", "minLength": 3, "maxLength": 5, "pattern": "^a+$"}
        string = character * length
        if length < 3 or length > 5 or character != "a":
            with pytest.raises(ValidationError):
                # Sanity check
                validate(instance=string, schema=schema)
            check_match_failure(bad_string=json_dumps(string), schema_obj=schema)
        else:
            # Sanity check
            validate(instance=string, schema=schema)
            generate_and_check(string, schema)


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
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes"],
        [
            ("9999a7777", b"", b"9", {b"{"}),
            ('{"a": 1255.4567}', b'{"a": 1255', b".", {b"}", *INTEGER_FOLLOWING}),
            ('{"a": "123"}', b'{"a": ', b'"', INTEGER_LEADING),
        ],
    )
    def test_bad_object(self, bad_string, good_bytes, failure_byte, allowed_bytes):
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
        )

    def test_unsatisfiable_properties_ok(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": False},
            "additionalProperties": False,
        }
        generate_and_check({"a": 42}, schema)
        check_match_failure(
            bad_string=json_dumps({"a": 42, "b": 43}),
            good_bytes=b'{"a": 42',
            failure_byte=b",",
            allowed_bytes={b"}"} | INTEGER_FOLLOWING,
            schema_obj=schema,
        )

    def test_unsatisfiable_properties_raises(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": False},
            "required": ["b"],
            "additionalProperties": False,
        }
        with pytest.raises(ValueError) as ve:
            _ = gen_json(schema=schema)
        assert ve.value.args[0] == "Unsatisfiable schema: required property 'b' is unsatisfiable"
        # TODO: deeper traceback, e.g.
        # assert (
        #     ve.value.__cause__.args[0] == "Unsatisfiable schema: schema is false"
        # )

    def test_unsatisfiable_additional_properties_raises(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        }
        with pytest.raises(ValueError) as ve:
            _ = gen_json(schema=schema)
        assert ve.value.args[0].startswith("Unsatisfiable schema")
        # TODO: more informative error message, e.g.
        # "Required properties not in properties but additionalProperties is unsatisfiable. Missing required properties: ['b']"
        # TODO: deeper traceback, e.g.
        # assert (
        #     ve.value.__cause__.args[0] == "Unsatisfiable schema: schema is false"
        # )


class TestObjectWithMissingRequired:
    def test_required_is_required(self):
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["b"]}
        generate_and_check({"b": 1}, schema)
        generate_and_check({"a": 1, "b": "xyz"}, schema)
        check_match_failure(
            bad_string=json_dumps({"a": 1}),
            schema_obj=schema,
        )

    def test_validated_against_additionalProperties(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["b"],
            "additionalProperties": {"type": "integer"},
        }
        generate_and_check({"b": 1}, schema)
        generate_and_check({"a": 1, "b": 42}, schema)
        check_match_failure(
            bad_string=json_dumps({"a": 1, "b": "string"}),
            schema_obj=schema,
        )

    def test_false_additionalProperties_fails(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["b", "c"],
            "additionalProperties": False,
        }
        with pytest.raises(ValueError) as ve:
            _ = gen_json(schema=schema)
        assert ve.value.args[0].startswith("Unsatisfiable schema")
        # TODO: more informative error message, e.g.
        # "Required properties not in properties but additionalProperties is unsatisfiable. Missing required properties: ['b', 'c']"
        # TODO: deeper traceback, e.g.
        # assert (
        #     ve.value.__cause__.args[0] == "Unsatisfiable schema: schema is false"
        # )


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
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes"],
        [
            ("9999a7777", b"", b"9", {b"["}),
            ("[321.654]", b"[321", b".", {b"]", b",", *INTEGER_FOLLOWING}),
            ('["123"]', b"[", b'"', {b"]", *INTEGER_LEADING}),
        ],
    )
    def test_bad_object(self, bad_string, good_bytes, failure_byte, allowed_bytes):
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
        )

    def test_unsatisfiable_prefixItem_ok(self):
        schema = {"type": "array", "prefixItems": [{"type": "integer"}, False]}
        generate_and_check([42], schema)
        check_match_failure(
            bad_string="[42, 43]",
            good_bytes=b"[42",
            failure_byte=b",",
            allowed_bytes={b"]"} | INTEGER_FOLLOWING,
            schema_obj=schema,
        )

    def test_unsatisfiable_prefixItem_raises(self):
        schema = {
            "type": "array",
            "prefixItems": [{"type": "integer"}, False],
            "minItems": 2,
        }
        with pytest.raises(ValueError) as ve:
            _ = gen_json(schema=schema)
        assert ve.value.args[0] == "Unsatisfiable schema: prefixItems[1] is unsatisfiable but minItems is 2"
        # TODO: deeper traceback, e.g.
        # assert ve.value.args[0].__cause__.args[0] == "Unsatisfiable schema: schema is false"

    def test_unsatisfiable_items_ok(self):
        schema = {
            "type": "array",
            "prefixItems": [{"type": "integer"}],
            "items": {"allOf": [{"type": "integer"}, False]},
        }
        generate_and_check([42], schema)
        check_match_failure(
            bad_string="[42, 43]",
            good_bytes=b"[42",
            failure_byte=b",",
            allowed_bytes={b"]"} | INTEGER_FOLLOWING,
            schema_obj=schema,
        )

    def test_unsatisfiable_items_raises(self):
        schema = {
            "type": "array",
            "prefixItems": [{"type": "integer"}],
            "items": {"allOf": [{"type": "integer"}, False]},
            "minItems": 2,
        }
        with pytest.raises(ValueError) as ve:
            _ = gen_json(schema=schema)
        assert ve.value.args[0].startswith("Unsatisfiable schema: required item is unsatisfiable")
        # TODO: more detailed error message, e.g.
        # "prefixItems has too few elements (1) to satisfy minItems (2) but item schema is unsatisfiable"


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
        "min_items, max_items, bad_obj, good_bytes, failure_byte, allowed_bytes",
        [
            (
                1,
                4,
                [42, "string_not_bool", "hello", "extra"],
                b"[42, ",
                b'"',
                {b"t", b"f"},
            ),  # Second item does not match prefix schema
            (
                0,
                3,
                [42, True, 100],
                b"[42, true, ",
                b"1",
                {b'"'},
            ),  # Last item does not match general item schema
            (
                3,
                5,
                [42, True, "valid", "extra1", "extra2", "too_many"],
                b'[42, true, "valid", "extra1", "extra2"',
                b",",
                {b"]"},
            ),  # Exceeds maxItems
            (
                2,
                3,
                [42],
                b"[42",
                b"]",
                {b",", *INTEGER_FOLLOWING},
            ),  # Not enough items
            (
                1,
                1,
                [42, True],
                b"[42",
                b",",
                {b"]", *INTEGER_FOLLOWING},
            ),  # Too many items for maxItems
            (
                0,
                0,
                [42, True, "str"],
                b"[",
                b"4",
                {b"]"},
            ),  # maxItems set to 0, but array is not empty
            (
                3,
                5,
                [42, True],
                b"[42, true",
                b"]",
                {b","},
            ),  # Array has one fewer item than required by minItems
        ],
    )
    def test_bad_with_prefix_and_items(self, min_items, max_items, bad_obj, good_bytes, failure_byte, allowed_bytes):
        schema_obj = {
            "prefixItems": self.prefix_schema_obj,
            "items": self.items_schema_obj,
            "minItems": min_items,
            "maxItems": max_items,
            "type": "array",
        }
        bad_string = json_dumps(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
        )

    @pytest.mark.parametrize(
        "min_items, max_items, bad_obj, good_bytes, failure_byte, allowed_bytes",
        [
            (
                2,
                2,
                [42],
                b"[42",
                b"]",
                {b",", *INTEGER_FOLLOWING},
            ),  # Array too short to meet minItems, despite matching prefixItems
            (
                1,
                2,
                [42, "not_bool"],
                b"[42, ",
                b'"',
                {b"t", b"f"},
            ),  # Second item violates prefixItems type requirement
            (
                0,
                1,
                [42, True],
                b"[42",
                b",",
                {b"]", *INTEGER_FOLLOWING},
            ),  # Array exceeds maxItems with valid prefixItems types
            (
                1,
                5,
                [42, True, "extra"],
                b"[42, true",
                b",",
                {b"]"},
            ),  # Item beyond prefixItems with no "items" schema
            (
                0,
                0,
                [42],
                b"[",
                b"4",
                {b"]"},
            ),  # maxItems set to 0, but array is not empty
        ],
    )
    def test_bad_with_prefix(self, min_items, max_items, bad_obj, good_bytes, failure_byte, allowed_bytes):
        schema_obj = {
            "prefixItems": self.prefix_schema_obj,
            "items": False,
            "minItems": min_items,
            "maxItems": max_items,
            "type": "array",
        }
        bad_string = json_dumps(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
        )

    @pytest.mark.parametrize(
        "min_items, max_items, bad_obj, good_bytes, failure_byte, allowed_bytes",
        [
            (
                1,
                2,
                ["hello", "world", "extra"],
                b'["hello", "world"',
                b",",
                {b"]"},
            ),  # Too many items for maxItems
            (
                2,
                3,
                ["hello"],
                b'["hello"',
                b"]",
                {b","},
            ),  # Not enough items
            (
                2,
                3,
                ["hello", 42],
                b'["hello", ',
                b"4",
                {b'"'},
            ),  # Badly typed second item
            (
                0,
                0,
                ["hello"],
                b"[",
                b'"',
                {b"]"},
            ),  # maxItems set to 0, but array is not empty
        ],
    )
    def test_bad_with_items(self, min_items, max_items, bad_obj, good_bytes, failure_byte, allowed_bytes):
        schema_obj = {
            "items": self.items_schema_obj,
            "minItems": min_items,
            "maxItems": max_items,
            "type": "array",
        }
        bad_string = json_dumps(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
        )

    def test_unsatisfiable_length(self):
        schema = {"type": "array", "minItems": 10, "maxItems": 5}
        with pytest.raises(ValueError) as ve:
            _ = gen_json(schema=schema)
        assert ve.value.args[0] == "Unsatisfiable schema: minItems (10) is greater than maxItems (5)"


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

    def test_anyOf_unsatisfiable_ok(self):
        schema = {"anyOf": [{"type": "integer"}, False]}
        generate_and_check(3, schema)

    def test_anyOf_unsatisfiable_raises(self):
        schema = {
            "anyOf": [{"type": "integer", "minimum": 10, "maximum": 0}, False],
        }
        with pytest.raises(ValueError) as ve:
            _ = gen_json(schema=schema)
        assert ve.value.args[0] == "Unsatisfiable schema: minimum (10) is greater than maximum (0)"


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
        schema = {"allOf": [{"type": "integer"}, {"type": "string"}]}
        with pytest.raises(ValueError) as ve:
            _ = gen_json(schema=schema)
        assert ve.value.args[0].startswith("Unsatisfiable schema")
        # TODO: would be nice to have a more specific error message here, e.g.
        # f"Unsatisfiable schema: allOf has conflicting types: [{'integer'}, {'string'}]"


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

    @pytest.mark.parametrize(
        "schema, instances",
        [
            # Simple case, disjoint types
            ({"oneOf": [{"type": "integer"}, {"type": "boolean"}]}, [123, True]),
            # Simple case, disjoint enums
            ({"oneOf": [{"enum": ["a", "b", "c"]}, {"enum": [1, 2, 3]}]}, ["a", "b", "c", 1, 2, 3]),
            # More complex case, discriminated union
            (
                {
                    "oneOf": [
                        # Only one of them needs the prop key to be required
                        {"type": "object", "properties": {"prop": {"const": "foo"}}, "required": ["prop"]},
                        {"type": "object", "properties": {"prop": {"const": "bar"}}},
                    ]
                },
                [{"prop": "foo"}, {"prop": "bar"}],
            ),
            # Enums made disjoint by type
            ({"oneOf": [{"enum": [1, 2, "foo"]}, {"enum": [2, 3, "bar"]}], "type": "string"}, ["foo", "bar"]),
        ],
    )
    def test_oneOf_disjoint(self, schema, instances):
        for instance in instances:
            # First sanity check what we're setting up
            validate(instance=instance, schema=schema)

            # The actual check; we assert NO warning here because oneOf is disjoint
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                generate_and_check(instance, schema)

    @pytest.mark.parametrize(
        "schema, instances",
        [
            # Overlapping enums
            ({"oneOf": [{"enum": ["a", "b", "c"]}, {"enum": ["c", 2, 3]}]}, ["a", "b", 2, 3]),
            # More complex case, object without proper discriminator
            (
                {
                    "oneOf": [
                        # Only one of them needs the prop key to be required
                        {"type": "object", "properties": {"prop": {"const": "foo"}}},
                        {"type": "object", "properties": {"prop": {"const": "bar"}}},
                    ]
                },
                [{"prop": "foo"}, {"prop": "bar"}],
            ),
        ],
    )
    def test_oneOf_overlap(self, schema, instances):
        for instance in instances:
            # First sanity check what we're setting up
            validate(instance=instance, schema=schema)

            # The actual check; assert a warning here because oneOf is not disjoint and we can't guarantee correctness
            with pytest.warns() as record:
                generate_and_check(instance, schema)
            assert len(record) == 1
            assert (
                record[0].message.args[0]
                == "oneOf not fully supported, falling back to anyOf. This may cause validation errors in some cases."
            )


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
        "bad_obj, good_bytes, failure_byte, allowed_bytes",
        [
            ("1", b'"', b"1", {b"2"}),
            (2, b"", b"2", {b'"', b"1", b"f"}),
            (True, b"", b"t", {b'"', b"1", b"f"}),
        ],
    )
    def test_bad_enum(self, bad_obj, good_bytes, failure_byte, allowed_bytes):
        schema_obj = json.loads(self.simple_schema)
        bad_string = json_dumps(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
        )

    @pytest.mark.parametrize(
        "bad_obj, good_bytes, failure_byte, allowed_bytes",
        [
            ("ab", b'"a', b"b", {b"a"}),
            ("bc", b'"b', b"c", {b"b"}),
            ("ca", b'"c', b"a", {b"c"}),
        ],
    )
    def test_bad_prefix_enum(self, bad_obj, good_bytes, failure_byte, allowed_bytes):
        schema_obj = json.loads(self.prefix_schema)
        bad_string = json_dumps(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
        )

    @pytest.mark.parametrize(
        "obj, valid",
        [
            (1, True),
            (2, False),
            ("2", False),
            ("1", False),
            (True, False),
        ],
    )
    def test_typed_enum_single_type(self, obj, valid):
        schema_obj = {"enum": [1, "2", True], "type": "integer"}
        if valid:
            validate(instance=obj, schema=schema_obj)
            generate_and_check(obj, schema_obj)
        else:
            with pytest.raises(ValidationError):
                validate(instance=obj, schema=schema_obj)
            check_match_failure(bad_string=json_dumps(obj), schema_obj=schema_obj)

    @pytest.mark.parametrize(
        "obj, valid",
        [
            (1, True),
            (2, False),
            ("2", True),
            ("1", False),
            (True, False),
        ],
    )
    def test_typed_enum_multiple_types(self, obj, valid):
        schema_obj = {"enum": [1, "2", True], "type": ["integer", "string"]}
        if valid:
            validate(instance=obj, schema=schema_obj)
            generate_and_check(obj, schema_obj)
        else:
            with pytest.raises(ValidationError):
                validate(instance=obj, schema=schema_obj)
            check_match_failure(bad_string=json_dumps(obj), schema_obj=schema_obj)

    def test_invalid_typed_enum(self):
        schema_obj = {"enum": [1, "2"], "type": "boolean"}
        with pytest.raises(ValueError) as ve:
            gen_json(schema=schema_obj)
        assert ve.value.args[0].startswith("Unsatisfiable schema")
        # TODO: would be nice to have a more specific error message here, e.g.
        # f"Unsatisfiable schema: all enum options {[1, '2']} are inconsistent with parent schema: {schema_obj}"


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
        bad_string = json_dumps(2)

        check_match_failure(
            bad_string=bad_string,
            good_bytes=b"",
            failure_byte=b"2",
            allowed_bytes={b"1"},
            schema_obj=schema_obj,
        )

    def test_valid_typed_const(self):
        schema_obj = {"const": 1, "type": "integer"}
        target_obj = 1
        validate(instance=target_obj, schema=schema_obj)
        generate_and_check(target_obj, schema_obj)

    def test_invalid_typed_const(self):
        schema_obj = {"const": 1, "type": "boolean"}
        with pytest.raises(ValueError) as ve:
            gen_json(schema=schema_obj)
        assert ve.value.args[0].startswith("Unsatisfiable schema")
        # TODO: would be nice to have a more specific error message here, e.g.
        # f"Unsatisfiable schema: const {1!r} is inconsistent with parent schema: {schema_obj}"

    def test_valid_enum_const(self):
        schema_obj = {"const": 1, "enum": [1, 2, 3]}
        target_obj = 1
        validate(instance=target_obj, schema=schema_obj)
        generate_and_check(target_obj, schema_obj)

    def test_invalid_enum_const(self):
        schema_obj = {"const": 1, "enum": [2, 3]}
        with pytest.raises(ValueError) as ve:
            gen_json(schema=schema_obj)
        assert ve.value.args[0].startswith("Unsatisfiable schema")
        # TODO: would be nice to have a more specific error message here, e.g.

    def test_valid_typed_enum_const(self):
        schema_obj = {"const": 1, "enum": [1, "2", 3], "type": "integer"}
        target_obj = 1
        validate(instance=target_obj, schema=schema_obj)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "const",
        [
            "2",  # right enum, wrong type
            2,  # wrong enum, right type
            "3",  # wrong enum, wrong type
        ],
    )
    def test_invalid_typed_enum_const(self, const):
        schema_obj = {"const": const, "enum": [1, "2", 3], "type": "integer"}
        with pytest.raises(ValueError) as ve:
            gen_json(schema=schema_obj)
        assert ve.value.args[0].startswith("Unsatisfiable schema")
        # TODO: would be nice to have a more specific error message here, e.g.
        # f"Unsatisfiable schema: const {const!r} is inconsistent with parent schema: {schema_obj}"


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
        "bad_obj, good_bytes, failure_byte, allowed_bytes",
        [
            (
                {"a": "1"},
                b'{"a": ',
                b'"',
                INTEGER_LEADING,
            ),
            (
                {"a": 1, "b": 1.5},
                b'{"a": 1, "b": 1',
                b".",
                {b",", b"}", *INTEGER_FOLLOWING},
            ),
        ],
    )
    def test_simple_bad_type(self, bad_obj, good_bytes, failure_byte, allowed_bytes):
        schema_obj = json.loads(self.simple_schema)
        bad_string = json_dumps(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
        )

    @pytest.mark.parametrize("target_obj", [{}, {"a": 1}, {"a": "2"}, {"a": 1, "b": "2"}])
    def test_anyOf_additional_properties(self, target_obj):
        # First sanity check what we're setting up
        schema_obj = json.loads(self.anyOf_schema)
        validate(instance=target_obj, schema=schema_obj)

        # The actual check
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_obj, good_bytes, failure_byte, allowed_bytes",
        [
            ({"a": 1.5}, b'{"a": 1', b".", {b",", b"}", *INTEGER_FOLLOWING}),
            ({"a": True}, b'{"a": ', b"t", {b'"', *INTEGER_LEADING}),
            (
                {"a": 1, "b": False},
                b'{"a": 1, "b": ',
                b"f",
                {b'"', *INTEGER_LEADING},
            ),
        ],
    )
    def test_anyOf_bad_type(self, bad_obj, good_bytes, failure_byte, allowed_bytes):
        schema_obj = json.loads(self.anyOf_schema)
        bad_string = json_dumps(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
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
        "bad_obj, good_bytes, failure_byte, allowed_bytes",
        [
            ({}, b"{", b"}", {b'"'}),
            ({"a": 1}, b'{"', b"a", {b"m"}),
            ({"a": 1, "b": 2}, b'{"', b"a", {b"m"}),
        ],
    )
    def test_combined_missing_properties(self, bad_obj, good_bytes, failure_byte, allowed_bytes):
        schema_obj = json.loads(self.combined_schema)
        bad_string = json_dumps(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
        )

    @pytest.mark.parametrize(
        "bad_obj, good_bytes, failure_byte, allowed_bytes",
        [
            ({"mystr": 1}, b'{"mystr": ', b"1", {b'"'}),
            ({"mystr": 1, "a": 2}, b'{"mystr": ', b"1", {b'"'}),
            (
                {"mystr": "hello", "a": False},
                b'{"mystr": "hello", "a": ',
                b"f",
                INTEGER_LEADING,
            ),
        ],
    )
    def test_combined_bad_type(self, bad_obj, good_bytes, failure_byte, allowed_bytes):
        schema_obj = json.loads(self.combined_schema)
        bad_string = json_dumps(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
        )

    def test_out_of_order_non_required_properties_not_validated_as_additionalProperties(self):
        schema = {
            "type": "object",
            "properties": {"a": {"const": "foo"}, "b": {"const": "bar"}},
            "required": ["b"],
        }
        test_string = '{"b": "bar", "a": "BAD"}'
        grammar = gen_json(schema=schema)
        assert grammar.match(test_string) is None


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
        "bad_string, good_bytes, failure_byte, allowed_bytes",
        [
            # {} is not carte blanche for malformed JSON
            ("{a:1}", b"{", b"a", {b'"', b"}"}),
            (
                "[1, 2} ",
                b"[1, 2",
                b"}",
                {b",", b"]", b"e", b"E", b".", *INTEGER_FOLLOWING},
            ),
            ("123a", b"123", b"a", {b"e", b"E", b".", *INTEGER_FOLLOWING}),
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
            ),
        ],
    )
    def test_bad_empty_schema(self, bad_string, good_bytes, failure_byte, allowed_bytes):
        schema_obj = json.loads(self.empty_schema)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
        )

    @pytest.mark.parametrize(
        "schema_obj",
        [
            # Empty property
            {"type": "object", "properties": {"a": {}}, "required": ["a"]},
            # Empty reference
            {
                "type": "object",
                "properties": {"a": {"$ref": "#/$defs/A"}},
                "$defs": {"A": {}},
                "required": ["a"],
            },
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
            {
                "type": "object",
                "properties": {"a": {"$ref": "#/$defs/A"}},
                "$defs": {"A": {}},
                "required": ["a"],
            },
        ],
    )
    @pytest.mark.parametrize(
        "bad_obj, good_bytes, failure_byte, allowed_bytes",
        [
            # Missing property -- presence of {} deeper in the schema isn't carte blanche
            ({"b": 42}, b'{"', b"b", {b"a"}),
        ],
    )
    def test_nested_empty_schema_bad(self, schema_obj, bad_obj, good_bytes, failure_byte, allowed_bytes):
        bad_string = json_dumps(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
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
        "bad_obj, good_bytes, failure_byte, allowed_bytes",
        [
            # Missing property -- presence of {} deeper in the schema isn't carte blanche
            ({"b": 42}, b'{"', b"b", {b"a"}),
        ],
    )
    def test_nested_empty_schema_with_props_bad(self, bad_obj, good_bytes, failure_byte, allowed_bytes):
        schema_obj = json.loads(self.nested_empty_schema_with_props)

        bad_string = json_dumps(bad_obj)
        check_match_failure(
            bad_string=bad_string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            schema_obj=schema_obj,
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
        generate_and_check([1, 0.4, "hello", False, None, {"a": 42}, [1, 2, 3, "four"]], schema_obj)

    def test_no_items(self):
        schema_obj = {"type": "array", "items": False}
        check_match_failure(
            bad_string="[42]",
            good_bytes=b"[",
            failure_byte=b"4",
            allowed_bytes={b"]"},  # array must be empty
            schema_obj=schema_obj,
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

    def test_no_additionalProperties(self):
        schema_obj = {"type": "object", "additionalProperties": False}
        check_match_failure(
            bad_string='{"a": 42}',
            good_bytes=b"{",
            failure_byte=b'"',
            allowed_bytes={b"}"},  # object must be empty
            schema_obj=schema_obj,
        )


class TestRequiredProperties:
    schema_obj = {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "number"},
            "c": {"type": "boolean"},
        },
        "additionalProperties": True,
    }
    ALL_REQUIRED = ["a", "b", "c"]
    SOME_REQUIRED_SUBSETS = [
        [],
        ["a"],
        ["b"],
        ["c"],
        ["a", "b"],
        ["a", "c"],
        ["b", "c"],
        ["a", "b", "c"],
    ]
    NONE_REQUIRED: list[str] = []

    @pytest.mark.parametrize(
        "extra_items",
        [
            {},
            {"d": "hello"},
            {"d": 42, "e": True},
        ],
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
        ],
    )
    def test_all_required_bad(self, bad_obj):
        schema_obj = {**self.schema_obj, "required": self.ALL_REQUIRED}
        check_match_failure(
            bad_string=json_dumps(bad_obj),
            schema_obj=schema_obj,
        )

    @pytest.mark.parametrize(
        "extra_items",
        [
            {},
            {"d": "hello"},
            {"d": 42, "e": True},
        ],
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
                    bad_string=json_dumps(bad_obj),
                    schema_obj=schema_obj,
                )

    # No equivalent "bad" tests for none required, as the schema is satisfied by an empty object
    @pytest.mark.parametrize(
        "extra_items",
        [
            {},
            {"d": "hello"},
            {"d": 42, "e": True},
        ],
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
        ],
    )
    def test_none_required(self, target_obj, extra_items):
        schema_obj = {**self.schema_obj, "required": self.NONE_REQUIRED}
        generate_and_check({**target_obj, **extra_items}, schema_obj)


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
            True,
        ],
    )
    def test_true_schema(self, target_obj):
        # should be the same as an empty schema
        schema_obj = True
        generate_and_check(target_obj, schema_obj)

    def test_false_schema(self):
        schema_obj = False
        with pytest.raises(ValueError) as ve:
            gen_json(schema=schema_obj)
        assert ve.value.args[0] == "Unsatisfiable schema: schema is false"

    def test_false_required_property(self):
        schema_obj = {"type": "object", "properties": {"a": False}, "required": ["a"]}
        with pytest.raises(ValueError) as ve:
            gen_json(schema=schema_obj)
        assert ve.value.args[0] == "Unsatisfiable schema: required property 'a' is unsatisfiable"


class TestWhitespace:
    seps = [
        (", ", ": "),
        (",", ":"),
        (",", ": "),
        (", ", ":"),
    ]

    @pytest.mark.parametrize(
        "schema, obj",
        [
            # Dynamic object (both item and key seps)
            ({"type": "object"}, {"a": 1, "b": 2, "c": [1, 2, 3]}),
            # Static object: enum (both item and key seps)
            ({"enum": [{"a": 1, "b": 2, "c": [1, 2, 3]}]}, {"a": 1, "b": 2, "c": [1, 2, 3]}),
            # Static object: const (both item and key seps)
            ({"const": {"a": 1, "b": 2, "c": [1, 2, 3]}}, {"a": 1, "b": 2, "c": [1, 2, 3]}),
        ],
    )
    @pytest.mark.parametrize(
        "separators",
        seps,
    )
    def test_separators(self, separators, schema, obj):
        grammar = gen_json(schema=schema, separators=separators)
        for seps in self.seps:
            prepared_json = json.dumps(obj, separators=seps)
            if separators == seps:
                assert grammar.match(prepared_json) is not None
                model = models.Mock(f"<s>{prepared_json}".encode())
                assert str(model + grammar) == prepared_json
            else:
                assert grammar.match(prepared_json) is None

    @pytest.mark.parametrize(
        "schema, obj",
        [
            # Dynamic object (both item and key seps)
            ({"type": "object"}, {"a": 1, "b": 2, "c": [1, 2, 3]}),
            # Static object: enum (both item and key seps)
            ({"enum": [{"a": 1, "b": 2, "c": [1, 2, 3]}]}, {"a": 1, "b": 2, "c": [1, 2, 3]}),
            # Static object: const (both item and key seps)
            ({"const": {"a": 1, "b": 2, "c": [1, 2, 3]}}, {"a": 1, "b": 2, "c": [1, 2, 3]}),
        ],
    )
    @pytest.mark.parametrize(
        "separators",
        seps,
    )
    @pytest.mark.parametrize(
        "indent",
        [None, 0, 2, 4],
    )
    def test_whitespace_flexibility(self, indent, separators, schema, obj):
        grammar = gen_json(schema=schema, whitespace_flexible=True)
        prepared_json = json.dumps(obj, separators=separators, indent=indent)

        assert grammar.match(prepared_json, raise_exceptions=True) is not None
        model = models.Mock(f"<s>{prepared_json}".encode())
        assert str(model + grammar) == prepared_json


class TestStringSchema:
    def test_good(self):
        schema = """{"type": "object", "properties": {"a": {"type": "string"}}}"""
        target_obj = {"a": "hello"}
        generate_and_check(target_obj, schema)

    def test_bad(self):
        schema = """{"type": "object", "properties": {"a": {"type": "string"}}}"""
        check_match_failure(
            bad_string='{"a": 42}',
            schema_obj=schema,
        )
