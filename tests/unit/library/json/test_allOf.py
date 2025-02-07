"""Adapted from https://github.com/json-schema-org/JSON-Schema-Test-Suite/tree/9fc880bfb6d8ccd093bc82431f17d13681ffae8e/tests/draft2020-12/allOf.json"""

from json import dumps as json_dumps

import pytest
from jsonschema import ValidationError, validate

from guidance import json as gen_json

from .utils import check_match_failure, generate_and_check


class TestAllOf:
    @pytest.mark.parametrize(
        ["test_object", "valid"],
        [
            # allOf
            ({"bar": 2, "foo": "baz"}, True),
            # mismatch second
            ({"foo": "baz"}, False),
            # mismatch first
            ({"bar": 2}, False),
            # wrong type
            ({"bar": "quux", "foo": "baz"}, False),
        ],
    )
    def test_allOf(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "allOf": [
                {"properties": {"bar": {"type": "integer"}}, "required": ["bar"]},
                {"properties": {"foo": {"type": "string"}}, "required": ["foo"]},
            ],
        }
        if valid:
            validate(instance=test_object, schema=schema)
            generate_and_check(test_object, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance=test_object, schema=schema)
            check_match_failure(bad_string=json_dumps(test_object), schema_obj=schema)

    @pytest.mark.parametrize(
        ["test_object", "valid"],
        [
            # valid
            ({"bar": 2, "foo": "quux", "baz": None}, True),
            # mismatch base schema
            ({"foo": "quux", "baz": None}, False),
            # mismatch first allOf
            ({"bar": 2, "baz": None}, False),
            # mismatch second allOf
            ({"bar": 2, "foo": "quux"}, False),
            # mismatch both
            ({"bar": 2}, False),
        ],
    )
    def test_allOf_with_base_schema(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "properties": {"bar": {"type": "integer"}},
            "required": ["bar"],
            "allOf": [
                {"properties": {"foo": {"type": "string"}}, "required": ["foo"]},
                {"properties": {"baz": {"type": "null"}}, "required": ["baz"]},
            ],
        }
        if valid:
            validate(instance=test_object, schema=schema)
            generate_and_check(test_object, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance=test_object, schema=schema)
            check_match_failure(bad_string=json_dumps(test_object), schema_obj=schema)

    @pytest.mark.parametrize(
        ["test_object", "valid"],
        [
            # valid
            (25, True),
            # mismatch one
            (35, False),
            # mismatch other
            (15, False),
        ],
    )
    def test_allOf_simple_types(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "allOf": [{"maximum": 30}, {"minimum": 20}],
        }
        if valid:
            validate(instance=test_object, schema=schema)
            generate_and_check(test_object, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance=test_object, schema=schema)
            check_match_failure(bad_string=json_dumps(test_object), schema_obj=schema)

    @pytest.mark.parametrize(
        ["test_object", "valid"],
        [
            # mismatch both
            (15, False),
            # mismatch one
            (25, False),
            # valid
            (35, True),
        ],
    )
    def test_allOf_simple_minimum(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "allOf": [{"minimum": 30}, {"minimum": 20}],
        }
        if valid:
            validate(instance=test_object, schema=schema)
            generate_and_check(test_object, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance=test_object, schema=schema)
            check_match_failure(bad_string=json_dumps(test_object), schema_obj=schema)

    @pytest.mark.parametrize(
        ["test_object", "valid"],
        [
            # mismatch both
            (35, False),
            # mismatch one
            (25, False),
            # valid
            (15, True),
        ],
    )
    def test_allOf_simple_maximum(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "allOf": [{"maximum": 30}, {"maximum": 20}],
        }
        if valid:
            validate(instance=test_object, schema=schema)
            generate_and_check(test_object, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance=test_object, schema=schema)
            check_match_failure(bad_string=json_dumps(test_object), schema_obj=schema)

    @pytest.mark.parametrize(
        ["test_object", "valid"],
        [
            # any value is valid
            ("foo", True)
        ],
    )
    def test_allOf_with_boolean_schemas_all_true(self, test_object, valid):
        schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "allOf": [True, True]}
        if valid:
            validate(instance=test_object, schema=schema)
            generate_and_check(test_object, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance=test_object, schema=schema)
            check_match_failure(bad_string=json_dumps(test_object), schema_obj=schema)

    @pytest.mark.parametrize(
        ["test_object", "valid"],
        [
            # any value is invalid
            ("foo", False)
        ],
    )
    def test_allOf_with_boolean_schemas_some_false(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "allOf": [True, False],
        }
        if valid:
            validate(instance=test_object, schema=schema)
            generate_and_check(test_object, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance=test_object, schema=schema)
            with pytest.raises(ValueError) as ve:
                _ = gen_json(schema=schema)
            assert ve.value.args[0] == "Unsatisfiable schema: schema is false"
            # TODO: more informative error message, e.g. "allOf contains a 'false' schema"

    @pytest.mark.parametrize(
        ["test_object", "valid"],
        [
            # any value is invalid
            ("foo", False)
        ],
    )
    def test_allOf_with_boolean_schemas_all_false(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "allOf": [False, False],
        }
        if valid:
            validate(instance=test_object, schema=schema)
            generate_and_check(test_object, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance=test_object, schema=schema)
            with pytest.raises(ValueError) as ve:
                _ = gen_json(schema=schema)
            assert ve.value.args[0] == "Unsatisfiable schema: schema is false"
            # TODO: more informative error message, e.g. "allOf contains a 'false' schema"

    @pytest.mark.parametrize(
        ["test_object", "valid"],
        [
            # any data is valid
            (1, True)
        ],
    )
    def test_allOf_with_one_empty_schema(self, test_object, valid):
        schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "allOf": [{}]}
        if valid:
            validate(instance=test_object, schema=schema)
            generate_and_check(test_object, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance=test_object, schema=schema)
            check_match_failure(bad_string=json_dumps(test_object), schema_obj=schema)

    @pytest.mark.parametrize(
        ["test_object", "valid"],
        [
            # any data is valid
            (1, True)
        ],
    )
    def test_allOf_with_two_empty_schemas(self, test_object, valid):
        schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "allOf": [{}, {}]}
        if valid:
            validate(instance=test_object, schema=schema)
            generate_and_check(test_object, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance=test_object, schema=schema)
            check_match_failure(bad_string=json_dumps(test_object), schema_obj=schema)

    @pytest.mark.parametrize(
        ["test_object", "valid"],
        [
            # number is valid
            (1, True),
            # string is invalid
            ("foo", False),
        ],
    )
    def test_allOf_with_the_first_empty_schema(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "allOf": [{}, {"type": "number"}],
        }
        if valid:
            validate(instance=test_object, schema=schema)
            generate_and_check(test_object, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance=test_object, schema=schema)
            check_match_failure(bad_string=json_dumps(test_object), schema_obj=schema)

    @pytest.mark.parametrize(
        ["test_object", "valid"],
        [
            # number is valid
            (1, True),
            # string is invalid
            ("foo", False),
        ],
    )
    def test_allOf_with_the_last_empty_schema(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "allOf": [{"type": "number"}, {}],
        }
        if valid:
            validate(instance=test_object, schema=schema)
            generate_and_check(test_object, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance=test_object, schema=schema)
            check_match_failure(bad_string=json_dumps(test_object), schema_obj=schema)

    @pytest.mark.parametrize(
        ["test_object", "valid"],
        [
            # null is valid
            (None, True),
            # anything non-null is invalid
            (123, False),
        ],
    )
    def test_nested_allOf_to_check_validation_semantics(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "allOf": [{"allOf": [{"type": "null"}]}],
        }
        if valid:
            validate(instance=test_object, schema=schema)
            generate_and_check(test_object, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance=test_object, schema=schema)
            check_match_failure(bad_string=json_dumps(test_object), schema_obj=schema)

    @pytest.mark.parametrize(
        ["test_object", "valid"],
        [
            # allOf: false, anyOf: false, oneOf: false
            (1, False),
            # allOf: false, anyOf: false, oneOf: true
            (5, False),
            # allOf: false, anyOf: true, oneOf: false
            (3, False),
            # allOf: false, anyOf: true, oneOf: true
            (15, False),
            # allOf: true, anyOf: false, oneOf: false
            (2, False),
            # allOf: true, anyOf: false, oneOf: true
            (10, False),
            # allOf: true, anyOf: true, oneOf: false
            (6, False),
            # allOf: true, anyOf: true, oneOf: true
            (30, True),
        ],
    )
    def test_allOf_combined_with_anyOf_oneOf(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "allOf": [{"enum": [2, 6, 10, 30]}],
            "anyOf": [{"enum": [3, 6, 15, 30]}],
            "oneOf": [{"enum": [5, 10, 15, 30]}],
        }
        if valid:
            validate(instance=test_object, schema=schema)
            generate_and_check(test_object, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance=test_object, schema=schema)
            check_match_failure(bad_string=json_dumps(test_object), schema_obj=schema)

    @pytest.mark.parametrize(
        "test_object, valid",
        [
            # valid: foo is integer and less than 4, bar is equal to 5, baz is integer greater than 5
            ({"foo": 0, "bar": 5, "baz": 10}, True),
            # valid: foo is null, bar is equal to 5, baz is null
            ({"foo": None, "bar": 5, "baz": None}, True),
            # valid: foo is integer and less than 4, bar is non-number, baz is integer greater than 5
            ({"foo": 0, "bar": "quxx", "baz": 10}, True),
            # invalid: foo is integer and greater than 4
            ({"foo": 5, "bar": 5, "baz": 10}, False),
            # invalid: foo is not an integer or None
            ({"foo": "quxx", "bar": 5, "baz": 10}, False),
            # invalid: bar is greater than 5
            ({"foo": 0, "bar": 6, "baz": 10}, False),
            # invalid: bar is less than 5
            ({"foo": 0, "bar": 4, "baz": 10}, False),
            # invalid: baz is less than 5
            ({"foo": 0, "bar": 5, "baz": 4}, False),
            # invalid: baz is not an integer or null
            ({"foo": 0, "bar": 5, "baz": "quxx"}, False),
        ],
    )
    @pytest.mark.parametrize(
        "schema",
        [
            # The following are equivalent to this:
            {
                "properties": {
                    "foo": {"type": ["integer", "null"], "maximum": 4},
                    "bar": {"minimum": 5, "maximum": 5},
                },
                "additionalProperties": {"type": ["integer", "null"], "minimum": 5},
            },
            # additionalProperties in parent schema
            {
                "allOf": [
                    {"properties": {"bar": {"maximum": 5}}, "additionalProperties": {"type": ["integer", "null"]}}
                ],
                "properties": {"foo": {"maximum": 4}},
                "additionalProperties": {"minimum": 5},
            },
            # additionalProperties in allOf
            {
                "allOf": [
                    {
                        "properties": {"foo": {"maximum": 4}},
                        "additionalProperties": {"minimum": 5},
                    },
                    {
                        "properties": {"bar": {"maximum": 5}},
                        "additionalProperties": {"type": ["integer", "null"]},
                    },
                ]
            },
        ],
    )
    def test_additionalProperties_in_allOf(self, schema, test_object, valid):
        if valid:
            validate(instance=test_object, schema=schema)
            generate_and_check(test_object, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance=test_object, schema=schema)
            check_match_failure(bad_string=json_dumps(test_object), schema_obj=schema)

    @pytest.mark.parametrize(
        "test_object, valid",
        [
            ({}, True),  # empty object is valid
            ({"foo": 1}, False),  # foo is not a string
            ({"foo": "x"}, False),  # foo is not an integer
            ({"foo": True}, False),  # foo is not a string or an integer
        ],
    )
    def test_inconsistent_additionalProperties_in_allOf(self, test_object, valid):
        schema = {
            "type": "object",
            "allOf": [
                {"additionalProperties": {"type": "integer"}},
                {"additionalProperties": {"type": "string"}},
            ],
        }
        if valid:
            validate(instance=test_object, schema=schema)
            generate_and_check(test_object, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance=test_object, schema=schema)
            check_match_failure(bad_string=json_dumps(test_object), schema_obj=schema)

    @pytest.mark.parametrize(
        "test_object, valid",
        [
            # valid: foo is integer and less than 4, bar is equal to 5, baz is integer greater than 5
            ([0, 5, 10], True),
            # valid: foo is null, bar is equal to 5, baz is null
            ([None, 5, None], True),
            # valid: foo is integer and less than 4, bar is non-number, baz is integer greater than 5
            ([0, "quxx", 10], True),
            # invalid: foo is integer and greater than 4
            ([5, 5, 10], False),
            # invalid: foo is not an integer or None
            (["quxx", 5, 10], False),
            # invalid: bar is greater than 5
            ([0, 6, 10], False),
            # invalid: bar is less than 5
            ([0, 4, 10], False),
            # invalid: baz is less than 5
            ([0, 5, 4], False),
            # invalid: baz is not an integer or null
            ([0, 5, "quxx"], False),
        ],
    )
    @pytest.mark.parametrize(
        "schema",
        [
            # The following are equivalent to this:
            {
                "prefixItems": [
                    {"type": ["integer", "null"], "maximum": 4},
                    {"minimum": 5, "maximum": 5},
                ],
                "items": {"type": ["integer", "null"], "minimum": 5},
            },
            # items in parent schema
            {
                "allOf": [
                    {"prefixItems": [{"maximum": 4}], "items": {"minimum": 5}},
                ],
                "prefixItems": [{"type": ["integer", "null"]}, {"maximum": 5}],
                "items": {"type": ["integer", "null"]},
            },
            # items in allOf
            {
                "allOf": [
                    {"prefixItems": [{"maximum": 4}], "items": {"minimum": 5}},
                    {
                        "prefixItems": [{"type": ["integer", "null"]}, {"maximum": 5}],
                        "items": {"type": ["integer", "null"]},
                    },
                ]
            },
        ],
    )
    def test_items_and_prefixitems_in_allOf(self, schema, test_object, valid):
        if valid:
            validate(instance=test_object, schema=schema)
            generate_and_check(test_object, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance=test_object, schema=schema)
            check_match_failure(bad_string=json_dumps(test_object), schema_obj=schema)
