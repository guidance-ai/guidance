from json import dumps as json_dumps

import pytest
from jsonschema import ValidationError, validate

from .test_json import check_match_failure, generate_and_check


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
            check_match_failure(bad_string=json_dumps(test_object), schema_obj=schema)

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
            check_match_failure(bad_string=json_dumps(test_object), schema_obj=schema)

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
            "allOf": [{"multipleOf": 2}],
            "anyOf": [{"multipleOf": 3}],
            "oneOf": [{"multipleOf": 5}],
        }
        if valid:
            validate(instance=test_object, schema=schema)
            generate_and_check(test_object, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance=test_object, schema=schema)
            check_match_failure(bad_string=json_dumps(test_object), schema_obj=schema)
