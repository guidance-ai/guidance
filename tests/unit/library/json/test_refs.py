from json import dumps as json_dumps

import pytest
from jsonschema import ValidationError, validate

from .utils import check_match_failure, generate_and_check


class TestRefs:
    @pytest.mark.parametrize(
        ["test_object", "valid"],
        [
            # match
            ({"foo": False}, True),
            # recursive match
            ({"foo": {"foo": False}}, True),
            # mismatch
            ({"bar": False}, False),
            # recursive mismatch
            ({"foo": {"bar": False}}, False),
        ],
    )
    def test_root_pointer_ref(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "properties": {"foo": {"$ref": "#"}},
            "additionalProperties": False,
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
            # match
            ({"bar": 3}, True),
            # mismatch
            ({"bar": True}, False),
        ],
    )
    def test_relative_pointer_ref_to_object(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "properties": {"foo": {"type": "integer"}, "bar": {"$ref": "#/properties/foo"}},
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
            # match array
            ([1, 2], True),
            # mismatch array
            ([1, "foo"], False),
        ],
    )
    def test_relative_pointer_ref_to_array(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "prefixItems": [{"type": "integer"}, {"$ref": "#/prefixItems/0"}],
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
            # slash invalid
            ({"slash": "aoeu"}, False),
            # tilde invalid
            ({"tilde": "aoeu"}, False),
            # percent invalid
            ({"percent": "aoeu"}, False),
            # slash valid
            ({"slash": 123}, True),
            # tilde valid
            ({"tilde": 123}, True),
            # percent valid
            ({"percent": 123}, True),
        ],
    )
    def test_escaped_pointer_ref(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {
                "tilde~field": {"type": "integer"},
                "slash/field": {"type": "integer"},
                "percent%field": {"type": "integer"},
            },
            "properties": {
                "tilde": {"$ref": "#/$defs/tilde~0field"},
                "slash": {"$ref": "#/$defs/slash~1field"},
                "percent": {"$ref": "#/$defs/percent%25field"},
            },
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
            # nested ref valid
            (5, True),
            # nested ref invalid
            ("a", False),
        ],
    )
    def test_nested_refs(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {
                "a": {"type": "integer"},
                "b": {"$ref": "#/$defs/a"},
                "c": {"$ref": "#/$defs/b"},
            },
            "$ref": "#/$defs/c",
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
            # ref valid, maxItems valid
            ({"foo": []}, True),
            # ref valid, maxItems invalid
            ({"foo": [1, 2, 3]}, False),
            # ref invalid
            ({"foo": "string"}, False),
        ],
    )
    def test_ref_applies_alongside_sibling_keywords(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {"reffed": {"type": "array"}},
            "properties": {"foo": {"$ref": "#/$defs/reffed", "maxItems": 2}},
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
            # remote ref valid
            ({"minLength": 1}, True),
            # remote ref invalid
            ({"minLength": -1}, False),
        ],
    )
    @pytest.mark.xfail(reason="Remote refs are not supported")
    def test_remote_ref_containing_refs_itself(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": "https://json-schema.org/draft/2020-12/schema",
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
            # property named $ref valid
            ({"$ref": "a"}, True),
            # property named $ref invalid
            ({"$ref": 2}, False),
        ],
    )
    def test_property_named_ref_that_is_not_a_reference(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "properties": {"$ref": {"type": "string"}},
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
            # property named $ref valid
            ({"$ref": "a"}, True),
            # property named $ref invalid
            ({"$ref": 2}, False),
        ],
    )
    def test_property_named_ref_containing_an_actual_ref(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "properties": {"$ref": {"$ref": "#/$defs/is-string"}},
            "$defs": {"is-string": {"type": "string"}},
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
    def test_ref_to_boolean_schema_true(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": "#/$defs/bool",
            "$defs": {"bool": True},
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
    @pytest.mark.xfail(reason="false schema is not implemented")
    def test_ref_to_boolean_schema_false(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": "#/$defs/bool",
            "$defs": {"bool": False},
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
            # valid tree
            (
                {
                    "meta": "root",
                    "nodes": [
                        {
                            "value": 1,
                            "subtree": {
                                "meta": "child",
                                "nodes": [{"value": 1.1}, {"value": 1.2}],
                            },
                        },
                        {
                            "value": 2,
                            "subtree": {
                                "meta": "child",
                                "nodes": [{"value": 2.1}, {"value": 2.2}],
                            },
                        },
                    ],
                },
                True,
            ),
            # invalid tree
            (
                {
                    "meta": "root",
                    "nodes": [
                        {
                            "value": 1,
                            "subtree": {
                                "meta": "child",
                                "nodes": [{"value": "string is invalid"}, {"value": 1.2}],
                            },
                        },
                        {
                            "value": 2,
                            "subtree": {
                                "meta": "child",
                                "nodes": [{"value": 2.1}, {"value": 2.2}],
                            },
                        },
                    ],
                },
                False,
            ),
        ],
    )
    def test_Recursive_references_between_schemas(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "http://localhost:1234/draft2020-12/tree",
            "description": "tree of nodes",
            "type": "object",
            "properties": {
                "meta": {"type": "string"},
                "nodes": {"type": "array", "items": {"$ref": "node"}},
            },
            "required": ["meta", "nodes"],
            "$defs": {
                "node": {
                    "$id": "http://localhost:1234/draft2020-12/node",
                    "description": "node",
                    "type": "object",
                    "properties": {"value": {"type": "number"}, "subtree": {"$ref": "tree"}},
                    "required": ["value"],
                }
            },
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
            # object with numbers is valid
            ({'foo"bar': 1}, True),
            # object with strings is invalid
            ({'foo"bar': "1"}, False),
        ],
    )
    def test_refs_with_quote(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "properties": {'foo"bar': {"$ref": "#/$defs/foo%22bar"}},
            "$defs": {'foo"bar': {"type": "number"}},
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
            # referenced subschema doesn't see annotations from properties
            ({"prop1": "match"}, False)
        ],
    )
    @pytest.mark.xfail(reason="unevaluatedProperties is not implemented")
    def test_ref_creates_new_scope_when_adjacent_to_keywords(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {"A": {"unevaluatedProperties": False}},
            "properties": {"prop1": {"type": "string"}},
            "$ref": "#/$defs/A",
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
            # do not evaluate the $ref inside the enum, matching any string
            ("this is a string", False),
            # do not evaluate the $ref inside the enum, definition exact match
            ({"type": "string"}, False),
            # match the enum exactly
            ({"$ref": "#/$defs/a_string"}, True),
        ],
    )
    def test_naive_replacement_of_ref_with_its_destination_is_not_correct(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {"a_string": {"type": "string"}},
            "enum": [{"$ref": "#/$defs/a_string"}],
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
            # invalid on inner field
            ({"foo": {"bar": 1}, "bar": "a"}, False),
            # invalid on outer field
            ({"foo": {"bar": "a"}, "bar": 1}, False),
            # valid on both fields
            (
                {
                    "foo": {"bar": "a"},
                    "bar": "a",
                },
                True,
            ),
        ],
    )
    def test_refs_with_relative_uris_and_defs(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "http://example.com/schema-relative-uri-defs1.json",
            "properties": {
                "foo": {
                    "$id": "schema-relative-uri-defs2.json",
                    "$defs": {"inner": {"properties": {"bar": {"type": "string"}}}},
                    "$ref": "#/$defs/inner",
                }
            },
            "$ref": "schema-relative-uri-defs2.json",
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
            # invalid on inner field
            ({"foo": {"bar": 1}, "bar": "a"}, False),
            # invalid on outer field
            ({"foo": {"bar": "a"}, "bar": 1}, False),
            # valid on both fields
            ({"foo": {"bar": "a"}, "bar": "a"}, True),
        ],
    )
    def test_relative_refs_with_absolute_uris_and_defs(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "http://example.com/schema-refs-absolute-uris-defs1.json",
            "properties": {
                "foo": {
                    "$id": "http://example.com/schema-refs-absolute-uris-defs2.json",
                    "$defs": {"inner": {"properties": {"bar": {"type": "string"}}}},
                    "$ref": "#/$defs/inner",
                }
            },
            "$ref": "schema-refs-absolute-uris-defs2.json",
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
            # non-number is invalid
            ("a", False),
        ],
    )
    def test_id_must_be_resolved_against_nearest_parent_not_just_immediate_parent(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "http://example.com/a.json",
            "$defs": {
                "x": {
                    "$id": "http://example.com/b/c.json",
                    "not": {"$defs": {"y": {"$id": "d.json", "type": "number"}}},
                }
            },
            "allOf": [{"$ref": "http://example.com/b/d.json"}],
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
            # data is valid against first definition
            (5, True),
            # data is invalid against first definition
            (50, False),
        ],
    )
    def test_order_of_evaluation_id_and_ref(self, test_object, valid):
        schema = {
            "$comment": "$id must be evaluated before $ref to get the proper $ref destination",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://example.com/draft2020-12/ref-and-id1/base.json",
            "$ref": "int.json",
            "$defs": {
                "bigint": {
                    "$comment": "canonical uri: https://example.com/ref-and-id1/int.json",
                    "$id": "int.json",
                    "maximum": 10,
                },
                "smallint": {
                    "$comment": "canonical uri: https://example.com/ref-and-id1-int.json",
                    "$id": "/draft2020-12/ref-and-id1-int.json",
                    "maximum": 2,
                },
            },
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
            # data is valid against first definition
            (5, True),
            # data is invalid against first definition
            (50, False),
        ],
    )
    def test_order_of_evaluation_id_and_anchor_and_ref(self, test_object, valid):
        schema = {
            "$comment": "$id must be evaluated before $ref to get the proper $ref destination",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://example.com/draft2020-12/ref-and-id2/base.json",
            "$ref": "#bigint",
            "$defs": {
                "bigint": {
                    "$comment": "canonical uri: /ref-and-id2/base.json#/$defs/bigint; another valid uri for this location: /ref-and-id2/base.json#bigint",
                    "$anchor": "bigint",
                    "maximum": 10,
                },
                "smallint": {
                    "$comment": "canonical uri: https://example.com/ref-and-id2#/$defs/smallint; another valid uri for this location: https://example.com/ref-and-id2/#bigint",
                    "$id": "https://example.com/draft2020-12/ref-and-id2/",
                    "$anchor": "bigint",
                    "maximum": 2,
                },
            },
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
            # valid under the URN IDed schema
            ({"foo": 37}, True),
            # invalid under the URN IDed schema
            ({"foo": 12}, False),
        ],
    )
    def test_simple_URN_base_URI_with_ref_via_the_URN(self, test_object, valid):
        schema = {
            "$comment": "URIs do not have to have HTTP(s) schemes",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "urn:uuid:deadbeef-1234-ffff-ffff-4321feebdaed",
            "minimum": 30,
            "properties": {"foo": {"$ref": "urn:uuid:deadbeef-1234-ffff-ffff-4321feebdaed"}},
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
            # a string is valid
            ({"foo": "bar"}, True),
            # a non-string is invalid
            ({"foo": 12}, False),
        ],
    )
    def test_simple_URN_base_URI_with_JSON_pointer(self, test_object, valid):
        schema = {
            "$comment": "URIs do not have to have HTTP(s) schemes",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "urn:uuid:deadbeef-1234-00ff-ff00-4321feebdaed",
            "properties": {"foo": {"$ref": "#/$defs/bar"}},
            "$defs": {"bar": {"type": "string"}},
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
            # a string is valid
            ({"foo": "bar"}, True),
            # a non-string is invalid
            ({"foo": 12}, False),
        ],
    )
    def test_URN_base_URI_with_NSS(self, test_object, valid):
        schema = {
            "$comment": "RFC 8141 §2.2",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "urn:example:1/406/47452/2",
            "properties": {"foo": {"$ref": "#/$defs/bar"}},
            "$defs": {"bar": {"type": "string"}},
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
            # a string is valid
            ({"foo": "bar"}, True),
            # a non-string is invalid
            ({"foo": 12}, False),
        ],
    )
    def test_URN_base_URI_with_r_component(self, test_object, valid):
        schema = {
            "$comment": "RFC 8141 §2.3.1",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "urn:example:foo-bar-baz-qux?+CCResolve:cc=uk",
            "properties": {"foo": {"$ref": "#/$defs/bar"}},
            "$defs": {"bar": {"type": "string"}},
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
            # a string is valid
            ({"foo": "bar"}, True),
            # a non-string is invalid
            ({"foo": 12}, False),
        ],
    )
    def test_URN_base_URI_with_q_component(self, test_object, valid):
        schema = {
            "$comment": "RFC 8141 §2.3.2",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "urn:example:weather?=op=map&lat=39.56&lon=-104.85&datetime=1969-07-21T02:56:15Z",
            "properties": {"foo": {"$ref": "#/$defs/bar"}},
            "$defs": {"bar": {"type": "string"}},
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
            # a string is valid
            ({"foo": "bar"}, True),
            # a non-string is invalid
            ({"foo": 12}, False),
        ],
    )
    def test_URN_base_URI_with_URN_and_JSON_pointer_ref(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "urn:uuid:deadbeef-1234-0000-0000-4321feebdaed",
            "properties": {"foo": {"$ref": "urn:uuid:deadbeef-1234-0000-0000-4321feebdaed#/$defs/bar"}},
            "$defs": {"bar": {"type": "string"}},
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
            # a string is valid
            ({"foo": "bar"}, True),
            # a non-string is invalid
            ({"foo": 12}, False),
        ],
    )
    def test_URN_base_URI_with_URN_and_anchor_ref(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "urn:uuid:deadbeef-1234-ff00-00ff-4321feebdaed",
            "properties": {"foo": {"$ref": "urn:uuid:deadbeef-1234-ff00-00ff-4321feebdaed#something"}},
            "$defs": {"bar": {"$anchor": "something", "type": "string"}},
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
            # a string is valid
            ("bar", True),
            # a non-string is invalid
            (12, False),
        ],
    )
    def test_URN_ref_with_nested_pointer_ref(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": "urn:uuid:deadbeef-4321-ffff-ffff-1234feebdaed",
            "$defs": {
                "foo": {
                    "$id": "urn:uuid:deadbeef-4321-ffff-ffff-1234feebdaed",
                    "$defs": {"bar": {"type": "string"}},
                    "$ref": "#/$defs/bar",
                }
            },
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
            # a non-integer is invalid due to the $ref
            ("foo", False),
            # an integer is valid
            (12, True),
        ],
    )
    @pytest.mark.xfail(reason="if not implemented")
    def test_ref_to_if(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": "http://example.com/ref/if",
            "if": {"$id": "http://example.com/ref/if", "type": "integer"},
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
            # a non-integer is invalid due to the $ref
            ("foo", False),
            # an integer is valid
            (12, True),
        ],
    )
    @pytest.mark.xfail(reason="then not implemented")
    def test_ref_to_then(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": "http://example.com/ref/then",
            "then": {"$id": "http://example.com/ref/then", "type": "integer"},
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
            # a non-integer is invalid due to the $ref
            ("foo", False),
            # an integer is valid
            (12, True),
        ],
    )
    @pytest.mark.xfail(reason="else not implemented")
    def test_ref_to_else(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": "http://example.com/ref/else",
            "else": {"$id": "http://example.com/ref/else", "type": "integer"},
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
            # a string is valid
            ("foo", True),
            # an integer is invalid
            (12, False),
        ],
    )
    def test_ref_with_absolute_path_reference(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "http://example.com/ref/absref.json",
            "$defs": {
                "a": {"$id": "http://example.com/ref/absref/foobar.json", "type": "number"},
                "b": {"$id": "http://example.com/absref/foobar.json", "type": "string"},
            },
            "$ref": "/absref/foobar.json",
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
            # non-number is invalid
            ("a", False),
        ],
    )
    def test_id_with_file_URI_still_resolves_pointers___nix(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "file:///folder/file.json",
            "$defs": {"foo": {"type": "number"}},
            "$ref": "#/$defs/foo",
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
            # non-number is invalid
            ("a", False),
        ],
    )
    def test_id_with_file_URI_still_resolves_pointers___windows(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "file:///c:/folder/file.json",
            "$defs": {"foo": {"type": "number"}},
            "$ref": "#/$defs/foo",
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
            # non-number is invalid
            ("a", False),
        ],
    )
    def test_empty_tokens_in_ref_json_pointer(self, test_object, valid):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {"": {"$defs": {"": {"type": "number"}}}},
            "allOf": [{"$ref": "#/$defs//$defs/"}],
        }
        if valid:
            validate(instance=test_object, schema=schema)
            generate_and_check(test_object, schema)
        else:
            with pytest.raises(ValidationError):
                validate(instance=test_object, schema=schema)
            check_match_failure(bad_string=json_dumps(test_object), schema_obj=schema)
