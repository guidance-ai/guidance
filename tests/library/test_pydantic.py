import inspect
from json import dumps as json_dumps
from typing import Any, List, Literal, Type, Union

import pydantic
import pytest
from pydantic.json_schema import to_jsonable_python as pydantic_to_jsonable_python

from guidance import json as gen_json
from guidance import models
from guidance import pydantic as gen_pydantic
from guidance._parser import ParserException


def to_compact_json(target: Any) -> str:
    """
    See 'Compact Encoding':
    https://docs.python.org/3/library/json.html
    Since this is ultimately about the generated
    output, we don't need to worry about pretty printing
    and whitespace

    This function differs from the identically named one in
    `library._json` by the  `default=pydantic_to_jsonable_python`
    kwarg to json_dumps, which allows json_dumps to dump pydantic
    objects.
    """
    return json_dumps(
        target, separators=(",", ":"), default=pydantic_to_jsonable_python
    )


def validate_obj(
    target_obj: Any,
    pydantic_model: Union[Type[pydantic.BaseModel], pydantic.TypeAdapter],
):
    if inspect.isclass(pydantic_model) and issubclass(
        pydantic_model, pydantic.BaseModel
    ):
        return pydantic_model.model_validate(target_obj, strict=True)
    if isinstance(pydantic_model, pydantic.TypeAdapter):
        return pydantic_model.validate_python(target_obj, strict=True)
    raise TypeError(
        f"Expected pydantic_model to be a BaseModel or TypeAdapter, got {type(pydantic_model)}"
    )


def validate_string(
    target_str: Any,
    pydantic_model: Union[Type[pydantic.BaseModel], pydantic.TypeAdapter],
):
    if inspect.isclass(pydantic_model) and issubclass(
        pydantic_model, pydantic.BaseModel
    ):
        return pydantic_model.model_validate_json(target_str, strict=True)
    if isinstance(pydantic_model, pydantic.TypeAdapter):
        return pydantic_model.validate_json(target_str, strict=True)
    raise TypeError(
        f"Expected pydantic_model to be a BaseModel or TypeAdapter, got {type(pydantic_model)}"
    )


def generate_and_check(
    target_obj: Any,
    pydantic_model: Union[Type[pydantic.BaseModel], pydantic.TypeAdapter],
):
    # Sanity check what we're being asked
    validate_obj(target_obj, pydantic_model)

    # Define grammar with capture key
    CAPTURE_KEY = "my_capture"
    grammar = gen_pydantic(pydantic_model, name=CAPTURE_KEY)

    # Test that grammar matches string
    json_string = to_compact_json(target_obj)
    matches = grammar.match(json_string, raise_exceptions=True)
    assert matches.partial == False

    # Run with the mock model
    prepared_string = f"<s>{json_string}"
    lm = models.Mock(prepared_string.encode(), echo=False)
    lm += grammar

    # Make sure the round trip works
    round_trip_object = validate_string(lm[CAPTURE_KEY], pydantic_model)
    assert round_trip_object == target_obj


def test_simple_model():
    class Simple(pydantic.BaseModel):
        my_string: str

    my_obj = Simple(my_string="some string")
    generate_and_check(my_obj, Simple)


def test_model_with_int_list():
    class MyModel(pydantic.BaseModel):
        my_list: List[int] = pydantic.Field(default_factory=list)

    my_obj = MyModel(my_list=[1, 2, 3, 4])
    generate_and_check(my_obj, MyModel)


def test_nested_model():
    class A(pydantic.BaseModel):
        my_str: str = pydantic.Field(default="my_a_str")

    class B(pydantic.BaseModel):
        my_str: str = pydantic.Field(default="my_b_str")
        my_A: A = pydantic.Field(default_factory=A)

    class C(pydantic.BaseModel):
        my_str: str = pydantic.Field(default="my_c_str")
        my_B: B = pydantic.Field(default_factory=B)

    my_obj = C(my_str="some other string!")
    generate_and_check(my_obj, C)


@pytest.mark.parametrize("has_A", [True, False])
def test_model_with_optional(has_A):
    class A(pydantic.BaseModel):
        my_str: str = pydantic.Field(default="my_a_str")

    class B(pydantic.BaseModel):
        b_str: str = pydantic.Field(default="Some string")
        my_A: Union[A, None] = pydantic.Field(default=None)

    if has_A:
        my_obj = B(my_A=A(my_str="a long string or two"))
    else:
        my_obj = B(b_str="A long b string")

    generate_and_check(my_obj, B)


@pytest.mark.parametrize("target_obj", ["hello", 42, False])
def test_literal(target_obj):
    model = pydantic.TypeAdapter(Literal["hello", 42, False])
    generate_and_check(target_obj, model)


class TestTuple:

    @pytest.mark.parametrize("target_obj", [(1,), (1, 2), (1, 2, 3, 4, 5)])
    def test_variadic(self, target_obj):
        model = pydantic.TypeAdapter(tuple[int, ...])
        generate_and_check(target_obj, model)

    @pytest.mark.xfail(
        reason="Underlying guidance.json does not yet support sequences with length specifications"
    )
    def test_homogeneous(self):
        model = pydantic.TypeAdapter(tuple[float, float, float])
        generate_and_check((3.14, 2.718, 1.41), model)

    @pytest.mark.xfail(
        reason="Underlying guidance.json does not yet support prefixItems"
    )
    def test_heterogeneous(self):
        model = pydantic.TypeAdapter(tuple[int, bool])
        generate_and_check((1, True), model)


class TestDict:
    def test_simple(self):
        model = pydantic.TypeAdapter(dict[str, int])
        generate_and_check({"hello": 42}, model)

    @pytest.mark.xfail(reason="Json schemas cannot specify non-string keys")
    def test_non_string_keys_fail(self):
        model = pydantic.TypeAdapter(dict[int, int])
        bad_str = '{"one":2}'
        grammar = gen_json(model.json_schema())
        with pytest.raises(ParserException):
            grammar.match(bad_str, raise_exceptions=True)

    def test_prevent_non_string_keys(self):
        "Test that we catch attempt to generate non-string keys"
        model = pydantic.TypeAdapter(dict[int, int])
        with pytest.raises(TypeError):
            generate_and_check({1: 2}, model)
