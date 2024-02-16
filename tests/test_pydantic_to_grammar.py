from typing import List

import pydantic

from guidance._grammar import GrammarFunction
from guidance._pydantic_to_grammar import pydantic_model_to_grammar


def check_object_with_grammar(
    target_object: pydantic.BaseModel, grammar: GrammarFunction
):
    print(f"Checking {target_object}")
    json_string = target_object.model_dump_json()
    matches = grammar.match(json_string.encode(), raise_exceptions=True)
    assert matches.partial == False


def test_simple_model():
    class Simple(pydantic.BaseModel):
        my_string: str

    my_obj = Simple(my_string="some string")

    grammar = pydantic_model_to_grammar(my_obj)
    check_object_with_grammar(my_obj, grammar)


def test_model_with_int_list():
    class MyModel(pydantic.BaseModel):
        my_list: List[int] = pydantic.Field(default_factory=list)

    my_obj = MyModel(my_list=[1, 2, 3, 4])
    grammar = pydantic_model_to_grammar(my_obj)
    check_object_with_grammar(my_obj, grammar)


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
    grammar = pydantic_model_to_grammar(my_obj)
    check_object_with_grammar(my_obj, grammar)
