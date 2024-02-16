import pydantic

from guidance._grammar import GrammarFunction
from guidance._pydantic_to_grammar import pydantic_model_to_grammar


def check_object_with_grammar(target_object: pydantic.BaseModel, grammar: GrammarFunction):
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



