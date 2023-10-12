import guidance
from guidance import select, capture
from ..utils import get_model

def test_fstring():
    lm = get_model("transformers:gpt2")
    lm += f'this is a test {select(["item1", "item2"])}'
    assert str(lm) in ["this is a test item1", "this is a test item2"]

def test_fstring_custom():
    lm = get_model("transformers:gpt2")

    @guidance
    def my_function(lm):
        return lm + f'another {select(["item1", "item2"])}'

    lm += f'this is a test {my_function()}'
    assert str(lm) in ["this is a test another item1", "this is a test another item2"]