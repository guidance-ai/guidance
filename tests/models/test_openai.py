import pytest
import guidance
from guidance import gen, capture, select
from ..utils import get_model

def test_openai_basic():
    try:
        lm = guidance.models.OpenAI("text-davinci-003")
    except:
        pytest.skip("Skipping OpenAI test because we can't load the model!")
    # lm = guidance.models.Transformers("gpt2")
    lm += "Count to 20: 1,2,3,4,"
    nl = "\n"
    lm += f"""\
    5,6,7"""
    lm += f"""{gen(max_tokens=1, suffix=nl)}aaaaaa"""
    assert str(lm)[-5:] == "aaaaa"

def test_openai_select():
    try:
        lm = guidance.models.OpenAI("text-davinci-003")
    except:
        pytest.skip("Skipping OpenAI test because we can't load the model!")
    lm += "Pick a number: "
    lm += select(["1", "11", "111", "1111", "11111", "111111", "1111111"], name='the number')
    assert str(lm)[-1] in "123"