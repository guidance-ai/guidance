import guidance
from guidance import gen, capture
from ..utils import get_model

def test_openai_basic():
    lm = guidance.models.OpenAI("text-davinci-003")
    # lm = guidance.models.Transformers("gpt2")
    lm += "Count to 20: 1,2,3,4,"
    nl = "\n"
    lm += f"""\
    5,6,7"""
    lm += f"""{gen(max_tokens=1, suffix=nl)}aaaaaa"""
    assert str(lm)[-5:] == "aaaaa"