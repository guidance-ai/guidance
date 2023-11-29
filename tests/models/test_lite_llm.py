import pytest
import guidance
from guidance import gen, capture, select, user, system, assistant
from ..utils import get_model

def test_lite_llm_basic_openai():
    try:
        lm = guidance.models.LiteLLMCompletion("text-curie-001")
    except:
        pytest.skip("Skipping LiteLLM test because we can't load the model!")
    lm += "Count to 20: 1,2,3,4,"
    nl = "\n"
    lm += f"""\
    5,6,7"""
    lm += f"""{gen(max_tokens=1, suffix=nl)}aaaaaa"""
    assert str(lm)[-5:] == "aaaaa"

def test_lite_llm_basic_cohere():
    try:
        lm = guidance.models.LiteLLMCompletion("command-nightly")
    except:
        pytest.skip("Skipping LiteLLM test because we can't load the model!")
    lm += "Count to 20: 1,2,3,4,"
    nl = "\n"
    lm += f"""\
    5,6,7"""
    lm += f"""{gen(max_tokens=1, suffix=nl)}aaaaaa"""
    assert str(lm)[-5:] == "aaaaa"

def test_lite_llm_select():
    try:
        lm = guidance.models.LiteLLMCompletion("text-curie-001")
    except:
        pytest.skip("Skipping LiteLLM test because we can't load the model!")
    lm += "Pick a number: "
    lm += select(["1", "11", "111", "1111", "11111", "111111", "1111111"], name='the number')
    assert str(lm)[-1] in "123"

def test_lite_llm_chat():
    try:
        lm = guidance.models.LiteLLMChat("gpt-3.5-turbo")
    except:
        pytest.skip("Skipping LiteLLM test because we can't load the model!")
    with system():
        lm += "You are a math wiz."

    with user():
        lm += "What is 1 + 1?"

    with assistant():
        lm += gen(max_tokens=10, name="text")
        lm += "Pick a number: "

    assert len(lm["text"]) > 0