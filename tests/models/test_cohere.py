import pytest
import guidance
from guidance import gen, capture, select, user, system, assistant, instruction

def test_lite_llm_basic():
    try:
        lm = guidance.models.CohereCompletion("command-nightly")
    except:
        pytest.skip("Skipping Cohere test because we can't load the model!")
    lm += "Count to 20: 1,2,3,4,"
    nl = "\n"
    lm += f"""\
    5,6,7"""
    lm += f"""{gen(max_tokens=1, suffix=nl)}aaaaaa"""
    assert str(lm)[-5:] == "aaaaa"

def test_lite_llm_instruct():
    try:
        lm = guidance.models.CohereInstruct("command-nightly")
    except:
        pytest.skip("Skipping LiteLLM test because we can't load the model!")
    with instruction():
        lm += "Count to 20."
    lm += gen('val', max_tokens=1)
    assert len(lm['val']) > 0