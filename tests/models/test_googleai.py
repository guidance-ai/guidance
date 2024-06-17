import pytest

from guidance import gen, instruction, models, select

from ..utils import get_model

def test_googleai_basic():
    try:
        lm = models.GoogleAICompletion("gemini-pro")
    except:
        pytest.skip("Skipping GoogleAI test because we can't load the model!")
    
    lm += "Count to 20: 1,2,3,4,"
    nl = "\n"
    lm += f"""\
5,6,7"""
    lm += f"""{gen(max_tokens=1, suffix=nl)}aaaaaa"""
    assert str(lm)[-5:] == "aaaaa"

def test_googleai_instruct():
    try:
        lm = models.GoogleAIInstruct("gemini-pro")
    except:
        pytest.skip("Skipping GoogleAI test because we can't load the model!")
    
    with instruction():
        lm += "this is a test about"
    lm += gen("test", max_tokens=100)
    assert len(lm["test"]) > 0

def test_gemini_pro():
    from guidance import assistant, gen, models, system, user

    try:
        vmodel = models.GoogleAI("gemini-pro")
    except:
        pytest.skip("Skipping GoogleAI test because we can't load the model!")

    lm = vmodel

    with user():
        lm += "The economy is crashing!"

    with assistant():
        lm += gen("test1", max_tokens=100)

    with user():
        lm += "What is the best again?"

    with assistant():
        lm += gen("test2", max_tokens=100)

    assert len(lm["test1"]) > 0
    assert len(lm["test2"]) > 0

    # second time to make sure cache reuse is okay
    lm = vmodel

    with user():
        lm += "The economy is crashing!"

    with assistant():
        lm += gen("test1", max_tokens=100)

    with user():
        lm += "What is the best again?"

    with assistant():
        lm += gen("test2", max_tokens=100)

    assert len(lm["test1"]) > 0
    assert len(lm["test2"]) > 0
    assert lm["test1"].find("<|im_end|>") < 0
