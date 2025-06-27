import pytest


def test_gemini_pro():
    from guidance import assistant, gen, models, user

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
