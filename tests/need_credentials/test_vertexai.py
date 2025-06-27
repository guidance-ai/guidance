import pytest

from guidance import gen, models, role



def test_palm2_instruct():
    try:
        vmodel = models.VertexAI("text-bison@001")
    except:
        pytest.skip("Skipping VertexAI test because we can't load the model!")

    with role("instruction"):
        lm = vmodel + "this is a test about"
    lm += gen("test", max_tokens=100)
    assert len(lm["test"]) > 0


def test_palm2_chat():
    from guidance import assistant, gen, models, system, user

    try:
        vmodel = models.VertexAI("chat-bison@001")
    except:
        pytest.skip("Skipping VertexAI test because we can't load the model!")

    with system():
        lm = vmodel + "You are an always-happy agent no matter what."

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
    with system():
        lm = vmodel + "You are an always-happy agent no matter what."

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


def test_gemini_chat():
    from guidance import assistant, gen, models, user

    try:
        vmodel = models.VertexAI("gemini-pro")
    except:
        pytest.skip("Skipping VertexAI test because we can't load the model!")

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
