import pytest
from guidance import select, gen, models, instruction
from ..utils import get_model

def test_basic():
    try:
        vmodel = models.VertexAI("text-bison@001")
    except:
        pytest.skip("Skipping OpenAI test because we can't load the model!")

    with instruction():
        lm = vmodel + "this is a test about"
    lm += gen("test", max_tokens=100)
    assert len(lm["test"]) > 0