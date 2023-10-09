import guidance
from guidance import gen
from ..utils import get_model

def test_basic():
    lm = guidance.models.LM("mock")
    lm += "Start text"
    with lm.silent() as lm:
        lm += "silent text"
    lm += "end text"
    assert "display: none" in lm._state