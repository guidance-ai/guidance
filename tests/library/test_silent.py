from guidance import gen, models
from ..utils import get_model

def test_basic():
    lm = models.LocalMock()
    lm += "Start text"
    with lm.silent() as lm:
        lm += "silent text"
    lm += "end text"
    assert "display: none" in lm._state