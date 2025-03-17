import pytest
from guidance import models, silent

@pytest.mark.xfail(reason="silent isn't currently implemented")
def test_basic():
    lm = models.Mock()
    lm += "Start text"
    with silent():
        lm += "silent text"
    lm += "end text"
    assert "display: none" in lm._state
    lm.close()
