from guidance import models, silent


def test_basic():
    lm = models.Mock()
    lm += "Start text"
    with silent():
        lm += "silent text"
    lm += "end text"
    assert "display: none" in lm._state
