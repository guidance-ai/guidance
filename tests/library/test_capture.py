from guidance import select, capture, models, one_or_more
from ..utils import get_model

def test_capture():
    model = models.LocalMock()
    model += 'This is' + capture(select(options=['bad', 'quite bad']), name="my_var")
    assert model["my_var"] in ["bad", "quite bad"]

def test_capture_star():
    lm = models.LocalMock(b"<s>1234233234<s>")
    grammar = capture(one_or_more(select(['1', '2'])), name='test')
    lm2 = lm + grammar
    assert lm2['test'] == '12'