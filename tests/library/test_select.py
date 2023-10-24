from guidance import select, models
from ..utils import get_model

def test_reset_pos():
    model = models.LocalMock()
    model += 'This is' + select(values=['bad', 'quite bad'])
    assert str(model) in ["This isbad", "This isquite bad"]

def test_simple():
    lm = get_model("transformers:gpt2")
    options = ['baad I think', 'bad I think', 'bad']
    lm = lm + 'Scott is quite ' + select(name='bad', values=options)
    assert lm['bad'] in options

def test_longer_select():
    lm = models.LocalMock(b"<s>Scott is a very nice man.")
    lm += "Scott is a very " + select(name='text', values=['nice', 'nice man.'])
    assert lm["text"] == "nice man."
