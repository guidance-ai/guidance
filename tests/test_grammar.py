from guidance import models, select
from .utils import get_model

def test_select_reset_pos():
    model = models.LocalMock()
    model += 'This is' + select(options=['bad', 'quite bad'])
    assert str(model) in ["This isbad", "This isquite bad"]

def test_select_simple():
    lm = get_model("transformers:gpt2")
    options = ['baad I think', 'bad I think', 'bad']
    lm = lm + 'Scott is quite ' + select(name='bad', options=options)
    assert lm['bad'] in options

def test_select_longer():
    '''This tests to ensure that the grammar is extended greedily.'''
    lm = models.LocalMock(b"<s>Scott is a very nice man.")
    lm += "Scott is a very " + select(name='text', options=['nice', 'nice man.'])
    assert lm["text"] == 'nice man.'
