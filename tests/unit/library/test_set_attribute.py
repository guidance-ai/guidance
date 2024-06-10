from guidance import set_attribute, models, gen, select

from ..utils import get_model


def test_set_attribute():
    lm = models.Mock(b"<s>1234233234<s>")
    with set_attribute("echo", False):
        lm += "1"
        assert lm.echo == False
        out = (lm + gen('name', max_tokens=1))['name']
    assert out == "2"