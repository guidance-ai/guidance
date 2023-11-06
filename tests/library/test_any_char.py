from guidance import models, any_char

def test_single_char():
    model = models.LocalMock("<s>abc")
    assert str(model + '<s>' + any_char()) == "<s>a"