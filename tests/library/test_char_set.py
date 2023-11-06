from guidance import models, char_set

def test_single_char():
    model = models.LocalMock("<s>abc")
    assert str(model + '<s>' + char_set("a")) == "<s>a"
    assert str(model + '<s>' + char_set("ab")) == "<s>a"
    assert str(model + '<s>' + char_set("ba")) == "<s>a"
    assert str(model + '<s>' + char_set("b")) == "<s>b"

def test_char_range():
    model = models.LocalMock("<s>bac")
    assert str(model + '<s>' + char_set("a-c")) == "<s>b"
    assert str(model + '<s>' + char_set("b-z")) == "<s>b"
    assert str(model + '<s>' + char_set("0-9")) != "<s>b"
    assert str(model + '<s>' + char_set("b0-9")) == "<s>b"