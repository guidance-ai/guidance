from guidance import char_set, models


def test_single_char():
    model = models.Mock("<s>abc")
    assert str(model + "<s>" + char_set("a")) == "<s>a"
    assert str(model + "<s>" + char_set("ab")) == "<s>a"
    assert str(model + "<s>" + char_set("ba")) == "<s>a"
    assert str(model + "<s>" + char_set("b")) == "<s>b"


def test_char_range():
    model = models.Mock("<s>bac")
    assert str(model + "<s>" + char_set("a-c")) == "<s>b"
    assert str(model + "<s>" + char_set("b-z")) == "<s>b"
    assert str(model + "<s>" + char_set("0-9")) != "<s>b"
    assert str(model + "<s>" + char_set("b0-9")) == "<s>b"
