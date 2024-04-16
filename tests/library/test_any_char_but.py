from guidance import any_char_but, models


def test_single_char():
    model = models.Mock("<s>abc")
    assert str(model + "<s>" + any_char_but("a")) != "<s>a"
    assert str(model + "<s>" + any_char_but("!")) == "<s>a"


def test_multi_char():
    model = models.Mock(["<s>abc", "<s>bbc"])
    assert str(model + "<s>" + any_char_but("ab")) not in ("<s>a", "<s>b")
    assert str(model + "<s>" + any_char_but("a!")) == "<s>b"
    assert str(model + "<s>" + any_char_but("5b")) == "<s>a"
