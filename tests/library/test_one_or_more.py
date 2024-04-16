from guidance import models, one_or_more, char_set


def test_string():
    model = models.Mock("<s>aaabc")
    assert str(model + "<s>" + one_or_more("a")) == "<s>aaa"


def test_grammar():
    model = models.Mock("<s>bac")
    assert str(model + "<s>" + one_or_more(char_set("ab"))) == "<s>ba"


def test_at_least_one():
    model = models.Mock("<s>cbac")
    assert not str(model + "<s>" + one_or_more(char_set("ab"))).startswith("<s>c")
