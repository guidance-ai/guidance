from guidance import models, one_or_more, regex


def test_string():
    model = models.Mock("<s>aaabc")
    try:
        assert str(model + "<s>" + one_or_more("a")) == "<s>aaa"
    finally:
        model.close()


def test_grammar():
    model = models.Mock("<s>bac")
    try:
        assert str(model + "<s>" + one_or_more(regex(r"[ab]"))) == "<s>ba"
    finally:
        model.close()


def test_at_least_one():
    model = models.Mock("<s>cbac")
    try:
        assert not str(model + "<s>" + one_or_more(regex(r"[ab]"))).startswith("<s>c")
    finally:
        model.close()
