from guidance import any_char, models


def test_single_char():
    model = models.Mock("<s>abc")
    assert str(model + "<s>" + any_char()) == "<s>a"
