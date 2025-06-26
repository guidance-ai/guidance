import pytest

from guidance import models, one_or_more, regex


def test_string():
    model = models.Mock("<s>aaabc")
    assert str(model + "<s>" + one_or_more("a")) == "<s>aa"


@pytest.mark.xfail(
    reason=""""
        guidance allows generating 'bad'/masked tokens when the grammar is in an accepting state, terminating the grammar if such tokens are generated.
        This breaks this particular test, as 'aa' is generated where 'ab' is expected, and we don't keep the leading 'a' under this exceptional
        termination case.
    """
)
def test_string_token_boundary():
    model = models.Mock("<s>aaabc")
    assert str(model + "<s>" + one_or_more("a")) == "<s>aaa"


def test_grammar():
    model = models.Mock("<s>bac")
    assert str(model + "<s>" + one_or_more(regex(r"[ab]"))) == "<s>ba"


def test_at_least_one():
    model = models.Mock("<s>cbac")
    assert not str(model + "<s>" + one_or_more(regex(r"[ab]"))).startswith("<s>c")
