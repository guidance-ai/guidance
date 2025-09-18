import pytest

from guidance import block, models, regex


def test_text_opener():
    model = models.Mock("<s>open texta")
    with block(opener="open text"):
        model += regex(r".")
    assert str(model) == "open texta"


def test_text_closer():
    # NOTE(nopdive): Behavioral change, no longer need closer for str call.
    model = models.Mock("<s>a")
    model += "<s>"
    with block(closer="close text"):
        model += regex(r".")
    assert str(model) == "<s>a"


def test_grammar_opener():
    model = models.Mock("<s>open texta")
    with block(opener="open tex" + regex(r".")):
        model += regex(r".")
    assert str(model) == "open texta"


# TODO(nopdive): Review this exception later -- how should we be going about grammars in blocks overall.
@pytest.mark.skip(reason="requires review")
def test_grammar_closer():
    model = models.Mock(["<s>aclose text", "<s>close text"])
    model += "<s>"
    try:
        with block(closer=regex(r".") + "lose text"):
            model += regex(r".")
    except:
        return  # we expect an exception
    raise AssertionError("We should have thrown an exception using a context (prompt) based grammar in the closer!")


def test_block_name_capture():
    model = models.Mock("<s>open texta")
    with block("my_data"):
        model += "open text"
        model += regex(r".")
    assert model["my_data"] == "open texta"


def test_block_name_capture_closed():
    model = models.Mock("<s>open texta")
    with block("my_data"):
        model += "open text"
        model += regex(r".")
    model += "tmp"
    assert model["my_data"] == "open texta"
