from guidance import models, block, any_char

def test_text_opener():
    model = models.LocalMock("<s>open texta")
    with block(opener="open text"):
        model += any_char()
    assert str(model) == "open texta"

def test_text_closer():
    model = models.LocalMock("<s>aclose text")
    model += "<s>"
    with block(closer="close text"):
        model += any_char()
    assert str(model) == "<s>aclose text"

def test_grammar_opener():
    model = models.LocalMock("<s>open texta")
    with block(opener="open tex" + any_char()):
        model += any_char()
    assert str(model) == "open texta"

def test_grammar_closer():
    model = models.LocalMock(["<s>aclose text", "<s>close text"])
    model += "<s>"
    try:
        with block(closer=any_char() + "lose text"):
            model += any_char()
    except:
        return # we expect an exception
    assert False, "We should have thrown an exception using a context (prompt) based grammar in the closer!"