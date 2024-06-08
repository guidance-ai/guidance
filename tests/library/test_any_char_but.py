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


def test_singleton_compliment():
    # Char in middle of range
    not_newline = [chr(i) for i in range(128) if i != 10]
    assert any_char_but(not_newline).match(chr(10)) is not None
    assert any_char_but(not_newline).match("".join(not_newline)) is None

    # Char at start of range
    not_char0 = [chr(i) for i in range(128) if i != 0]
    assert any_char_but(not_char0).match(chr(0)) is not None
    assert any_char_but(not_char0).match("".join(not_char0)) is None

    # Char at end of range
    not_char127 = [chr(i) for i in range(128) if i != 127]
    assert any_char_but(not_char127).match(chr(127)) is not None
    assert any_char_but(not_char127).match("".join(not_char127)) is None
