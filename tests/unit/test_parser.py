from guidance import char_set, one_or_more, select, string, zero_or_more
from guidance._parser import ByteParser


def test_one_or_more():
    g = one_or_more("a")
    parser = ByteParser(g)
    assert parser.valid_next_bytes() == set([b"a"])
    parser.consume_bytes(b"a")
    assert parser.valid_next_bytes() == set([b"a"])


def test_zero_or_more_and_one_or_more():
    g = zero_or_more("a") + one_or_more("b")
    parser = ByteParser(g)
    assert parser.valid_next_bytes() == set([b"a", b"b"])
    parser.consume_bytes(b"a")
    assert parser.valid_next_bytes() == set([b"a", b"b"])
    parser.consume_bytes(b"b")
    assert parser.valid_next_bytes() == set([b"b"])

    parser = ByteParser(g)
    assert parser.valid_next_bytes() == set([b"a", b"b"])
    parser.consume_bytes(b"b")
    assert parser.valid_next_bytes() == set([b"b"])
    parser.consume_bytes(b"b")
    assert parser.valid_next_bytes() == set([b"b"])


def test_zero_or_more_and_one_or_more_mixed():
    g = zero_or_more("a") + "test" + one_or_more("b")
    parser = ByteParser(g)
    assert parser.valid_next_bytes() == set([b"a", b"t"])
    parser.consume_bytes(b"t")
    parser.consume_bytes(b"e")
    parser.consume_bytes(b"s")
    assert parser.valid_next_bytes() == set([b"t"])
    parser.consume_bytes(b"t")
    assert parser.valid_next_bytes() == set([b"b"])


def test_select():
    g = select(["bob", "bill", "sue"])
    parser = ByteParser(g)
    assert parser.valid_next_bytes() == set([b"b", b"s"])
    parser.consume_bytes(b"s")
    assert parser.valid_next_bytes() == set([b"u"])
    parser.consume_bytes(b"u")
    assert parser.valid_next_bytes() == set([b"e"])


def test_select_nested():
    g = select(["bob", "bill", select(["mark", "mary"])])
    parser = ByteParser(g)
    assert parser.valid_next_bytes() == set([b"b", b"m"])
    parser.consume_bytes(b"m")
    assert parser.valid_next_bytes() == set([b"a"])
    parser.consume_bytes(b"a")
    assert parser.valid_next_bytes() == set([b"r"])
    parser.consume_bytes(b"r")
    assert parser.valid_next_bytes() == set([b"k", b"y"])


def test_select_joined():
    g = select(["bob", "bill"]) + select(["mark", "mary"])
    parser = ByteParser(g)
    assert parser.valid_next_bytes() == set([b"b"])
    parser.consume_bytes(b"b")
    assert parser.valid_next_bytes() == set([b"o", b"i"])
    parser.consume_bytes(b"i")
    assert parser.valid_next_bytes() == set([b"l"])
    parser.consume_bytes(b"l")
    assert parser.valid_next_bytes() == set([b"l"])
    parser.consume_bytes(b"l")
    assert parser.valid_next_bytes() == set([b"m"])
    parser.consume_bytes(b"m")
    assert parser.valid_next_bytes() == set([b"a"])
    parser.consume_bytes(b"a")
    assert parser.valid_next_bytes() == set([b"r"])
    parser.consume_bytes(b"r")
    assert parser.valid_next_bytes() == set([b"k", b"y"])


def test_char_set():
    g = char_set("b-f")
    parser = ByteParser(g)
    assert parser.valid_next_bytes() == {bytes([i]) for i in range(ord("b"), ord("f") + 1)}
    parser.consume_bytes(b"b")


def test_byte_mask_char_set():
    g = char_set("b-f")
    parser = ByteParser(g)
    m = parser.next_byte_mask()
    for i in range(256):
        if ord(b"b") <= i <= ord(b"f"):
            assert m[i]
        else:
            assert not m[i]


def test_byte_mask_char_set2():
    g = char_set("bf")
    parser = ByteParser(g)
    m = parser.next_byte_mask()
    for i in range(256):
        if i == ord(b"b") or i == ord(b"f"):
            assert m[i]
        else:
            assert not m[i]


def test_char_set_one_or_more():
    g = one_or_more(char_set("b-f"))
    parser = ByteParser(g)
    assert parser.valid_next_bytes() == {bytes([i]) for i in range(ord("b"), ord("f") + 1)}
    parser.consume_bytes(b"b")
    assert parser.valid_next_bytes() == {bytes([i]) for i in range(ord("b"), ord("f") + 1)}
    parser.consume_bytes(b"b")
    assert parser.valid_next_bytes() == {bytes([i]) for i in range(ord("b"), ord("f") + 1)}
    parser.consume_bytes(b"f")
    assert parser.valid_next_bytes() == {bytes([i]) for i in range(ord("b"), ord("f") + 1)}


def test_string_utf8():
    b = bytes("¶", encoding="utf8")
    g = string("¶")
    parser = ByteParser(g)
    assert parser.valid_next_bytes() == set([b[:1]])
    parser.consume_bytes(b[:1])
    assert parser.valid_next_bytes() == set([b[1:]])
    parser.consume_bytes(b[1:])
