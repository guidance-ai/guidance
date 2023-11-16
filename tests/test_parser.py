from guidance import one_or_more, zero_or_more, char_set, string, select
from guidance._grammar import Byte, ByteRange
from guidance._parser import EarleyCommitParser

def test_one_or_more():
    g = one_or_more("a")
    parser = EarleyCommitParser(g)
    assert parser.valid_next_bytes() == set([Byte(b'a')])
    parser.consume_byte(b'a')
    assert parser.valid_next_bytes() == set([Byte(b'a')])

def test_zero_or_more_and_one_or_more():
    g = zero_or_more("a") + one_or_more("b")
    parser = EarleyCommitParser(g)
    assert parser.valid_next_bytes() == set([Byte(b'a'), Byte(b'b')])
    parser.consume_byte(b'a')
    assert parser.valid_next_bytes() == set([Byte(b'a'), Byte(b'b')])
    parser.consume_byte(b'b')
    assert parser.valid_next_bytes() == set([Byte(b'b')])

    parser = EarleyCommitParser(g)
    assert parser.valid_next_bytes() == set([Byte(b'a'), Byte(b'b')])
    parser.consume_byte(b'b')
    assert parser.valid_next_bytes() == set([Byte(b'b')])
    parser.consume_byte(b'b')
    assert parser.valid_next_bytes() == set([Byte(b'b')])

def test_zero_or_more_and_one_or_more_mixed():
    g = zero_or_more("a") + "test" + one_or_more("b")
    parser = EarleyCommitParser(g)
    assert parser.valid_next_bytes() == set([Byte(b'a'), Byte(b't')])
    parser.consume_byte(b't')
    parser.consume_byte(b'e')
    parser.consume_byte(b's')
    assert parser.valid_next_bytes() == set([Byte(b't')])
    parser.consume_byte(b't')
    assert parser.valid_next_bytes() == set([Byte(b'b')])

def test_select():
    g = select(["bob", "bill", "sue"])
    parser = EarleyCommitParser(g)
    assert parser.valid_next_bytes() == set([Byte(b'b'), Byte(b's')])
    parser.consume_byte(b's')
    assert parser.valid_next_bytes() == set([Byte(b'u')])
    parser.consume_byte(b'u')
    assert parser.valid_next_bytes() == set([Byte(b'e')])

def test_select_nested():
    g = select(["bob", "bill", select(["mark", "mary"])])
    parser = EarleyCommitParser(g)
    assert parser.valid_next_bytes() == set([Byte(b'b'), Byte(b'm')])
    parser.consume_byte(b'm')
    assert parser.valid_next_bytes() == set([Byte(b'a')])
    parser.consume_byte(b'a')
    assert parser.valid_next_bytes() == set([Byte(b'r')])
    parser.consume_byte(b'r')
    assert parser.valid_next_bytes() == set([Byte(b'k'), Byte(b'y')])

def test_select_joined():
    g = select(["bob", "bill"]) + select(["mark", "mary"])
    parser = EarleyCommitParser(g)
    assert parser.valid_next_bytes() == set([Byte(b'b')])
    parser.consume_byte(b'b')
    assert parser.valid_next_bytes() == set([Byte(b'o'), Byte(b'i')])
    parser.consume_byte(b'i')
    assert parser.valid_next_bytes() == set([Byte(b'l')])
    parser.consume_byte(b'l')
    assert parser.valid_next_bytes() == set([Byte(b'l')])
    parser.consume_byte(b'l')
    assert parser.valid_next_bytes() == set([Byte(b'm')])
    parser.consume_byte(b'm')
    assert parser.valid_next_bytes() == set([Byte(b'a')])
    parser.consume_byte(b'a')
    assert parser.valid_next_bytes() == set([Byte(b'r')])
    parser.consume_byte(b'r')
    assert parser.valid_next_bytes() == set([Byte(b'k'), Byte(b'y')])

def test_char_set():
    g = char_set("b-f")
    parser = EarleyCommitParser(g)
    assert parser.valid_next_bytes() == set([ByteRange(b'bf')])
    parser.consume_byte(b'b')

def test_byte_mask_char_set():
    g = char_set("b-f")
    parser = EarleyCommitParser(g)
    m = parser.next_byte_mask()
    for i in range(256):
        if ord(b'b') <= i <= ord(b'f'):
            assert m[i]
        else:
            assert not m[i]

def test_byte_mask_char_set2():
    g = char_set("bf")
    parser = EarleyCommitParser(g)
    m = parser.next_byte_mask()
    for i in range(256):
        if i == ord(b'b') or i == ord(b'f'):
            assert m[i]
        else:
            assert not m[i]

def test_char_set_one_or_more():
    g = one_or_more(char_set("b-f"))
    parser = EarleyCommitParser(g)
    assert parser.valid_next_bytes() == set([ByteRange(b'bf')])
    parser.consume_byte(b'b')
    assert parser.valid_next_bytes() == set([ByteRange(b'bf')])
    parser.consume_byte(b'b')
    assert parser.valid_next_bytes() == set([ByteRange(b'bf')])
    parser.consume_byte(b'f')
    assert parser.valid_next_bytes() == set([ByteRange(b'bf')])

def test_string_utf8():
    b = bytes("¶", encoding="utf8")
    g = string("¶")
    parser = EarleyCommitParser(g)
    assert parser.valid_next_bytes() == set([Byte(b[:1])])
    parser.consume_byte(b[:1])
    assert parser.valid_next_bytes() == set([Byte(b[1:])])
    parser.consume_byte(b[1:])