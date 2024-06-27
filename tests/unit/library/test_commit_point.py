from guidance import commit_point, one_or_more, byte_range, models


def test_hidden():
    model = models.Mock()
    model += " one" + commit_point(" two", hidden=True) + " three"
    assert str(model) == " one three"


def test_pseudo_regex():
    grm = commit_point(
        '"' + byte_range(b"A", b"Z") + one_or_more(byte_range(b"a", b"z")) + '"'
    )
    assert grm.match('"Abcd"') is not None
    assert grm.match('"Zyxwv"') is not None
    assert grm.match('"A"') is None
    assert grm.match('"abc"') is None
