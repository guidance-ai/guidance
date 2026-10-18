import pytest

from guidance import capture
from guidance.library._subgrammar import lexeme, subgrammar


class TestEndingLexemeAmbiguous:
    @pytest.mark.parametrize("skip_rx", [None, r"\s", r"\s+", r"\s*"])
    @pytest.mark.parametrize("string", ["123"])
    def test_lexeme_can_be_done_even_if_could_match_more(self, string, skip_rx):
        g1 = subgrammar(body=lexeme(r"\d+"), skip_regex=skip_rx, name="mycap")
        assert (m := g1.match(string)) is not None and m.captures["mycap"] == string
        g2 = g1 + "x"
        assert (m := g2.match(f"{string}x")) is not None and m.captures["mycap"] == string

    @pytest.mark.parametrize("string", ["1", "123", "1x", "123x"])
    def test_nullable_final_lexeme(self, string):
        g = subgrammar(body=lexeme(r"\d+") + lexeme(r"x?"), name="mycap")
        match = g.match(string)
        assert match is not None
        assert match.captures["mycap"] == string


@pytest.mark.parametrize("string", ["x=1", "x = 1", "x = // note\n1", "x/=/1"])
def test_skip_regex_containing_slash(string):
    body = capture(lexeme(r"[a-z]+"), "key") + "=" + capture(lexeme(r"[0-9]+"), "value")
    g = subgrammar(body=body, skip_regex=r"(\s|/|//[^\n]*)+", name="mycap")
    match = g.match(string)
    assert match is not None
    assert match.captures["mycap"] == string


@pytest.mark.parametrize(
    "skip_regex",
    [
        "(?x)\n    \\s+\n",
        r"""(?x)
        (
            \s+          # whitespace
          | \#[^\n]*     # hash comment
        )+
        """,
    ],
)
@pytest.mark.parametrize("string", ["x=1", "x = 1", "x\n=\n1"])
def test_multiline_verbose_skip_regex(skip_regex, string):
    body = capture(lexeme(r"[a-z]+"), "key") + "=" + capture(lexeme(r"[0-9]+"), "value")
    g = subgrammar(body=body, skip_regex=skip_regex, name="mycap")
    match = g.match(string)
    assert match is not None
    assert match.captures["mycap"] == string
