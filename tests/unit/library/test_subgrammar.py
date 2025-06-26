import pytest
from guidance.library._subgrammar import subgrammar, lexeme


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
