import pytest
from guidance import greedy_grammar, lexeme

class TestEndingLexemeAmbiguous:
    @pytest.mark.parametrize(
        "string",
        ["123"]
    )
    def test_no_skip_rx(self, string):
        g1 = greedy_grammar(body=lexeme(r"\d+"), name="mycap")
        assert (m := g1.match(string)) is not None and m.captures["mycap"] == string
        g2 = g1 + "x"
        assert (m := g2.match(f"{string}x")) is not None and m.captures["mycap"] == string

    @pytest.mark.parametrize(
        "whitespace_rx",
        [r"\s", r"\s+", r"\s*"]
    )
    @pytest.mark.parametrize(
        "string",
        ["123", "123 ", "123  "]
    )
    def test_skip_rx(self, string, whitespace_rx):
        g1 = greedy_grammar(body=lexeme(r"\d+"), skip_regex=whitespace_rx, name="mycap")
        assert (m := g1.match(string)) is not None and m.captures["mycap"] == string
        g2 = g1 + "x"
        assert (m := g2.match(f"{string}x")) is not None and m.captures["mycap"] == string

    @pytest.mark.parametrize(
        "string",
        ["1", "123", "1x", "123x"]
    )
    def test_nullable_final_lexeme(self, string):
        g = greedy_grammar(body=lexeme(r"\d+")+lexeme(r"x?"), name="mycap")
        match = g.match(string)
        assert match is not None
        assert match.captures["mycap"] == string
