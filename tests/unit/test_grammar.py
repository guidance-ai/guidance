import pytest
import guidance
from guidance import gen, models, optional, select, string
from guidance._parser import ByteParserException


def test_select_reset_pos():
    model = models.Mock()
    model += "This is" + select(options=["bad", "quite bad"])
    assert str(model) in ["This isbad", "This isquite bad"]


def test_select_longer():
    """This tests to ensure that the grammar is extended greedily."""
    lm = models.Mock(b"<s>Scott is a very nice man.")
    lm += "Scott is a very " + select(name="text", options=["nice", "nice man."])
    assert lm["text"] == "nice man."


@pytest.mark.xfail(reason="Lexer sees 'a' then 'b' and here decides to continue matching abq)")
def test_select_ambiguous_lexeme_boundary():
    lm = models.Mock(b"<s>abQ<s>")
    lm += select(options=["a", "abq", "c"], name="prefix") + optional("bQ")
    assert lm["prefix"] == "a"


def test_select_ambiguous_lexeme_boundary_manual_fix():
    # Manual fix to the issue in test_select_ambiguous_lexeme_boundary by splitting the "abq" lexeme into two lexemes
    lm = models.Mock(b"<s>abQ<s>")
    lm += select(options=["a", string("a") + string("bq"), "c"], name="prefix") + optional("bQ")
    assert lm["prefix"] == "a"


def test_select_empty():
    """This tests to ensure that we save empty capture groups."""
    lm = models.Mock(b"<s>This is a test")
    lm += "This is a" + select(name="text", options=["", "nope"])
    assert lm["text"] == ""


def test_grammar_plus_fstring():
    @guidance(stateless=True, dedent=False)
    def test(lm):
        val = 4
        lm += f"the value of {val} is best! {gen(max_tokens=1)}"
        return lm

    lm = models.Mock()
    lm += test()
    assert "{{G|" not in str(lm)


class TestRecursion:
    def test_simple_recursion(self):
        @guidance(stateless=True, dedent=False)
        def grammar(lm):
            return lm + "x" + optional(grammar())

        grammar()

    def test_mutual_recursion(self):
        @guidance(stateless=True, dedent=False)
        def grammar1(lm):
            return lm + "x" + grammar2()

        @guidance(stateless=True, dedent=False)
        def grammar2(lm):
            return lm + "y" + optional(grammar1())

        grammar1()
        grammar2()

    def test_multiple_mutual_recursion(self):
        @guidance(stateless=True, dedent=False)
        def grammar1(lm):
            return lm + "x" + grammar2()

        @guidance(stateless=True, dedent=False)
        def grammar2(lm):
            return lm + "y" + grammar3()

        @guidance(stateless=True, dedent=False)
        def grammar3(lm):
            return lm + "z" + optional(grammar1())

        grammar1()
        grammar2()
        grammar3()

    def test_branching_mutual_recursion(self):
        @guidance(stateless=True, dedent=False)
        def grammar1(lm):
            return lm + "x" + grammar2()

        @guidance(stateless=True, dedent=False)
        def grammar2(lm):
            return lm + "y" + select([grammar1(), grammar3()])

        @guidance(stateless=True, dedent=False)
        def grammar3(lm):
            return lm + "z" + optional(grammar1())

        grammar1()
        grammar2()
        grammar3()


class TestMatch:
    @pytest.mark.parametrize("string", ["456", "456x"])
    def test_full_match(self, string):
        g = "123" + gen(regex=r"\d+x?", name="mycap")
        match = g.match(f"123{string}")
        assert match is not None
        assert not match.partial
        assert match.captures["mycap"] == string

    @pytest.mark.parametrize(
        "string",
        # "456" fails -- think about supporting?
        # (reasonable to expect either behavior)
        ["456x"],
    )
    def test_partial_match(self, string):
        g = "123" + gen(regex=r"\d+x?", name="mycap") + "789"
        assert g.match(f"123{string}") is None
        match = g.match(f"123{string}", allow_partial=True)
        assert match is not None
        assert match.partial
        assert match.captures["mycap"] == string

    def test_raises_on_incomplete_input(self):
        g = "123" + gen(regex=r"\d+x?", name="mycap")
        # Ok since we allow partial
        assert g.match(b"123", raise_exceptions=True, allow_partial=True) is not None
        # Shold raise since we don't allow partial
        with pytest.raises(ByteParserException):
            g.match(b"123", raise_exceptions=True)
