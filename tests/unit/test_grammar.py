import pytest

import guidance
from guidance import gen, lark, models, optional, select, with_temperature
from guidance._parser import ByteParserException
from guidance.library._subgrammar import subgrammar


def test_select_reset_pos():
    model = models.Mock()
    model += "This is" + select(options=["bad", "quite bad"])
    assert str(model) in ["This isbad", "This isquite bad"]


def test_select_longer():
    """This tests to ensure that the grammar is extended greedily."""
    lm = models.Mock(b"<s>Scott is a very nice man.")
    lm += "Scott is a very " + select(name="text", options=["nice", "nice man."])
    assert lm["text"] == "nice man."


@pytest.mark.xfail(
    reason="Lexer sees 'a' then 'b' and here decides to continue matching abq; see https://github.com/guidance-ai/llguidance/issues/2"
)
def test_select_ambiguous_lexeme_boundary():
    lm = models.Mock(b"<s>abQ<s>")
    lm += lark("""\
start: prefix "bQ"?
prefix[capture]: "a" | "abq" | "c"
""")
    assert lm["prefix"] == "a"


def test_select_ambiguous_lexeme_boundary_manual_fix():
    # Manual fix to the issue in test_select_ambiguous_lexeme_boundary by splitting the "abq" lexeme into two lexemes
    lm = models.Mock(b"<s>abQ<s>")
    lm += lark("""\
start: prefix "bQ"?
prefix[capture]: "a" | "a" "bq" | "c"
""")
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


@pytest.mark.parametrize("inner_temperature", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("outer_temperature", [0.0, 0.9])
@pytest.mark.parametrize("use_subgrammar", [False, True])
def test_with_temperature_preserves_explicit_temperature(inner_temperature, outer_temperature, use_subgrammar):
    node = gen(name="answer", regex="[ab]{3}", max_tokens=3, stop=";", save_stop_text="stop")
    if use_subgrammar:
        node = subgrammar(node)
    original = with_temperature(node, inner_temperature)

    wrapped = with_temperature(original, outer_temperature)

    assert wrapped == original
    assert original.temperature == inner_temperature
    assert f"temperature={inner_temperature}" in wrapped.ll_grammar()


@pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("use_subgrammar", [False, True])
def test_with_temperature_sets_unset_temperature(temperature, use_subgrammar):
    original = gen(name="answer", regex="[ab]{3}", max_tokens=3)
    if use_subgrammar:
        original = subgrammar(original)

    wrapped = with_temperature(original, temperature)

    assert wrapped.temperature == temperature
    assert original.temperature is None
    lm = models.Mock(b"<s>aaa") + wrapped
    assert lm["answer"] == "aaa"
    assert lm.engine.called_temperatures
    assert all(value == temperature for value in lm.engine.called_temperatures)


@pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("use_subgrammar", [False, True])
def test_with_temperature_preserves_sampling_temperature(temperature, use_subgrammar):
    node = gen(name="answer", regex="[ab]+", max_tokens=10, stop=";", save_stop_text="stop")
    if use_subgrammar:
        node = subgrammar(node)
    original = with_temperature(node, temperature)

    lm = models.Mock(b"<s>aaa;") + with_temperature(original, 0.9)

    assert lm["answer"] == "aaa"
    assert lm["stop"] == ";"
    assert lm.engine.called_temperatures
    assert all(value == temperature for value in lm.engine.called_temperatures)


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
