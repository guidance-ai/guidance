import pytest

from guidance._ast import GrammarNode, RuleNode
from guidance.library._subgrammar import lexeme, subgrammar


def _collect_temperatures(node: GrammarNode) -> set[float]:
    """Walk a grammar tree and collect every non-None RuleNode.temperature."""
    found: set[float] = set()
    seen: set[int] = set()

    def _walk(n: GrammarNode) -> None:
        if not isinstance(n, GrammarNode) or id(n) in seen:
            return
        seen.add(id(n))
        if isinstance(n, RuleNode) and n.temperature is not None:
            found.add(n.temperature)
        for c in n.children():
            _walk(c)

    _walk(node)
    return found


class TestSubgrammarFactoryTemperature:
    """Regression tests for silently-dropped ``temperature=0.0`` in
    ``guidance._grammar.subgrammar``.

    ``temperature=0.0`` is the canonical way to request deterministic / greedy
    sampling and must not be dropped by a ``if temperature:`` truthiness check.
    """

    def test_temperature_zero_is_preserved(self):
        node = subgrammar(lexeme(r"\d+"), temperature=0.0)
        observed = _collect_temperatures(node)
        assert 0.0 in observed, f"temperature=0.0 was dropped by subgrammar(); observed={observed}"

    def test_temperature_nonzero_still_works(self):
        node = subgrammar(lexeme(r"\d+"), temperature=0.5)
        assert 0.5 in _collect_temperatures(node)

    def test_temperature_none_sets_nothing(self):
        node = subgrammar(lexeme(r"\d+"))
        assert _collect_temperatures(node) == set()


def _collect_max_tokens(node: GrammarNode) -> set[int]:
    """Walk a grammar tree and collect every non-None RuleNode.max_tokens."""
    found: set[int] = set()
    seen: set[int] = set()

    def _walk(n: GrammarNode) -> None:
        if not isinstance(n, GrammarNode) or id(n) in seen:
            return
        seen.add(id(n))
        if isinstance(n, RuleNode) and n.max_tokens is not None:
            found.add(n.max_tokens)
        for c in n.children():
            _walk(c)

    _walk(node)
    return found


class TestSubgrammarFactoryMaxTokens:
    """Regression tests for silently-dropped ``max_tokens=0`` in
    ``guidance._grammar.subgrammar``.

    Same falsy-truthiness class of bug as ``temperature=0.0``: ``if max_tokens:``
    dropped the valid value ``0``. Guarding the two limits together prevents
    a one-line regression to either from slipping past the suite.
    """

    def test_max_tokens_zero_is_preserved(self):
        node = subgrammar(lexeme(r"\d+"), max_tokens=0)
        observed = _collect_max_tokens(node)
        assert 0 in observed, f"max_tokens=0 was dropped by subgrammar(); observed={observed}"

    def test_max_tokens_nonzero_still_works(self):
        node = subgrammar(lexeme(r"\d+"), max_tokens=16)
        assert 16 in _collect_max_tokens(node)

    def test_max_tokens_none_sets_nothing(self):
        node = subgrammar(lexeme(r"\d+"))
        assert _collect_max_tokens(node) == set()


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
