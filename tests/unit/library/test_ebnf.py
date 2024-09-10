from typing import Any

import pytest

from guidance import ebnf
from guidance._grammar import Join, Select


class TestIntegerArithmetic:
    start = "expr"
    grammar_def = """
    expr    : expr "+" term     -> add
            | expr "-" term     -> sub
            | term

    term    : term "*" factor   -> mul
            | term "/" factor   -> div
            | factor

    factor  : integer
            | "(" expr ")"

    integer : DIGIT+
            | "-" integer       -> neg

    %import common.DIGIT
    """
    grammar = ebnf(grammar=grammar_def, start=start)

    def test_no_redundant_nonterminals(self):
        # Accumulate a set of all nonterminal nodes in the grammar
        seen = set()

        def accumulate(g: Any):
            if g in seen or not isinstance(g, (Join, Select)):
                return
            seen.add(g)
            for v in g.values:
                accumulate(v)

        accumulate(self.grammar)

        # Magic number 14 is minimal number of nodes (derived "by inspection")
        assert len(seen) == 14
        assert len({s.name for s in seen}) == 14

    @pytest.mark.parametrize(
        "matchstr", ["1+2+3+4", "1/2+3/4", "(1+2/3)*(4+(5+3/2))", "8/-4", "-9", "42"]
    )
    def test_good(self, matchstr):
        assert self.grammar.match(matchstr) is not None

    @pytest.mark.parametrize("matchstr", ["2+3+4(5)", "7(+8)", "8/*6"])
    def test_bad(self, matchstr):
        assert self.grammar.match(matchstr) is None
