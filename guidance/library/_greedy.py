from .._grammar import NestedGrammar, GenGrammar, GenLexeme, GrammarFunction, capture
from typing import Optional


def lexeme(
    body_regex: str,
    allow_others: bool = False,
):
    return GenLexeme(body_regex=body_regex, allow_others=allow_others)


def greedy_grammar(
    body: GrammarFunction,
    skip_regex: Optional[str] = None,
):
    return NestedGrammar(body=body, greedy_lexer=True, greedy_skip_regex=skip_regex)


def lazy_grammar(
    body: GrammarFunction,
):
    return NestedGrammar(body=body, greedy_lexer=False, greedy_skip_regex=None)


def gen_grammar(
    name: str,
    body: NestedGrammar,
    stop_regex: Optional[str] = None,
    no_initial_skip: bool = False,
    max_tokens=100000000,
):
    r = GenGrammar(
        grammar=body,
        stop_regex=stop_regex,
        no_initial_skip=no_initial_skip,
        max_tokens=max_tokens,
    )
    if name:
        r = capture(r, name)
    return r
