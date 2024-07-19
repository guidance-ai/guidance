from .._grammar import Subgrammar, GenLexeme, GrammarFunction, capture
from typing import Optional


def lexeme(
    body_regex: str,
    contextual: bool = False,
):
    return GenLexeme(body_regex=body_regex, contextual=contextual)


def subgrammar(
    name: str = None,
    *,
    body: GrammarFunction,
    skip_regex: Optional[str] = None,
    no_initial_skip: bool = False,
    max_tokens=100000000,
):
    r = Subgrammar(
        body=body,
        skip_regex=skip_regex,
        no_initial_skip=no_initial_skip,
        max_tokens=max_tokens,
    )
    if name:
        r = capture(r, name)
    return r
