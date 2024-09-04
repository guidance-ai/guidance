from .._grammar import Subgrammar, Lexeme, GrammarFunction, capture
from typing import Optional


def lexeme(
    body_regex: str,
    contextual: bool = False,
    json_string: bool = False,
) -> Lexeme:
    """
    Constructs a Lexeme based on a given regular expression.

    Parameters:
    body_regex (str): The regular expression that will greedily match the input.
    contextual (bool): If false, all other lexemes are excluded when this lexeme is recognized.
        This is normal behavior for keywords in programming languages.
        Set to true for eg. a JSON schema with both `/"type"/` and `/"[^"]*"/` as lexemes,
        or for "get"/"set" contextual keywords in C#.
    json_string (bool): Specifies if the lexeme should be quoted as a JSON string.
        For example, /[a-z"]+/ will be quoted as /([a-z]|\\")+/.
        Defaults to False.
    """
    return Lexeme(body_regex=body_regex, contextual=contextual, json_string=json_string)


def subgrammar(
    name: str = None,
    *,
    body: GrammarFunction,
    skip_regex: Optional[str] = None,
    no_initial_skip: bool = False,
    max_tokens=100000000,
) -> GrammarFunction:
    r: GrammarFunction = Subgrammar(
        body=body,
        skip_regex=skip_regex,
        no_initial_skip=no_initial_skip,
        max_tokens=max_tokens,
    )
    if name:
        r = capture(r, name)
    return r
