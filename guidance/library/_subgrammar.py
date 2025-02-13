from .._ast import GrammarNode
from .._grammar import subgrammar, regex

def as_regular_grammar(node: GrammarNode, lexeme=False):
    raise NotImplementedError("No longer needed?")

def lexeme(body_regex: str, contextual: bool = False, json_string: bool = False):
    if json_string:
        raise NotImplementedError("JSON strings are not supported")
    if contextual:
        raise NotImplementedError("Contextual lexemes are not supported")
    return regex(body_regex)

subgrammar = subgrammar
