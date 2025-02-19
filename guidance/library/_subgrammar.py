from .._ast import GrammarNode, RuleNode
from .._grammar import subgrammar, regex

__all__ = ["subgrammar", "regex", "as_regular_grammar", "lexeme"]

def as_regular_grammar(node: GrammarNode, lexeme=False):
    # TODO: Remove this assertion-only check?
    if isinstance(node, RuleNode):
        rule = node
    else:
        rule = RuleNode("dummy", node)
    assert rule.is_terminal
    return node

def lexeme(body_regex: str, contextual: bool = False, json_string: bool = False):
    if json_string:
        raise NotImplementedError("JSON strings are not supported")
    if contextual:
        raise NotImplementedError("Contextual lexemes are not supported")
    return regex(body_regex)
