from typing import Optional

from llguidance.gbnf_to_lark import gbnf_to_lark

from .._ast import LarkNode
from .._grammar import capture, token_limit, with_temperature


def lark(
    lark_grammar: str,
    *,
    name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
):
    node = LarkNode(
        name=name or "lark",
        lark_grammar=lark_grammar,
    )

    if temperature is not None:
        node = with_temperature(node, temperature)
    if max_tokens is not None:
        node = token_limit(node, max_tokens)
    if name is not None:
        node = capture(node, name)

    return node


def gbnf(
    gbnf_grammar: str,
    *,
    name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
):
    lark_grammar = gbnf_to_lark(gbnf_grammar)
    node = LarkNode(
        name=name or "gbnf",
        lark_grammar=lark_grammar,
    )

    if temperature is not None:
        node = with_temperature(node, temperature)
    if max_tokens is not None:
        node = token_limit(node, max_tokens)
    if name is not None:
        node = capture(node, name)

    return node
