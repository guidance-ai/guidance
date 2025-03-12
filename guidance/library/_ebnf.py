from typing import Optional

from llguidance.gbnf_to_lark import gbnf_to_lark as _gbnf_to_lark

from .._ast import LarkNode, GrammarNode
from .._grammar import capture, token_limit, with_temperature


def lark(
    lark_grammar: str,
    *,
    name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> GrammarNode:
    """
    Builds a guidance grammar from (a variant of) the EBNF syntax used by the Lark parsing toolkit.

    See documentation at https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md for more
    details.
    """
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


def gbnf_to_lark(gbnf_grammar: str) -> str:
    """
    Converts a GBNF (llama.cpp) grammar to Lark(-like) syntax. This is a best-effort
    conversion and may not work for all grammars. We recommend using this function
    as a starting point and then manually editing the resulting Lark grammar to suit
    your needs.

    See documentation at https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md
    for more information on the output syntax's semantics.
    """
    return _gbnf_to_lark(gbnf_grammar)
