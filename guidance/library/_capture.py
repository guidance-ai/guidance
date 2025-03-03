from .._guidance import guidance
from .._grammar import capture as grammar_capture, GrammarNode
from ._block import block

@guidance(stateless=lambda *args, **kwargs: isinstance(args[0], GrammarNode))
def capture(lm, value, name):
    if isinstance(value, GrammarNode):
        return lm + grammar_capture(value, name)
    else:
        with block(name):
            lm += value
        return lm
