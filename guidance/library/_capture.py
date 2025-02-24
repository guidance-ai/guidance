from .._guidance import guidance
from .._grammar import capture as grammar_capture, GrammarFunction
from ._block import block

@guidance(stateless=lambda *args, **kwargs: isinstance(args[0], GrammarFunction))
def capture(lm, value, name):
    if isinstance(value, GrammarFunction):
        return lm + grammar_capture(value, name)
    else:
        with block(name):
            lm += value
        return lm
