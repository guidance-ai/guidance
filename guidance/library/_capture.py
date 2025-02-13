from .._guidance import guidance
from .._grammar import capture as grammar_capture, GrammarNode


@guidance(stateless=lambda *args, **kwargs: isinstance(args[0], GrammarNode))
def capture(lm, value, name):
    if isinstance(value, GrammarNode):
        return lm + grammar_capture(value, name)
    else:
        start_len = len(lm)
        lm += value
        return lm.set(name, str(lm)[start_len:])
