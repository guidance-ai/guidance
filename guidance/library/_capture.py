from .._guidance import guidance
from .._grammar import capture as grammar_capture, GrammarFunction


@guidance(stateless=lambda *args, **kwargs: isinstance(args[0], GrammarFunction))
def capture(lm, value, name):
    if isinstance(value, GrammarFunction):
        return lm + grammar_capture(value, name)
    else:
        start_len = len(lm)
        lm += value
        return lm.set(name, str(lm)[start_len:])
