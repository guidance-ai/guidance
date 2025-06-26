from .._grammar import repeat
from .._guidance import guidance


@guidance(stateless=True)
def optional(lm, value):
    return lm + repeat(value, 0, 1)
