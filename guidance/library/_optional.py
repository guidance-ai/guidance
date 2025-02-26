from .._guidance import guidance
from .._grammar import repeat

@guidance(stateless=True)
def optional(lm, value):
    return lm + repeat(value, 0, 1)
