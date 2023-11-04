import guidance
from .._grammar import select

@guidance(stateless=True)
def optional(lm, value):
    return lm + select([value, ""])