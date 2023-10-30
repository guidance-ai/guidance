import guidance
from ._select import select

@guidance(stateless=True)
def optional(lm, value):
    return lm + select([value, ""])