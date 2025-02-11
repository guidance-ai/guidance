from .._guidance import guidance
from ..ast import select

@guidance(stateless=True)
def optional(lm, value):
    return lm + select([value, ""])
