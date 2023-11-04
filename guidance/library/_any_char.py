import guidance
from .._grammar import byte_range

@guidance(stateless=True)
def any_char(lm):
    # TODO: extend this to support utf-8 encoded multibyte unicode characters
    return lm + byte_range(b'\x00', b'\xff')