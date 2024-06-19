from .._guidance import guidance
from .._grammar import Byte, byte_range, select


@guidance(stateless=True)
def any_char_but(lm, forbidden):
    """Allows any char except those in forbidden"""
    # TODO: extend this to support utf-8 encoded multibyte unicode characters
    forb = sorted(set(ord(x) for x in forbidden))
    # Add end value to simplify loop (assumes 0-127 ascii)
    forb.append(128)
    start = 0
    ranges = []
    singletons = []
    for i in forb:
        if start < i - 1:
            # Add range if there is more than one character
            ranges.append((start, i - 1))
        if start == i - 1:
            # Add singleton if there is only one character
            singletons.append(start)
        start = i + 1
    ranges = [byte_range(bytes([i]), bytes([j])) for i, j in ranges]
    singletons = [Byte(bytes([i])) for i in singletons]
    return select(ranges + singletons)
