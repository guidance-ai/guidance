import guidance
from .._grammar import byte_range, select

@guidance(stateless=True)
def any_char_but(lm, forbidden):
    """Allows any char except those in forbidden"""
    # TODO: extend this to support utf-8 encoded multibyte unicode characters
    forb = sorted(set([ord(x) for x in forbidden]))
    start = 0
    ranges = []
    for i in forb:
        if i == 0:
            continue
        newrange = (start, i - 1)
        if newrange[0] < newrange[1]:
            ranges.append(newrange)
        start = i + 1
    if start < 127:
        ranges.append((start, 127))
    ranges = [(i.to_bytes(1, 'big'), j.to_bytes(1, 'big')) for i, j in ranges]
    return select([byte_range(x[0], x[1]) for x in ranges])