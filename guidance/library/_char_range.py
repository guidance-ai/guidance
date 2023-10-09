from .._grammar import ByteRange
    
def char_range(low, high):
    low_bytes = bytes(low, encoding="utf8")
    high_bytes = bytes(high, encoding="utf8")
    if len(low_bytes) > 1 or len(high_bytes) > 1:
        raise Exception("We don't yet support multi-byte character ranges!")
    return ByteRange(low_bytes + high_bytes)