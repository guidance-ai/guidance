from .._grammar import byte_range
    
def char_range(low, high):
    low_bytes = bytes(low, encoding="utf8")
    high_bytes = bytes(high, encoding="utf8")
    if len(low_bytes) > 1 or len(high_bytes) > 1:
        raise Exception("We don't yet support multi-byte character ranges!")
    return byte_range(low_bytes, high_bytes)