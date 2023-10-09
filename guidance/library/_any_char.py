from guidance import ByteRange

def any_char():
    return ByteRange(bytes([0,255]))