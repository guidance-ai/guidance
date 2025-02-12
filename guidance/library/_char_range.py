from .._grammar import regex

def char_range(low: str, high: str):
    if len(low) != 1 or len(high) != 1:
        raise ValueError("char_range only supports single characters")
    if high < low:
        raise ValueError("char_range: high must be greater than low")
    return regex(f"[{low}-{high}]")
