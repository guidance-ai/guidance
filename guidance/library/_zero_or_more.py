from ._select import select

def zero_or_more(value):
    return select([value, ""], recurse=True)