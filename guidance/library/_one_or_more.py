from ._select import select

def one_or_more(value):
    return select([value], recurse=True)