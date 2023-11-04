import guidance
from .._grammar import select

@guidance(stateless=True)
def one_or_more(model, value):
    return model + select([value], recurse=True)