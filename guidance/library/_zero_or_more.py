import guidance
from ._select import select

@guidance(stateless=True)
def zero_or_more(model, value):
    return model + select([value, ""], recurse=True)