from typing import Union

from .._grammar import repeat
from .._guidance import guidance


@guidance(stateless=True)
def exactly_n_repeats(model, value, n_repeats: int):
    return model + repeat(value, min=n_repeats, max=n_repeats)


@guidance(stateless=True)
def at_most_n_repeats(model, value, n_repeats: int):
    return model + repeat(value, min=0, max=n_repeats)


@guidance(stateless=True)
def sequence(model, value, min_length: int = 0, max_length: Union[int, None] = None):
    # Just an alias for repeat for now -- TODO: remove?
    return model + repeat(value, min=min_length, max=max_length)


@guidance(stateless=True)
def one_or_more(model, value):
    return model + repeat(value, min=1)


@guidance(stateless=True)
def zero_or_more(model, value):
    return model + repeat(value, min=0)
