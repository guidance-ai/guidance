from typing import Union

from .._grammar import Join, select
from .._guidance import guidance
from ._optional import optional


@guidance(stateless=True)
def exactly_n_repeats(model, value, n_repeats: int):
    assert n_repeats >= 0, f"Must have n_repeats>=0 ({n_repeats})"
    return model + Join([value] * n_repeats)


@guidance(stateless=True)
def at_most_n_repeats(model, value, n_repeats: int):
    assert n_repeats >= 0

    return model + exactly_n_repeats(optional(value), n_repeats)


@guidance(stateless=True)
def sequence(
    model, value, min_length: Union[int, None] = None, max_length: Union[int, None] = None
):
    assert min_length is None or min_length >= 0
    assert max_length is None or max_length >= 0

    if min_length is not None and max_length is not None:
        assert max_length >= min_length
        model += exactly_n_repeats(value=value, n_repeats=min_length)
        model += at_most_n_repeats(value=value, n_repeats=(max_length - min_length))
    elif min_length is not None:
        model += exactly_n_repeats(value=value, n_repeats=min_length)
        model += select([optional(value)], recurse=True)
    elif max_length is not None:
        model += at_most_n_repeats(value=value, n_repeats=max_length)
    else:
        model += select(
            [value],
            recurse=True,
        )
    return model


@guidance(stateless=True)
def one_or_more(model, value):
    return model + select([value], recurse=True)


@guidance(stateless=True)
def zero_or_more(model, value):
    return model + select(["", value], recurse=True)
