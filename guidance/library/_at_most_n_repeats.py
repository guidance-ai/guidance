from .._guidance import guidance
from ._exactly_n_repeats import exactly_n_repeats
from ._optional import optional


@guidance(stateless=True)
def at_most_n_repeats(model, value, n_repeats: int):
    assert n_repeats >= 0

    return model + exactly_n_repeats(optional(value), n_repeats)
