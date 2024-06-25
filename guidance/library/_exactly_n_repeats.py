from .._guidance import guidance
from .._grammar import Join


@guidance(stateless=True)
def exactly_n_repeats(model, value, n_repeats: int):
    assert n_repeats >= 0, f"Must have n_repeats>=0 ({n_repeats})"
    return model + Join([value] * n_repeats)
