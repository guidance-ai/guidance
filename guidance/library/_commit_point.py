from ._string import string

def commit_point(value, hidden=False):
    # TODO: this should not modify things inplace
    # TODO: assert that value is not empty since we don't yet support that
    if isinstance(value, str):
        value = string(value)
    value.commit_point = True
    if hidden:
        _rec_hide(value)
    return value

def _rec_hide(grammar):
    if not grammar.hidden:
        grammar.hidden = True
        if hasattr(grammar, "values"):
            for g in grammar.values:
                _rec_hide(g)