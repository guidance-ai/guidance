def hide(value):
    _rec_hide(value)
    return value

def _rec_hide(grammar):
    grammar.hidden = True
    if hasattr(grammar, "values"):
        for g in grammar.values:
            _rec_hide(g)