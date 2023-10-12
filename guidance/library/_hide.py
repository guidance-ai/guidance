import guidance

@guidance(stateless=True)
def hide(model, value):
    _rec_hide(value)
    return model + value

def _rec_hide(grammar):
    grammar.hidden = True
    if hasattr(grammar, "values"):
        for g in grammar.values:
            _rec_hide(g)