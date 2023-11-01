import guidance

# This is not something we can support in the parser...see hidden_commit_point for something that works
# @guidance(stateless=True)
# def hidden_commit_point(model, value):
#     _rec_hide(value)
#     return model + value

# def _rec_hide(grammar):
#     grammar.hidden = True
#     if hasattr(grammar, "values"):
#         for g in grammar.values:
#             _rec_hide(g)