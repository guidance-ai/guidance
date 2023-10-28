import types

import guidance
from guidance import TextRange, models

class ContextBlock:
    def __init__(self, opener, closer):
        self.opener = opener
        self.closer = closer

    def __enter__(self):
        guidance.models.Model._open_contexts[self] = None
    
    def __exit__(self, exc_type, exc_value, traceback):
        del guidance.models.Model._open_contexts[self]

# @guidance(model=models.Model)
def block(name=None, opener="", closer="", hidden=False):
    assert name is None
    assert hidden is False
    return ContextBlock(opener, closer)
    offset = len(lm._state) + len(open_text)

    def __enter__(lm):
        lm._context_head = lm + open_text
        return lm._context_head
    
    def __exit__(lm, exc_type, exc_value, traceback):
        _rec_close(lm._context_head, close_text, hidden, text_name=name, text_offset=offset)
    
    # bind the enter and exit methods
    lm.instance__enter__.append(types.MethodType(__enter__, lm))
    lm.instance__exit__.append(types.MethodType(__exit__, lm))

    return lm

# @guidance
# def hidden(lm):
#     return lm.block(hidden=True)

def _rec_close(lm, close_text, hidden, text_name=None, text_offset=0):
    if text_name is not None:
        lm[text_name] = TextRange(text_offset, len(lm), lm)
    if close_text != "":
        lm._inplace_append(close_text)
    if hidden:
        lm._reset(text_offset, clear_variables=False)
    
    for child in lm._children:
        _rec_close(child, close_text, hidden, text_name=text_name, text_offset=text_offset)