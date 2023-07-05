import types

import guidance
from guidance import TextRange

@guidance
def block(self, name=None, open_text="", close_text=""):
    new_lm = self._clone()
    offset = len(self) + len(open_text)

    def __enter__(self):
        return self + open_text
    
    def __exit__(self, exc_type, exc_value, traceback):
        _rec_close(self, close_text, text_name=name, text_offset=offset)
    
    # bind the enter and exit methods
    new_lm.instance__enter__ = types.MethodType(__enter__, new_lm)
    new_lm.instance__exit__ = types.MethodType(__exit__, new_lm)

    return new_lm

def _rec_close(lm, close_string, text_name=None, text_offset=0):
    for child in lm._children:
        if text_name is not None:
            child[text_name] = TextRange(text_offset, len(child), child)
        if close_string != "":
            child._inplace_append(close_string, force_silent=len(child._children) > 1) # use hidden method for speed over (InPlace)
        _rec_close(child, close_string, text_name=text_name, text_offset=text_offset)