import types

import guidance
from guidance import TextRange

@guidance
def block(self, name=None, open_text="", close_text=""):
    offset = len(self) + len(open_text)

    def __enter__(self):
        return self.append(open_text)
    
    def __exit__(self, exc_type, exc_value, traceback):
        _rec_close(self, close_text, text_name=name, text_offset=offset)
    
    # bind the enter and exit methods
    self.instance__enter__.append(types.MethodType(__enter__, self))
    self.instance__exit__.append(types.MethodType(__exit__, self))

    return self

def _rec_close(lm, close_string, text_name=None, text_offset=0):
    if text_name is not None:
        lm[text_name] = TextRange(text_offset, len(lm), lm)
    if close_string != "":
        lm += close_string
    
    for child in lm._children:
        _rec_close(child, close_string, text_name=text_name, text_offset=text_offset)