from guidance import models

class ContextBlock:
    def __init__(self, opener, closer):
        self.opener = opener
        self.closer = closer

    def __enter__(self):
        models.Model.open_blocks[self] = None
    
    def __exit__(self, exc_type, exc_value, traceback):
        del models.Model.open_blocks[self]

def block(name=None, opener="", closer="", hidden=False):
    assert name is None
    assert hidden is False
    return ContextBlock(opener, closer)