from guidance import models

class ContextBlock:
    def __init__(self, opener, closer, name=None):
        self.opener = opener
        self.closer = closer
        self.name = name

    def __enter__(self):
        models.Model.open_blocks[self] = None
    
    def __exit__(self, exc_type, exc_value, traceback):
        del models.Model.open_blocks[self]

def block(name=None, opener="", closer=""):
    return ContextBlock(opener, closer, name=name)