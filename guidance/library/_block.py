from guidance import models


class ContextBlock:
    def __init__(self, opener, closer, name=None):
        self.opener = opener
        self.closer = closer
        self.name = name

    def __enter__(self):
        models.Model.global_active_blocks.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        models.Model.global_active_blocks.remove(self)


def block(name=None, opener="", closer=""):
    return ContextBlock(opener, closer, name=name)
