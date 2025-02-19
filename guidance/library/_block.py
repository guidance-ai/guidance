from guidance.models import Model


class ContextBlock:
    def __init__(self, opener, closer, name=None):
        self.opener = opener
        self.closer = closer
        self.name = name

    def __enter__(self):
        Model.Model.global_active_blocks.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        Model.Model.global_active_blocks.remove(self)


def block(name=None, opener="", closer=""):
    return ContextBlock(opener, closer, name=name)
