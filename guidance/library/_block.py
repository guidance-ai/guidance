from guidance import models
from guidance.models._base._model import Block

def block(name=None, opener=None, closer=None):
    return models._base._model.Block(name, opener, closer)
