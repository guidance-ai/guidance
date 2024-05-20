from .._guidance import guidance
from ..models._model import Model
from ._block import block


@guidance
def set_attr_opener(lm, name, value, cache):
    if hasattr(lm, name):
        cache[name] = getattr(lm, name)
    return lm.setattr(name, value)


@guidance
def set_attr_closer(lm, name, cache):
    if not Model._grammar_only:
        if name in cache:
            return lm.setattr(name, cache[name])
        else:
            return lm.delattr(name)
    else:
        return lm


def set_attribute(name, value=True):
    cache = {}
    return block(
        opener=set_attr_opener(name, value, cache),
        closer=set_attr_closer(name, cache),
    )
