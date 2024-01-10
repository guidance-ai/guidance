import guidance
from ._block import block

@guidance
def set_attr_opener(lm, name, value):
    if hasattr(lm, name):
        lm = lm.setattr("__save" + name, getattr(lm, name))
    return lm.setattr(name, value)

@guidance
def set_attr_closer(lm, name):
    if hasattr(lm, "__save" + name):
        return lm.setattr(name, lm["__save" + name]).delattr("__save" + name)
    else:
        return lm.delattr(name)

def set_attribute(name, value=True):
    return block(
        opener=set_attr_opener(name, value),
        closer=set_attr_closer(name),
    )