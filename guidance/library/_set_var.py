from .._guidance import guidance
from ._block import block


@guidance
def set_opener(lm, name, value):
    if name in lm:
        lm = lm.set("__save" + name, lm[name])
    return lm.set(name, value)


@guidance
def set_closer(lm, name):
    if "__save" + name in lm:
        return lm.set(name, lm["__save" + name]).remove("__save" + name)
    else:
        return lm.remove(name)


def set_var(name, value=True):
    return block(
        opener=set_opener(name, value),
        closer=set_closer(name),
    )
