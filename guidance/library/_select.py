from .._grammar import Select, Grammar, _find_name

def select(values, name=None, recurse=False):
    if name is None:
        name = _find_name() + "_" + Grammar._new_name()
    if recurse:
        node = Select([], name)
        node.values = [v + node for v in values if v != ""] + values
        return node
    else:
        if len(values) == 1 and name is None:
            return values[0]
        else:
            return Select(values, name)