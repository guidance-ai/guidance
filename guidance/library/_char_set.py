from .._grammar import select
from ._char_range import char_range

def char_set(def_string):
    parts = []
    pos = 0
    while pos < len(def_string):
        if pos + 2 < len(def_string) and def_string[pos + 1] == "-":
            parts.append(char_range(def_string[pos], def_string[pos + 2]))
            pos += 3
        elif pos + 1 < len(def_string) and def_string[pos] == "\\":
            parts.append(def_string[pos + 1])
            pos += 2
        else:
            parts.append(def_string[pos])
            pos += 1
    return select(parts)