from ._string import string

def commit_point(value):
    if isinstance(value, str):
        value = string(value)
    value.commit_point = True
    return value