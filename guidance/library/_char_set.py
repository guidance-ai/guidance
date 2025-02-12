import re
from .._grammar import regex


def char_set(def_string: str):
    # TODO: probably should deprecate this function
    if not re.match(r"(?:\\.|[^\-\\])(?:-(?:\\.|[^\-\\]))?", def_string):
        raise ValueError("Invalid character set definition (did you want a general regex instead?)")
    return regex(f"[{def_string}]")
