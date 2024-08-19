__version__ = "0.1.16"

import sys
import types

from . import models
from ._guidance import _decorator, guidance

from ._grammar import (
    Placeholder,
    RawFunction,
    GrammarFunction,
    Terminal,
    replace_grammar_node,
    string,
)
from ._utils import strip_multiline_string_indents
from ._server import Server


# This makes the guidance module callable
class _Guidance(types.ModuleType):
    def __call__(
        self, f=None, *, stateless=False, cache=None, dedent=True, model=models.Model
    ):
        return _decorator(
            f, stateless=stateless, cache=cache, dedent=dedent, model=model
        )


sys.modules[__name__].__class__ = _Guidance

# we expose all the library functions at the top level of the module
from .library import *
