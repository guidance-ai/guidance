__version__ = "0.2.3"

import sys
import types

from . import models
from ._ast import Function, GrammarNode
from ._guidance import guidance
from ._utils import strip_multiline_string_indents


# This makes the guidance module callable
class _Guidance(types.ModuleType):
    def __call__(self, f=None, *, stateless=False, cache=None, dedent=True, model=models.Model):
        return guidance(f, stateless=stateless, cache=cache, dedent=dedent, model=model)


sys.modules[__name__].__class__ = _Guidance

# we expose all the library functions at the top level of the module
# widget debug utilities
from .debug import clear_widget_debug, dump_widget_debug, enable_widget_debug, widget_debug_info
from .library import *
