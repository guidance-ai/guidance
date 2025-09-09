__version__ = "0.3.0"

import sys
from types import ModuleType

from . import library, models
from ._guidance import guidance
from ._tools import Tool

# we expose all the library functions at the top level of the module
from .library import *  # noqa: F403

__all__ = [
    "guidance",
    "library",
    "models",
    "Tool",
    *library.__all__,
]


# This makes the guidance module callable
class _Guidance(ModuleType):
    def __call__(self, f=None, *, stateless=False, cache=None, dedent=True, model=models.Model):
        return guidance(f, stateless=stateless, cache=cache, dedent=dedent, model=model)


sys.modules[__name__].__class__ = _Guidance
