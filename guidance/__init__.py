__version__ = "0.2.5"

import sys
from types import ModuleType

from . import library, models, types
from ._guidance import guidance

# we expose all the library functions at the top level of the module
from .library import *  # noqa: F403

__all__ = [
    "guidance",
    "models",
    "types",
    "library",
    *library.__all__,
]


# This makes the guidance module callable
class _Guidance(ModuleType):
    def __call__(self, f=None, *, stateless=False, cache=None, dedent=True, model=models.Model):
        return guidance(f, stateless=stateless, cache=cache, dedent=dedent, model=model)


sys.modules[__name__].__class__ = _Guidance
