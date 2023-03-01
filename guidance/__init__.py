__version__ = "0.0.22"

import types
import sys
from ._prompt import Prompt
from . import endpoints
from . import library

endpoint = endpoints.OpenAI()

# def __call__(self, template, call_function=None, endpoint=None, echo=False, cache_seed=0, logprobs=None):
#     return Prompt(template, call_function=call_function, endpoint=endpoint, echo=echo, cache_seed=cache_seed, logprobs=logprobs)

# This is makes the guidance module callable
class Guidance(types.ModuleType):
    def __call__(self, template, call_function=None, endpoint=None, echo=False, cache_seed=0, logprobs=None):
        return Prompt(template, call_function=call_function, endpoint=endpoint, echo=echo, cache_seed=cache_seed, logprobs=logprobs)
sys.modules[__name__].__class__ = Guidance