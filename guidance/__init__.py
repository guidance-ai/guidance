__version__ = "0.0.57"

import types
import sys
import os
import requests
from ._program import Program
from . import llms
from . import library
from ._utils import load, chain
from . import selectors
import nest_asyncio
import asyncio

# the user needs to set an LLM before they can use guidance
llm = None

# This makes the guidance module callable
class Guidance(types.ModuleType):
    def __call__(self, template, llm=None, cache_seed=0, logprobs=None, silent=None, async_mode=False, stream=None, caching=None, await_missing=False, **kwargs):
        return Program(template, llm=llm, cache_seed=cache_seed, logprobs=logprobs, silent=silent, async_mode=async_mode, stream=stream, caching=caching, await_missing=await_missing, **kwargs)
sys.modules[__name__].__class__ = Guidance


def load(guidance_file):
    ''' Load a guidance program from the given text file.

    If the passed file is a valid local file it will be loaded directly.
    Otherwise, if it starts with "http://" or "https://" it will be loaded
    from the web.
    '''

    if os.path.exists(guidance_file):
        with open(guidance_file, 'r') as f:
            template = f.read()
    elif guidance_file.startswith('http://') or guidance_file.startswith('https://'):
        template = requests.get(guidance_file).text
    else:
        raise ValueError('Invalid guidance file: %s' % guidance_file)
    
    return sys.modules[__name__](template)
