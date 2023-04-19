__version__ = "0.0.29"

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

# allows us to start inner event loops within jupyter notebooks
nest_asyncio.apply()

llm = llms.OpenAI()

# This is makes the guidance module callable
class Guidance(types.ModuleType):
    def __call__(self, template, call_function=None, llm=None, echo=False, cache_seed=0, logprobs=None, **kwargs):
        return Program(template, call_function=call_function, llm=llm, echo=echo, cache_seed=cache_seed, logprobs=logprobs, **kwargs)
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