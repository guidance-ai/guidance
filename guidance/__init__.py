__version__ = "0.0.64"

import nest_asyncio
nest_asyncio.apply()
import types
import sys
import os
import requests
from . import template_commands as commands
from ._program import Program
from . import endpoints
llms = endpoints # backwards compatibility
from . import models

from ._utils import load, chain, Silent, CaptureEvents, InPlace, TextRange
from . import _utils
from . import selectors
import asyncio
import threading
import functools
import queue

# the user needs to set an LLM before they can use guidance
llm = None

# This makes the guidance module callable
class Guidance(types.ModuleType):
    def __call__(self, template=None, llm=None, cache_seed=0, logprobs=None, silent=None, async_mode=False, stream=None, caching=None, await_missing=False, logging=False, **kwargs):
        if callable(template) or template is None:
            return _decorator(template, model=kwargs.get("model", None))
        else:
            return Program(template, llm=llm, cache_seed=cache_seed, logprobs=logprobs, silent=silent, async_mode=async_mode, stream=stream, caching=caching, await_missing=await_missing, logging=logging, **kwargs)
sys.modules[__name__].__class__ = Guidance

def _decorator(f, *, model=None):

    def _decorator_inner(f, model=models.LM):
        """Decorator to turn a normal function into a guidance function.
        
        Guidance functions have the added ability to be called as methods of LM objects (for dot-chaining),
        and to be optionally iterated over to get a stream of results (syncronously or asyncronously).
        TODO: In the future we plan to add network aware guidance acceleration as well.
        """
        
        def sync_wrapper(lm, *args, silent=None, **kwargs):
            with Silent(lm, silent):
                return f(lm, *args, **kwargs)

        def sync_iter_wrapper(lm, *args, silent=None, **kwargs):

            # create a worker thread and run the function in it
            with Silent(lm, silent):
                with CaptureEvents(lm) as events:
                    worker_thread = threading.Thread(target=f, args=(lm, *args), kwargs=kwargs)
                    worker_thread.start()
                
                    # loop over the queue and display the results
                    while True:
                        try:
                            val = events.get(timeout=0.1)
                            yield val
                        except queue.Empty:
                            if not worker_thread.is_alive():
                                break

        async def async_wrapper(lm, *args, silent=None, **kwargs):
            with Silent(lm, silent):
                return await f(lm, *args, **kwargs)

        async def async_iter_wrapper(lm, *args, **kwargs):
            iterator = _utils.ThreadSafeAsyncIterator(sync_iter_wrapper(lm, *args, **kwargs))
            async for item in iterator:
                yield item

        @functools.wraps(f)
        def wrapper(lm, *args, stream=False, async_mode=False, **kwargs):
            if async_mode:
                if stream:
                    return async_iter_wrapper(lm, *args, **kwargs)
                else:
                    return async_wrapper(lm, *args, **kwargs)
            else:
                if stream:
                    return sync_iter_wrapper(lm, *args, **kwargs)
                else:
                    return sync_wrapper(lm, *args, **kwargs)

        setattr(model, f.__name__, wrapper)

        return wrapper

    if model is None:
        return _decorator_inner(f)
    else:
        return functools.partial(_decorator_inner, model=model)

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

from . import library