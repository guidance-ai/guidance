__version__ = "0.1.3"

import nest_asyncio
nest_asyncio.apply()
import types
import sys
import os
import requests
from . import models
import inspect

from ._utils import load, chain, CaptureEvents, TextRange, strip_multiline_string_indents
from . import _utils
from . import selectors

from ._grammar import StatelessFunction, StatefulFunction, string, Terminal, Placeholder, replace_grammar_node
import functools
from contextlib import nullcontext

curr_module = sys.modules[__name__]

# the user needs to set an LLM before they can use guidance
llm = None

newline = "\n"

# This makes the guidance module callable
class Guidance(types.ModuleType):
    def __call__(self, f=None, *, stateless=False, cache=None, dedent=True, model=models.Model):
        return _decorator(f, stateless=stateless, cache=cache, dedent=dedent, model=model)
sys.modules[__name__].__class__ = Guidance

# def optional_hidden(f, lm, hidden, kwargs):
#     """This only enters a hidden context if the function does not manage the hidden parameter itself.
#     """
#     if 'hidden' in inspect.signature(f).parameters:
#         kwargs['hidden'] = hidden
#         return nullcontext()
#     else:
#         return Hidden(lm, hidden)
    
_function_cache = {} # used to enable recursive grammar definitions
_null_grammar = string('')
# TODO: enable streaming for guidance function evaluation

def _decorator(f, *, stateless, cache, dedent, model):
    
    # if we are not yet being used as a decorator, then save the args
    if f is None:
        return functools.partial(_decorator, stateless=stateless, cache=cache, dedent=dedent, model=model)
    
    # if we are being used as a decorator then return the decorated function
    else:

        # this strips out indentation in multiline strings that aligns with the current python indentation
        if dedent is True or dedent == 'python':
            f = strip_multiline_string_indents(f)

        # we cache if requested
        if cache:
            f = functools.cache(f)

        @functools.wraps(f)
        def wrapped(*args, **kwargs):

            # make a stateless grammar if we can
            if stateless is True or (callable(stateless) and stateless(*args, **kwargs)):
                
                # if we have a placeholder set then we must be in a recursive definition and so we return the placeholder
                placeholder = getattr(f, "_self_call_placeholder_", None)
                if placeholder is not None:
                    return placeholder
                
                # otherwise we call the function to generate the grammar
                else:
                    
                    # set a placeholder for recursive calls (only if we don't have arguments that might make caching a bad idea)
                    no_args = len(args) + len(kwargs) == 0
                    if no_args:
                        f._self_call_placeholder_ = Placeholder()

                    # call the function to get the grammar node
                    node = f(_null_grammar, *args, **kwargs)
                    if not isinstance(node, Terminal):
                        node.name = f.__name__

                    # replace all the placeholders with our generated node
                    if no_args:
                        replace_grammar_node(node, f._self_call_placeholder_, node)
                        del f._self_call_placeholder_

                    return node

            # otherwise must be stateful (which means we can't be inside a select() call)
            else:
                return StatefulFunction(f, args, kwargs)
        
        # attach this as a method of the model class (if given)
        if model is not None:
            setattr(model, f.__name__, f)
        
        return wrapped

# def _decorator(f, *, model=None, dedent='python'):

#     def _decorator_inner(f, model=models.Model):
#         """Decorator to turn a normal function into a guidance function.
        
#         Guidance functions have the added ability to be called as methods of LM objects (for dot-chaining),
#         and to be optionally iterated over to get a stream of results (syncronously or asyncronously).
#         TODO: In the future we plan to add network aware guidance acceleration as well.
#         """

#         # this strips out indentation in multiline strings that aligns with the current python indentation
#         if dedent == 'python':
#             f = strip_multiline_string_indents(f)
        
#         def sync_wrapper(lm, *args, silent=None, hidden=False, **kwargs):
#             with Silent(lm, silent), optional_hidden(f, lm, hidden, kwargs):
#                 return f(lm, *args, **kwargs)

#         def sync_iter_wrapper(lm, *args, silent=None, hidden=False, **kwargs):

#             # create a worker thread and run the function in it
#             with Silent(lm, silent), optional_hidden(f, lm, hidden, kwargs):
#                 with CaptureEvents(lm) as events:
#                     worker_thread = threading.Thread(target=f, args=(lm, *args), kwargs=kwargs)
#                     worker_thread.start()
                
#                     # loop over the queue and display the results
#                     while True:
#                         try:
#                             val = events.get(timeout=0.1)
#                             yield val
#                         except queue.Empty:
#                             if not worker_thread.is_alive():
#                                 break

#         async def async_wrapper(lm, *args, silent=None, hidden=False, **kwargs):
#             with Silent(lm, silent), optional_hidden(f, lm, hidden, kwargs):
#                 return await f(lm, *args, **kwargs)

#         async def async_iter_wrapper(lm, *args, **kwargs):
#             iterator = _utils.ThreadSafeAsyncIterator(sync_iter_wrapper(lm, *args, **kwargs))
#             async for item in iterator:
#                 yield item

#         @functools.wraps(f)
#         def wrapper(*args, stream=False, async_mode=False, **kwargs):

#             # check if we are making a delayed call
#             if len(args) == 0 or not isinstance(args[0], models.Model):
#                 return wrapper.wrapper_delayed(*args, stream=stream, async_mode=async_mode, **kwargs)
#             else:
#                 lm = args[0]
#                 args = args[1:]

#             # if not we execute
#             if async_mode:
#                 if stream:
#                     return async_iter_wrapper(lm, *args, **kwargs)
#                 else:
#                     return async_wrapper(lm, *args, **kwargs)
#             else:
#                 if stream:
#                     return sync_iter_wrapper(lm, *args, **kwargs)
#                 else:
#                     return sync_wrapper(lm, *args, **kwargs)
                    
#         @functools.wraps(f)
#         def wrapper_delayed(*args, stream=False, async_mode=False, **kwargs):
#             '''Converts a guidance function call to a string, so it can be called later once it is added to an LM object.'''
            
#             # save the call in our call pool, ready to be run when it is attached to an LM object
#             id = str(uuid.uuid4())
#             models.Model._call_pool[id] = lambda lm: wrapper(lm, *args, stream=stream, async_mode=async_mode, **kwargs)

#             # return a string representation of this call so it can be combined with other strings/calls
#             return models.Model.tag_start + id + models.Model.tag_end
#         wrapper.wrapper_delayed = wrapper_delayed

#         setattr(model, f.__name__, wrapper) # as a method on the LM object
#         setattr(curr_module, f.__name__, wrapper_delayed) # as a top level class

#         return wrapper

#     if model is None:
#         return _decorator_inner(f)
#     else:
#         return functools.partial(_decorator_inner, model=model)

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

# we expose all the library functions at the top level of the module
from .library import *