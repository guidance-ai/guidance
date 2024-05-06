import functools
import inspect

from . import models
from ._grammar import Placeholder, RawFunction, Terminal, replace_grammar_node, string, StatefulException
from ._utils import strip_multiline_string_indents


def guidance(f=None, *, stateless=False, cache=None, dedent=True, model=models.Model):
    return _decorator(f, stateless=stateless, cache=cache, dedent=dedent, model=model)


_null_grammar = string("")


def _decorator(f, *, stateless, cache, dedent, model):

    # if we are not yet being used as a decorator, then save the args
    if f is None:
        return functools.partial(
            _decorator, stateless=stateless, cache=cache, dedent=dedent, model=model
        )

    # if we are being used as a decorator then return the decorated function
    else:

        # this strips out indentation in multiline strings that aligns with the current python indentation
        if dedent is True or dedent == "python":
            f = strip_multiline_string_indents(f)

        # we cache if requested
        if cache:
            f = functools.cache(f)

        @functools.wraps(f)
        def wrapped(*args, **kwargs):

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

                # try to trace the function to get a grammar node
                node = None
                try:
                    node = f(_null_grammar, *args, **kwargs)
                    if not isinstance(node, (Terminal, str)):
                        node.name = f.__name__
                
                # if that fails we must be stateful (which means we can't be inside a select() call)
                except StatefulException:
                    return RawFunction(f, args, kwargs)
                
                # clean up, replacing all the placeholders with our generated node
                finally:
                    if no_args:
                        if node:
                            replace_grammar_node(node, f._self_call_placeholder_, node)
                        del f._self_call_placeholder_

                return node

        # Remove the first argument from the wrapped function
        signature = inspect.signature(f)
        params = list(signature.parameters.values())
        params.pop(0)
        wrapped.__signature__ = signature.replace(parameters=params)

        # attach this as a method of the model class (if given)
        # if model is not None:
        #     setattr(model, f.__name__, f)

        return wrapped
