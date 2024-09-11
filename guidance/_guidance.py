import functools
import inspect

from ._grammar import DeferredReference, RawFunction, Terminal, string
from ._utils import strip_multiline_string_indents
from .models import Model


def guidance(
    f = None,
    *,
    stateless = False,
    cache = False,
    dedent = True,
    model = Model,
):
    """Decorator used to define guidance grammars"""
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

            # make a stateless grammar if we can
            if stateless is True or (
                callable(stateless) and stateless(*args, **kwargs)
            ):

                # if we have a (deferred) reference set, then we must be in a recursive definition and so we return the reference
                reference = getattr(f, "_self_call_reference_", None)
                if reference is not None:
                    return reference

                # otherwise we call the function to generate the grammar
                else:

                    # set a DeferredReference for recursive calls (only if we don't have arguments that might make caching a bad idea)
                    no_args = len(args) + len(kwargs) == 0
                    if no_args:
                        f._self_call_reference_ = DeferredReference()

                    try:
                        # call the function to get the grammar node
                        node = f(_null_grammar, *args, **kwargs)
                    except:
                        raise
                    else:
                        if not isinstance(node, (Terminal, str)):
                            node.name = f.__name__
                        # set the reference value with our generated node
                        if no_args:
                            f._self_call_reference_.value = node
                    finally:
                        if no_args:
                            del f._self_call_reference_

                    return node

            # otherwise must be stateful (which means we can't be inside a select() call)
            else:
                return RawFunction(f, args, kwargs)

        # Remove the first argument from the wrapped function
        signature = inspect.signature(f)
        params = list(signature.parameters.values())
        params.pop(0)
        wrapped.__signature__ = signature.replace(parameters=params)

        # attach this as a method of the model class (if given)
        # if model is not None:
        #     setattr(model, f.__name__, f)

        return wrapped
