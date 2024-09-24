import functools
import inspect
import threading

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
    # if we are not yet being used as a decorator, then save the args

    if f is None:
        return functools.partial(
            guidance, stateless=stateless, cache=cache, dedent=dedent, model=model,
        )

    # this strips out indentation in multiline strings that aligns with the current python indentation
    if dedent is True or dedent == "python":
        f = strip_multiline_string_indents(f)

    # we cache if requested
    if cache:
        f = functools.cache(f)

    return GuidanceFunction(f, stateless=stateless, model=model)


class GuidanceFunction:
    def __init__(
        self,
        f,
        *,
        stateless = False,
        model = Model,
    ):
        # Update self with the wrapped function's metadata
        functools.update_wrapper(self, f)
        # Remove the first argument from the wrapped function
        signature = inspect.signature(f)
        params = list(signature.parameters.values())
        params.pop(0)
        self.__signature__ = signature.replace(parameters=params)

        self.f = f
        self.stateless = stateless
        self.model = model
        self._wrapper = None

    def __call__(self, *args, **kwargs):
        # "Cache" the wrapped function (needed for recursive calls)
        if self._wrapper is None:
            self._wrapper = _decorator(self.f, stateless=self.stateless, model=self.model)
        return self._wrapper(*args, **kwargs)

    # Cache the bound method (needed for recursive calls)
    @functools.cache
    def __get__(self, instance, owner=None, /):
        if instance is None:
            return self

        return GuidanceMethod(
            self.f,
            stateless=self.stateless,
            model=self.model,
            instance=instance,
            owner=owner,
        )

    def __repr__(self):
        return f"<GuidanceFunction {self.__module__}.{self.__qualname__}{self.__signature__}>"


class GuidanceMethod(GuidanceFunction):
    def __init__(self, f, *, stateless=False, model=Model, instance, owner):
        super().__init__(
            f.__get__(instance, owner),
            stateless=stateless,
            model=model
        )
        # Save the instance and owner for introspection
        self._instance = instance
        self._owner = owner

    def __get__(self, instance, owner=None, /):
        raise AttributeError("GuidanceMethod is already bound to an instance")

    def __repr__(self):
        return f"<bound GuidanceMethod {self.__qualname__} of {self._instance!r}>"


_null_grammar = string("")


def _decorator(f, *, stateless, model):
    # Use thread local to store the reference to the grammar node for recursive calls
    # Otherwise, shared state between threads may otherwise trick us into thinking we are in a recursive call
    thread_local = threading.local()

    def wrapped(*args, **kwargs):

        # make a stateless grammar if we can
        if stateless is True or (
            callable(stateless) and stateless(*args, **kwargs)
        ):

            # if we have a (deferred) reference set, then we must be in a recursive definition and so we return the reference
            reference = getattr(thread_local, "_self_call_reference_", None)
            if reference is not None:
                return reference

            # otherwise we call the function to generate the grammar
            else:

                # set a DeferredReference for recursive calls (only if we don't have arguments that might make caching a bad idea)
                no_args = len(args) + len(kwargs) == 0
                if no_args:
                    thread_local._self_call_reference_ = DeferredReference()

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
                        thread_local._self_call_reference_.value = node
                finally:
                    if no_args:
                        del thread_local._self_call_reference_

                return node

        # otherwise must be stateful (which means we can't be inside a select() call)
        else:
            return RawFunction(f, args, kwargs)

    # attach this as a method of the model class (if given)
    # if model is not None:
    #     setattr(model, f.__name__, f)

    return wrapped
