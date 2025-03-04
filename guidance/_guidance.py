import functools
import inspect
import threading
from typing import Any
import weakref
import dataclasses
from contextvars import ContextVar

from ._grammar import string

from ._ast import Function, RuleRefNode, RuleNode
from ._utils import strip_multiline_string_indents, make_weak_bound_method, signature_pop
from .models import Model

_in_stateless_context: ContextVar[bool] = ContextVar("in_stateless_context", default=False)

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

    return GuidanceFunction(f, stateless=stateless, cache=cache, model=model)


class GuidanceFunction:
    def __init__(
        self,
        f,
        *,
        stateless = False,
        cache = False,
        model = Model,
    ):
        self.f = f
        self.stateless = stateless
        self.cache = cache
        self.model = model
        self._impl = _decorator(f, stateless=stateless, cache=cache, model=model)
        self._methods: dict[Any, GuidanceMethod] = {}

        # Update self with the wrapped function's metadata
        functools.update_wrapper(self, self._impl)
        # Pretend to be one level of wrapping lower than we are
        self.__wrapped__ = self._impl.__wrapped__

    def __call__(self, *args, **kwargs):
        return self._impl(*args, **kwargs)

    def __get__(self, instance, owner=None, /):
        """
        Return a GuidanceMethod bound to the instance.
        """
        if instance is None:
            return self
        return GuidanceMethod.from_guidance_function(self, instance)

    def __repr__(self):
        return f"<GuidanceFunction {self.__module__}.{self.__qualname__}{self.__signature__}>"

class GuidanceMethod:
    impl_cache = {}
    def __init__(self, impl, instance):
        # Make object that looks like a method (__self__ and __func__) in order to be able to better support weak referencing via weakref.WeakMethod
        # Note we keep a hard reference to the instance to keep it (and therefore our cached impl) alive as long as we are alive
        self.__self__ = instance
        self.__func__ = impl

        # Update self with the wrapped function's metadata
        functools.update_wrapper(self, impl)
        # Pretend to be one level of wrapping lower than we are
        self.__wrapped__ = impl.__wrapped__

    @classmethod
    def from_guidance_function(cls, guidance_function: GuidanceFunction, instance: Any) -> "GuidanceMethod":
        # We can't directly use a weakref.WeakKeyDictionary because those don't really work when the key objects
        # are allowed to change their hash value.

        # Instead use instance hash in addition to identity to make sure we miss the cache if the instance is meaningfully mutated.
        # This should be safe because an id will only be reused after the original object is garbage collected, at which point we
        # should have removed the cache entry (since we use weakref.finalize to remove the cache entry when the instance is deleted).
        key = (guidance_function.f, hash(instance), id(instance))
        try:
            impl = cls.impl_cache[key]
        except KeyError:
            # Make a weak bound method to prevent the instance from being kept alive by the cache
            weak_method = make_weak_bound_method(guidance_function.f, instance)
            impl = _decorator(weak_method, stateless=guidance_function.stateless, cache=guidance_function.cache, model=guidance_function.model)
            cls.impl_cache[key] = impl
            # Clean up the cache when the instance is deleted
            weakref.finalize(instance, cls.impl_cache.pop, key)
        return cls(impl, instance)

    def __call__(self, *args, **kwargs):
        return self.__func__(*args, **kwargs)

    def __repr__(self):
        return f"<bound GuidanceMethod {self.__qualname__} of {self.__self__!r}>"


_null_grammar = string("")


def _decorator(f, *, stateless, cache, model):
    # we cache the function itself if requested
    # do this before updating the wrapper so we can maintain the __wrapped__ chain
    if cache:
        f = functools.cache(f)

    # Use thread local to store the reference to the grammar node for recursive calls
    # Otherwise, shared state between threads may otherwise trick us into thinking we are in a recursive call
    thread_local = threading.local()

    @functools.wraps(f)
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
                # set the stateless context variable so that others can detect that we're currently calling a stateless function
                token = _in_stateless_context.set(True)

                # set a RuleRefNode for recursive calls (only if we don't have arguments that might make caching a bad idea)
                no_args = len(args) + len(kwargs) == 0
                if no_args:
                    thread_local._self_call_reference_ = RuleRefNode()

                try:
                    # call the function to get the grammar node
                    node = f(_null_grammar, *args, **kwargs)
                except:
                    raise
                else:
                    # If we're just wrapping a RuleNode, don't add an extra layer of RuleNode
                    if isinstance(node, RuleNode):
                        rule = dataclasses.replace(node, name=f.__name__)
                    else:
                        rule = RuleNode(name=f.__name__, value=node)
                    # set the reference value with our generated node
                    if no_args:
                        thread_local._self_call_reference_.set_target(rule)
                finally:
                    # Reset the stateless context back to the previous value
                    _in_stateless_context.reset(token)
                    # Clean up the thread local reference
                    if no_args:
                        del thread_local._self_call_reference_

                return rule

        # otherwise must be stateful (which means we can't be inside a select() call)
        else:
            return Function(f, args, kwargs)
 
    # Remove the first argument from the wrapped function since we're going to drop the `lm` argument
    wrapped.__signature__ = signature_pop(inspect.signature(f), 0)

    # attach this as a method of the model class (if given)
    # if model is not None:
    #     setattr(model, f.__name__, f)

    return wrapped
