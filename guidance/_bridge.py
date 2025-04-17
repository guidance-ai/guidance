"""
Heavily inspired by (read: largely stolen from)
https://github.com/miguelgrinberg/greenletio
"""

from greenlet import greenlet, getcurrent
import sys
import asyncio
import threading
import warnings
from typing import Awaitable, TypeVar, Callable, Union, Optional, cast
from typing_extensions import ParamSpec, Never

P = ParamSpec("P")
T = TypeVar("T")

class AwaitException(Exception):
    """Exception raised when a coroutine is awaited in a non-greenlet context."""
    pass

def await_(coro: Awaitable[T]) -> T:
    """
        Sends a coroutine to the parent greenlet. The parent greenlet should either
        1. await the coroutine in an async function
        2. await_ the coroutine in a sync function, punting the problem further up

        If there is no parent greenlet, we'll call the usual asyncio.run() to run the
        coroutine.

        If this fails due to an existing event loop, that means that the caller is a
        foreign async function (not one of "ours"), and they need to use one of our
        async entry-points to run the coroutine with await.
    """
    parent_gl = getcurrent().parent
    if parent_gl is None:
        return run_async_maybe_in_thread(coro)
    return cast(T, parent_gl.switch(coro))

def async_(fn: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    """
    Decorator to convert a synchronous function into an asynchronous one.

    If `await_` is called somewhere down the call stack, we are prepared to
    receive it and take responsibility for awaiting the coroutine.
    """
    async def decorator(*args: P.args, **kwargs: P.kwargs) -> T:
        gl = greenlet(fn)
        coro: Union[T, Awaitable[T]] = gl.switch(*args, **kwargs)
        while gl:
            coro = cast(Awaitable[T], coro)
            try:
                result = await coro
            except:  # noqa: E722
                # this catches exceptions from async functions awaited in
                # sync code, and re-raises them in the greenlet
                coro = cast(Never, gl.throw(*sys.exc_info()))
            else:
                coro = cast(Union[T, Awaitable[T]], gl.switch(result))
        return coro
    return decorator


def run_async_maybe_in_thread(
    coro: Awaitable[T]
) -> T:
    """
    Run a coroutine in a thread if not already in an async context.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        warnings.warn(
            "Synchronous access to model state from an async context is ill-advised...",
            stacklevel=2
        )
        # We're already in an async loop, so we have to run the coroutine in a nested event loop.
        # TODO: consider raising an exception (sync guidance function called in async context)
        # TODO: consider using some global thread and call asyncio.run_coroutine_threadsafe
        result: Optional[T] = None
        exception: Optional[Exception] = None
        event = threading.Event()
        def run():
            nonlocal result, exception
            try:
                result = cast(T, asyncio.run(coro))
            except Exception as ex:
                exception = ex
            finally:
                event.set()
        thread = threading.Thread(target=run)
        thread.start()
        event.wait()
        thread.join()
        if exception is not None:
            raise exception
        return cast(T, result)
