"""
Heavily inspired by (read: largely stolen from)
https://github.com/miguelgrinberg/greenletio
"""

import contextvars
import sys
import threading
from functools import wraps
from typing import Any, Callable, Coroutine, TypeVar, cast

from greenlet import getcurrent, greenlet  # type: ignore[import-untyped]
from typing_extensions import ParamSpec


class ReentrantAsyncException(RuntimeError):
    """Exception raised when a coroutine is awaited in a non-greenlet context."""

    pass


P = ParamSpec("P")
T = TypeVar("T")


def reentrant_await(coro: Coroutine[Any, Any, T]) -> T:
    """
    Sends a coroutine to the parent greenlet, which is expected to await the coroutine
    for us and send the result back.

    When there is no parent greenlet, we raise a ReentrantAsyncException.
    """

    parent_gl = getcurrent().parent
    if parent_gl is None:
        coro.close()
        raise ReentrantAsyncException("Attempted to use synchronous entry-point in async context")
    return cast(T, parent_gl.switch(coro))


def sync_to_reentrant_async(fn: Callable[P, T]) -> Callable[P, Coroutine[Any, Any, T]]:
    """
    Decorator to convert a synchronous function into a re-entrant asynchronous one.

    Calls to `reentrant_await` down the stack will bounce back here, and we we'll await
    the coroutine for them
    """

    @wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        gl = greenlet(fn)
        gl.gr_context = contextvars.copy_context()
        coro = gl.switch(*args, **kwargs)
        while gl:
            coro = coro
            try:
                result = await coro
            except:  # noqa: E722
                # this catches exceptions from async functions awaited in
                # sync code, and re-raises them in the greenlet
                coro = gl.throw(*sys.exc_info())
            else:
                coro = gl.switch(result)
        return coro

    return wrapper


def run_async_coroutine_in_bg_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run a coroutine in the background thread and wait for it to finish.
    (This is a blocking call.)
    """

    from .registry import get_bg_async

    bg_async = get_bg_async()
    thread, _ = bg_async._thread_and_loop()
    if thread is threading.current_thread():
        coro.close()
        raise RuntimeError("Cannot nest async call -- already in background thread.")
    fut = bg_async.run_async_coroutine(coro)
    return fut.result()
