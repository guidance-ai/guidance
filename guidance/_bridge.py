"""
Heavily inspired by (read: largely stolen from)
https://github.com/miguelgrinberg/greenletio
"""

import asyncio
import sys
import contextvars
from typing import Any, Callable, Coroutine, TypeVar, cast

from greenlet import getcurrent, greenlet  # type: ignore[import-untyped]
from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


class AwaitException(RuntimeError):
    """Exception raised when a coroutine is awaited in a non-greenlet context."""

    pass


def await_(coro: Coroutine[Any, Any, T]) -> T:
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
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        else:
            # Close the coro to avoid leaking resources
            coro.close()
            raise AwaitException("Cannot use synchronous await_ within a running event loop.")
    return cast(T, parent_gl.switch(coro))


def async_(fn: Callable[P, T]) -> Callable[P, Coroutine[Any, Any, T]]:
    """
    Decorator to convert a synchronous function into an asynchronous one.

    If `await_` is called somewhere down the call stack, we are prepared to
    receive it and take responsibility for awaiting the coroutine.
    """

    async def decorator(*args: P.args, **kwargs: P.kwargs) -> T:
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

    return decorator
