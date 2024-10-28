""" Asynchronous handling for visual components.

This includes a separate thread dedicated for visualization and messaging.
"""
# NOTE(nopdive): This is run on a single global thread.
# Consider per engine threads later after memory review on engine.

import asyncio
import threading
from asyncio import AbstractEventLoop, Future, Task
from typing import Tuple, Coroutine


def _start_asyncio_loop(loop: AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _asyncio_background_thread() -> Tuple[threading.Thread, AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=_start_asyncio_loop, args=(loop,))
    thread.daemon = True
    return thread, loop

def _run_thread_if_needed():
    global _loop
    global _thread

    if _loop is None:
        _thread, _loop = _asyncio_background_thread()
        _thread.start()
    return _thread, _loop


async def async_task(coroutine: Coroutine) -> Task:
    """ Creates an asyncio task from coroutine.

    Args:
        coroutine: Coroutine within task.

    Returns:
        Asyncio task.
    """
    task = asyncio.create_task(coroutine)
    return task


async def print_all_tasks():  # pragma: no cover
    """Prints all tasks running in visual thread loop."""
    for task in asyncio.all_tasks():
        print(task)


def async_loop() -> AbstractEventLoop:
    """ Returns async loop in visual thread.

    Returns:
        Async loop of visual thread.
    """
    _, loop = _run_thread_if_needed()
    return loop


# TODO: Test.
def call_soon_threadsafe(cb, *args, context = None):
    loop = async_loop()
    return loop.call_soon_threadsafe(cb, *args, context=context)


def run_async_coroutine(coroutine: Coroutine) -> Future:
    """ Runs an asynchronous coroutine in the visual thread.

    Args:
        coroutine: Coroutine to be run on visual thread.

    Returns:
        Future of coroutine.
    """
    _, loop = _run_thread_if_needed()
    future = asyncio.run_coroutine_threadsafe(coroutine, loop)
    return future

_loop = None
_thread = None