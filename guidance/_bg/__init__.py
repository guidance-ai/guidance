""" Background thread for asyncio handling.

This is currently being used for messaging, visualization and metrics.
"""

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


class BackgroundAsync:
    """ Runs background thread that has an asyncio event loop."""

    def __init__(self):
        """ Initializes. """
        self._loop = None
        self._thread = None

    def _thread_and_loop(self) -> Tuple[threading.Thread, AbstractEventLoop]:
        if self._loop is None:
            self._thread, self._loop = _asyncio_background_thread()
            self._thread.start()
        return self._thread, self._loop

    def call_soon_threadsafe(self, cb, *args, context = None):
        """ Fires callback in background thread."""

        _, loop = self._thread_and_loop()
        return loop.call_soon_threadsafe(cb, *args, context=context)

    def run_async_coroutine(self, coroutine: Coroutine) -> Future:
        """ Runs an asynchronous coroutine in the visual thread.

        Args:
            coroutine: Coroutine to be run on visual thread.

        Returns:
            Future of coroutine.
        """
        _, loop = self._thread_and_loop()
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        return future

    @staticmethod
    async def async_task(coroutine: Coroutine) -> Task:
        """ Creates an asyncio task from coroutine.

        Args:
            coroutine: Coroutine within task.

        Returns:
            Asyncio task.
        """
        task = asyncio.create_task(coroutine)
        return task

    @staticmethod
    async def print_all_tasks():  # pragma: no cover
        """Prints all tasks running in background thread loop (for debugging purposes)."""
        for task in asyncio.all_tasks():
            print(task)
