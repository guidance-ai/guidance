""" Asynchronous handling for visual components.

This includes a separate thread dedicated for visualization and messaging.
"""

import asyncio
import threading
from asyncio import AbstractEventLoop
from typing import Tuple


class ThreadSafeAsyncCondVar:
    def __init__(self, loop:AbstractEventLoop = None):
        self._loop = loop or asyncio.get_event_loop()
        self._waiters = []
        self._waiters_lock = threading.Lock()

    async def wait(self):
        future = self._loop.create_future()
        with self._waiters_lock:
            self._waiters.append(future)
        try:
            await future
        finally:
            with self._waiters_lock:
                if future in self._waiters:
                    self._waiters.remove(future)

    def notify(self, n=1):
        with self._waiters_lock:
            if n == 0:
                waiters_to_notify = self._waiters
                self._waiters.clear()
            else:
                waiters_to_notify = self._waiters[:n]
                self._waiters = self._waiters[n:]

        for future in waiters_to_notify:
            self._loop.call_soon_threadsafe(future.set_result, None)

    def notify_all(self):
        with self._waiters_lock:
            waiters_to_notify = self._waiters
            self._waiters.clear()
        for future in waiters_to_notify:
            self._loop.call_soon_threadsafe(future.set_result, None)


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


def run_async_task(task):
    _, loop = _run_thread_if_needed()
    asyncio.run_coroutine_threadsafe(task, loop)


def async_loop() -> AbstractEventLoop:
    _, loop = _run_thread_if_needed()
    return loop


_loop = None
_thread = None