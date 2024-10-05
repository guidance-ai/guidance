import atexit
from multiprocessing import Process, Queue
from threading import Thread, Event
import logging
from ._model import Engine, Model
from ._tokenizer import Tokenizer

logger = logging.getLogger(__name__)

shutdown_none = True

def set_shutdown_flag():
    global shutdown_none
    # python can set anything to None at shutdown, so use None
    shutdown_none = None

atexit.register(set_shutdown_flag)

class LocalEngine(Engine):
    def __init__(self, tokenizer: Tokenizer, compute_log_probs=False):
        super().__init__(tokenizer, compute_log_probs)

        self._response_queue = Queue()
        self._work_queue = Queue()
        self._outstanding = {}

        self._process = Process(target=remote_worker, args=(self._response_queue, self._work_queue), daemon=True)
        self._process.start()

        self._thread = Thread(target=local_worker, args=(self._outstanding, self._response_queue), daemon=True)
        self._thread.start()

    def __del__(self):
        if shutdown_none is not None:
            p = getattr(self, "_process", None)
            if p is not None:
                p.terminate()
                self._process = None
            t = getattr(self, "_thread", None)
            if t is not None:
                t.join(30)  # max 30 seconds wait
                self._thread = None

    def __call__(self, *args, **kwargs):
        #     event = Event()
        #     event_id = id(event)
        #     self._outstanding[event_id] = [event]
        #     work = ["add_three", event_id, 42]
        #     self._work_queue.put(work)
        #     event.wait()
        #     response = self._outstanding[event_id]
        #     del self._outstanding[event_id]

        return super().__call__(*args, **kwargs)


class LocalModel(Model):
    def __init__(self, engine, echo=True, **kwargs):
        """Build a new Local model object that represents a model in a given state."""

        super().__init__(engine, echo, **kwargs)

def local_worker(outstanding: dict, response_queue: Queue):
    while True:
        work = response_queue.get()
        event_id = work[1]
        original = outstanding[event_id]
        event = original[0]
        original[:] = work  # copy the response into the original list
        event.set()

def remote_worker(response_queue: Queue, work_queue: Queue):
    while True:
        work = work_queue.get()
        response_queue.put(work)
