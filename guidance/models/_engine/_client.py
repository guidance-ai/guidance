import weakref
import atexit
from multiprocessing import Process, Queue as ProcessQueue
from queue import Queue as ThreadQueue  # TODO: change to asyncio.Queue for async
from threading import Thread, Lock
from base64 import b64decode
from io import BytesIO
from typing import Iterator, Type

from ..._ast import GrammarNode, ImageBlob, LiteralNode, RoleEnd, RoleStart
from ...trace import ImageOutput, OutputAttr, TextOutput
from .._base import Client
from ._engine import Engine
from ._state import EngineState
from ...trace import CaptureOutput

class EngineClient(Client[EngineState]):
    def __init__(self, engine_process: bool, engine_type: Type[Engine], *args, **kwargs):
        self._keep_alive_list = [True]

        self._lock = Lock()
        self._response_queue = ProcessQueue()
        self._work_queue = ProcessQueue()
        self._message_id = 2

        initial_thread_queue = ThreadQueue()
        
        self._outstanding = {1: initial_thread_queue}
        self._work_queue.put(("init", 1, engine_type, args, kwargs))

        self._thread = Thread(target=local_worker, args=(self._keep_alive_list, self._lock, self._outstanding, self._response_queue), daemon=True)
        self._thread.start()

        if engine_process:
            self._worker = Process(target=remote_worker, args=(self._work_queue, self._response_queue), daemon=False)
            self._worker.start()
        else:
            # TODO: set daemon=True?
            self._worker = Thread(target=remote_worker, args=(self._work_queue, self._response_queue), daemon=False)
            self._worker.start()

        # wait for response that the engine has been created in the other process
        exception, *response_args = initial_thread_queue.get()
        if exception is not None:
            raise exception

        self.chat_template = response_args[0]

        def atexit_close(ref):
            orig = ref()
            if orig is not None:
                orig.close()

        ref = weakref.ref(self)
        # Register a weakref so that atexit doesn't hold a reference to self and therefore 
        # kept alive past the scope of anyone holding our object.
        #
        # !! IMPORTANT !!
        #     Call atexit.register AFTER creating the worker thread and process
        #     above because atexit guarantees that the exit handlers are 
        #     processed in reverse order as they were registerd. 
        #     The Thread/Process atexit handlers will HANG if they are
        #     called first because they join the thread/process without
        #     initiating a clean exit for this object.  By registering
        #     after the thread/process we guarantee that our close function
        #     puts sentinels on the queues to initiate clean exits before 
        #     the thread/process is joined.
        #
        atexit.register(atexit_close, ref)

        # Unregister when self is GCed to avoid keeping garbage on the atexit list.
        weakref.finalize(self, atexit.unregister, atexit_close)

    def __del__(self):
        call_close = getattr(self, "close", None)
        if callable(call_close):
            call_close()

    def close(self):
        work_queue = getattr(self, "_work_queue", None)
        if work_queue is not None:
            try:
                work_queue.put(None)
            except:
                pass
            self._work_queue = None

        worker = getattr(self, "_worker", None)
        if worker is not None:
            if work_queue is not None:
                # if work_queue was None from shutdown, just terminate and do not join
                try:
                    worker.join(30)
                except:
                    pass

            if isinstance(worker, Process):
                try:
                    is_alive = worker.is_alive()
                except:
                    is_alive = True  # if it is in a weird state, terminate it

                if is_alive:
                    try:
                        worker.terminate()
                    except:
                        pass

                    try:
                        worker.join(30)
                    except:
                        pass

                try:
                    worker.close()
                except:
                    pass
    
            self._worker = None

        response_queue = getattr(self, "_response_queue", None)
        if response_queue is not None:
            try:
                response_queue.put(None)
            except:
                pass
            self._response_queue = None

        t = getattr(self, "_thread", None)
        if t is not None:
            if response_queue is not None:
                # if response_queue was None from shutdown, do not wait for join
                try:
                    t.join(30)
                except:
                    pass

            self._thread = None

        keep_alive_list = getattr(self, "_keep_alive_list", None)
        if keep_alive_list is not None:
            keep_alive_list[0] = None

    def get_role_start(self, role: str) -> str:
        if self.chat_template is None:
            raise ValueError("Cannot use roles without a chat template")
        return self.chat_template.get_role_start(role)

    def get_role_end(self, role: str) -> str:
        if self.chat_template is None:
            raise ValueError("Cannot use roles without a chat template")
        return self.chat_template.get_role_end(role)

    def role_start(self, state: EngineState, node: RoleStart, **kwargs) -> Iterator[OutputAttr]:
        state.active_role = node.role
        # TODO: mark these as special tokens..?
        yield from self.run(state, LiteralNode(value=self.get_role_start(node.role)), **kwargs)

    def role_end(self, state: EngineState, node: RoleEnd, **kwargs) -> Iterator[OutputAttr]:
        state.active_role = None
        # TODO: mark these as special tokens..?
        yield from self.run(state, LiteralNode(value=self.get_role_end(node.role)), **kwargs)

    def text(self, state: EngineState, node: LiteralNode, **kwargs) -> Iterator[OutputAttr]:
        state.prompt += node.value
        yield TextOutput(value=node.value, is_input=True)

    def grammar(self, state: EngineState, node: GrammarNode, **kwargs) -> Iterator[OutputAttr]:
        thread_queue = ThreadQueue()

        with self._lock:
            message_id = self._message_id
            self._message_id = message_id + 1
            self._outstanding[message_id] = thread_queue

        self._work_queue.put(("grammar", message_id, state, node.ll_grammar()))

        exception, *response_args = thread_queue.get()
        if exception is not None:
            raise exception

        chunks = response_args[0]
        for chunk in chunks:
            if isinstance(chunk, TextOutput):
                state.prompt += chunk.value
                yield chunk
            elif isinstance(chunk, tuple):
                yield state.apply_capture(*chunk)
            else:
                raise Exception("Unexpected chunk type")

class Llama3VisionClient(EngineClient):
    def image_blob(self, state: EngineState, node: ImageBlob, **kwargs) -> Iterator[OutputAttr]:
        try:
            import PIL.Image
        except ImportError:
            raise Exception(
                "Please install the Pillow package `pip install Pillow` in order to use images with Llama3!"
            )

        image_bytes = b64decode(node.data)
        pil_image = PIL.Image.open(BytesIO(image_bytes))
        state.images.append(pil_image)
        state.prompt += "<|image|>"

        yield ImageOutput(value=node.data, input=True)


class Phi3VisionClient(EngineClient):
    def image_blob(self, state: EngineState, node: ImageBlob, **kwargs) -> Iterator[OutputAttr]:
        try:
            import PIL.Image
        except ImportError:
            raise Exception(
                "Please install the Pillow package `pip install Pillow` in order to use images with Llama3!"
            )

        image_bytes = b64decode(node.data)
        pil_image = PIL.Image.open(BytesIO(image_bytes))

        if pil_image in state.images:
            ix = state.images.index(pil_image) + 1
        else:
            state.images.append(pil_image)
            ix = len(state.images)
        state.prompt += f"<|image_{ix}|>"

        yield ImageOutput(value=node.data, input=True)


def partial_decode(data: bytes) -> tuple[str, bytes]:
    try:
        return (data.decode("utf-8"), b"")
    except UnicodeDecodeError as e:
        valid_part = data[: e.start].decode("utf-8")
        delayed_part = data[e.start :]
    return (valid_part, delayed_part)


def local_worker(keep_alive_list: list, lock: Lock, outstanding: dict, response_queue: ProcessQueue):
    # use extremely defensive programming here to handle unclean shutdowns
    while True:
        if keep_alive_list is None:
            break
        is_keep_alive = keep_alive_list[0]
        if is_keep_alive is None:
            break

        if response_queue is None:
            break

        response_args = response_queue.get()
        if response_args is None:
            break
        event_id, *response_args = response_args

        if lock is None:
            break
        with lock:
            if outstanding is None or event_id is None:
                break
            thread_queue = outstanding[event_id]
            del outstanding[event_id]

        if thread_queue is None:
            break
           
        if response_args is None:
            thread_queue.put(tuple(Exception("Model was closed with active work.")))
            break

        thread_queue.put(response_args)

    # flush any outstanding items with Exceptions
    # it is illegal to add work after closing, so do not use the lock.
    if outstanding is None:
        return
    for thread_queue in outstanding:
        if thread_queue is not None:
            thread_queue.put(tuple(Exception("Model was closed with active work.")))


def remote_worker(work_queue: ProcessQueue, response_queue: ProcessQueue):
    engine = None
    while True:
        try:
            work_args = work_queue.get()
        except:
            # If the parent process closes it closes work_queue and we get an exception
            break
        if work_args is None:
            break
        work_type, *work_args = work_args
        if work_type == "init":
            try:
                try:
                    event_id, engine_type, args, kwargs = work_args
                    engine = engine_type(*args, **kwargs)
                    chat_template = engine.get_chat_template()
                except Exception as e:
                    response_queue.put((event_id, e))
                else:
                    response_queue.put((event_id, None, chat_template))
            except:
                # If the parent process closes it closes 
                # response_queue and we get an exception.
                # Exit in this case because the parent is dead.
                break
        elif work_type == "grammar":
            try:
                try:
                    event_id, state, grammar = work_args

                    engine_gen = engine.execute_grammar(
                        state,
                        grammar,
                        ensure_bos_token=True,
                        echo=False,
                    )

                    chunks = []

                    delayed_bytes = b""
                    for chunk in engine_gen:
                        new_bytes = chunk.new_bytes
                        new_text, delayed_bytes = partial_decode(new_bytes)

                        # Update the state
                        chunks.append(TextOutput(value=new_text, token_count=chunk.new_token_count, is_generated=True))

                        # TODO -- rewrite engine internals to make sure chunk.{generated,fast_forwarded}_tokens aren't empty...
                        # # TODO: GenTokenExtra
                        # for token in chunk.generated_tokens:
                        #     yield TextOutput(
                        #         value=token.text,  # TODO: this should really be the token bytes
                        #         is_generated=True,
                        #         token_count=1,
                        #         prob=token.prob,
                        #         tokens=[token],  # TODO: drop this
                        #     )
                        # for token in chunk.force_forwarded_tokens:
                        #     yield TextOutput(
                        #         value=token.text,  # TODO: this should really be the token bytes
                        #         is_generated=False,
                        #         token_count=1,
                        #         prob=token.prob,
                        #         tokens=[token],  # TODO: drop this
                        #     )

                        # # TODO: yield some kind of backtrack signal?

                        for name in chunk.capture_groups.keys():
                            values = chunk.capture_groups[name]
                            log_probs = chunk.capture_group_log_probs[name]
                            if isinstance(values, list):
                                assert isinstance(log_probs, list)
                                assert len(values) == len(log_probs)
                                for value, log_prob in zip(values, log_probs):
                                    chunks.append((name, value.decode("utf-8"), log_prob, True))
                            else:
                                assert isinstance(log_probs, float)
                                chunks.append((name, values.decode("utf-8"), log_probs, False))

                    if delayed_bytes:
                        raise RuntimeError("Shouldn't have any delayed bytes left...")
                except Exception as e:
                    response_queue.put((event_id, e))
                else:
                    response_queue.put((event_id, None, chunks))
            except:
                # If the parent process closes it closes 
                # response_queue and we get an exception.
                # Exit in this case because the parent is dead.
                break
        else:
            raise Exception("Unknown message type.")
