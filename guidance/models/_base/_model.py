# TODO(nopdive): This module requires a memory review.

import queue
import threading
import asyncio
from contextvars import ContextVar, copy_context
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Iterator, Optional, TypeVar, Union, Sequence
from typing_extensions import Self

from ..._ast import (
    ASTNode,
    Function,
    AsyncFunction,
    CaptureStart,
    CaptureEnd,
    _parse_tags,
)
from ...trace import (
    NodeAttr,
    TraceNode,
)
from ...visual import TraceMessage
from ..._reentrant_async import sync_to_reentrant_async, reentrant_await, run_async_coroutine_in_bg_async

from ._interpreter import Interpreter
from ._state import State

if TYPE_CHECKING:
    from ...library._block import Block

_below_entry_point: ContextVar[bool] = ContextVar("below_entry_point", default=False)
_active_blocks: ContextVar[tuple["Block", ...]] = ContextVar("active_blocks", default=())
_event_queues: ContextVar[tuple[queue.Queue["Model"], ...]] = ContextVar(
    "event_queues", default=()
)
_id_counter: int = 0


def _gen_id():
    global _id_counter

    _id = _id_counter
    _id_counter += 1

    return _id


S = TypeVar("S", bound=State)
D = TypeVar("D", bound=Any)

from dataclasses import dataclass, field, InitVar

@dataclass
class Model:
    interpreter: InitVar[Interpreter[S]]
    echo: bool = True

    # Private init attributes
    _interpreter: Interpreter = field(init=False)
    _parent: Optional["Model"] = None
    _pending: Union[None, ASTNode, Function] = None
    _active_blocks: tuple["Block", ...] = ()

    # Private non-init attributes
    _parent_id: Optional[int] = field(init=False, default=None)
    _id: int = field(init=False, default_factory=_gen_id)
    _trace_nodes: set[TraceNode] = field(init=False, default_factory=set)

    def __post_init__(self, interpreter: Interpreter) -> None:
        self._interpreter = interpreter
        # Set the parent ID if we have a parent
        if self._parent is not None:
            self._parent_id = self._parent._id

    def copy(self) -> Self:
        obj = object.__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        # Use the base-class's __init__ to set up the new object
        # TODO: if we can move to having just the one Model class,
        # we can replace this all with a simple `dataclasses.replace(self, ...)`
        Model.__init__(
            obj,
            interpreter=deepcopy(self._interpreter),
            # TODO: should this be our parent? Or is the copy really our child?
            _parent=self,
            _pending=self._pending,
            _active_blocks=self._active_blocks,
        )
        return obj

    def _update_trace_node(
        self, identifier: int, parent_id: Optional[int], node_attr: Optional[NodeAttr] = None
    ) -> None:
        from ...registry import get_trace_handler, get_renderer

        trace_handler = get_trace_handler()
        trace_node = trace_handler.update_node(identifier, parent_id, node_attr)
        self._trace_nodes.add(trace_node)
        if self.echo:
            get_renderer().update(
                TraceMessage(
                    trace_id=identifier,
                    parent_trace_id=parent_id,
                    node_attr=node_attr,
                ),
            )
        pass

    def _increment_trace_id(self) -> None:
        # This is a bit of a hack to get the trace ids working (only one output attr is allowed per id, so we need to increment.)
        # Parent will be the real parent, so this is all a bit of a mess. TODO: allow multiple output attrs per id
        self._parent_id = self._id
        self._id = _gen_id()

    def _add_to_pending(self, item: Union[ASTNode, Function]) -> None:
        if self._pending is None:
            self._pending = item
        else:
            self._pending += item

    def __add__(self, other: Union[str, Function, AsyncFunction, ASTNode]) -> Self:
        self = self.copy()
        self._apply_blocks()
        if isinstance(other, str):
            if other == "":
                return self
            other = _parse_tags(other)
        if isinstance(other, (ASTNode, Function, AsyncFunction)):
            self._add_to_pending(other)
            return self
        return NotImplemented

    def _send_to_event_queue(self) -> None:
        """For streaming"""
        for event_queue in _event_queues.get():
            event_queue.put(self.copy())

    def stream(self) -> "ModelStream":
        """Return a new model stream object that delays execution until it is iterated over."""
        return ModelStream(self)

    def _apply_blocks(self) -> None:
        global_active_blocks = _active_blocks.get()
        new_active_blocks = []
        for block in reversed(self._active_blocks):
            # Close blocks that are not globally active anymore
            if block not in global_active_blocks:
                if block.closer is not None:
                    closer = block.closer
                    if isinstance(closer, str):
                        closer = _parse_tags(closer)
                    self._add_to_pending(closer)
                if block.name is not None:
                    self._add_to_pending(CaptureEnd(name=block.name))
            else:
                # Not closed, so keep it
                new_active_blocks.append(block)
        new_active_blocks = list(reversed(new_active_blocks))
        for block in global_active_blocks:
            # Open blocks that are not yet locally active
            if block not in self._active_blocks:
                new_active_blocks.append(block)
                if block.name is not None:
                    self._add_to_pending(CaptureStart(name=block.name))
                if block.opener is not None:
                    opener = block.opener
                    if isinstance(opener, str):
                        opener = _parse_tags(opener)
                    self._add_to_pending(opener)
        self._active_blocks = tuple(new_active_blocks)

    def __str__(self) -> str:
        return str(self._get_state())

    def __len__(self):
        return len(str(self))

    def __setitem__(self, key, value):
        raise Exception(
            "Model objects are immutable so you can't use __setitem__! Consider using the .set(key, value) method instead to create a new updated model object."
        )

    def __getitem__(self, key: str) -> Any:
        try:
            captures = self._get_state().captures[key]
        except KeyError:
            raise KeyError(f"Model does not contain the variable '{key}'")
        if isinstance(captures, list):
            return [c["value"] for c in captures]
        else:
            return captures["value"]

    def __contains__(self, key: str) -> bool:
        return key in self._get_state().captures

    def get(self, key: str, default: Optional[D] = None) -> Union[str, list[str], None, D]:
        """Return the value of a variable, or a default value if the variable is not present.

        Parameters
        ----------
        key : str
            The name of the variable.
        default : Any
            The value to return if the variable is not current set.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def set(self, key: str, value: Union[str, list[str]]) -> Self:
        """Return a new model with the given variable value set.

        Parameters
        ----------
        key : str
            The name of the variable to be set.
        value : str
            The value to set the variable to.
        """
        self = self.copy()
        if isinstance(value, list):
            self._get_state().captures[key] = [{"value": v, "log_prob": None} for v in value]
        else:
            self._get_state().captures[key] = {"value": value, "log_prob": None}
        return self

    def remove(self, key: str) -> Self:
        """Return a new model with the given variable deleted.

        Parameters
        ----------
        key : str
            The variable name to remove.
        """
        self = self.copy()
        self._get_state().captures.pop(key)
        return self

    def log_prob(
        self, key: str, default: Optional[D] = None
    ) -> Union[float, list[Union[float, None]], None, D]:
        """Return the log probability of a variable, or a default value if the variable is not present.

        Parameters
        ----------
        key : str
            The name of the variable.
        default : Any
            The value to return if the variable is not current set.
        """
        try:
            captures = self._get_state().captures[key]
        except KeyError:
            return default
        if isinstance(captures, list):
            return [c["log_prob"] for c in captures]
        else:
            return captures["log_prob"]

    def __getattribute__(self, name):
        if name == "engine":
            # For legacy model.engine access (mostly for tests...)
            return getattr(self._interpreter, "engine")
        return super().__getattribute__(name)

    async def run_batched_async(self, items: Sequence[Union[str, Function, AsyncFunction, ASTNode]]) -> Self:
        lms = [self + item for item in items]
        coros = [lm._run() for lm in lms]
        await asyncio.gather(*coros)
        return lms

    def run_batched(self, items: Sequence[Union[str, Function, AsyncFunction, ASTNode]]) -> Self:
        if not _below_entry_point.get():
            return run_async_coroutine_in_bg_async(self.run_batched_async(items))
        return reentrant_await(self.run_batched_async(items))

    async def _run(self) -> None:
        # TODO: trace `InputAttr`s
        async def inner():
            new_self = self.copy()
            # may be some pending blocks
            new_self._apply_blocks()
            while isinstance(new_self._pending, (Function, AsyncFunction)):
                func = new_self._pending
                new_self._pending = None
                new_self._active_blocks = ()
                if isinstance(func, AsyncFunction):
                    new_self = await func(new_self)
                else:
                    # If someone awaits us directly (i.e. we're not below an `await_`),
                    # we need to wrap the sync part in `async_` to avoid blocking our caller's
                    # event loop.
                    # Otherwise, this is effectively equivalent to func(new_self)
                    new_self = await sync_to_reentrant_async(func)(new_self)
                # may be some pending blocks
                new_self._apply_blocks()
            self.__dict__ = new_self.__dict__ # I guess
            if self._pending is None:
                return

            assert isinstance(self._pending, ASTNode)
            node = self._pending
            self._pending = None
            await self._run_node(node)

        # Mark that we are below the entry point so that
        # `_run_sync` knows to use `await_` instead of
        # running in the background thread.
        token = _below_entry_point.set(True)
        try:
            return await inner()
        finally:
            _below_entry_point.reset(token)

    def _run_sync(self) -> None:
        if not _below_entry_point.get():
            return run_async_coroutine_in_bg_async(self._run())
        return reentrant_await(self._run())

    async def _run_node(self, node: ASTNode) -> None:
        async for node_attr in self._interpreter.run(node):
            self._increment_trace_id()
            self._update_trace_node(self._id, self._parent_id, node_attr)
            # Stream current model state
            self._send_to_event_queue()

    async def _get_state_async(self) -> State:
        """Get the state of the model."""
        await self._run()
        return self._interpreter.state

    def _get_state(self) -> State:
        """Get the state of the model."""
        self._run_sync()
        return self._interpreter.state

    async def get_async(self, key: str) -> Any:
        try:
            captures = (await self._get_state_async()).captures[key]
        except KeyError:
            raise KeyError(f"Model does not contain the variable '{key}'")
        if isinstance(captures, list):
            return [c["value"] for c in captures]
        else:
            return captures["value"]

    async def to_string_async(self) -> str:
        """Get the string representation of the model."""
        return str(await self._get_state_async())

    async def length_async(self) -> int:
        """Get the length of the model."""
        return len(await self.to_string_async())

    async def get_token_count_async(self) -> int:
        """Get the token count of the model."""
        return (await self._get_state_async()).token_count

    def get_token_count(self) -> int:
        """Get the token count of the model."""
        return self._get_state().token_count

class ModelStream:
    def __init__(
        self,
        model: Model,
        timeout: float = 5.0,
    ) -> None:
        """Create a model stream object that delays execution until it is iterated over."""
        if model.echo:
            model = model.copy()
            model.echo = False  # turn off display echoing
        self.model = model
        self.timeout = timeout

    def __add__(self, other: Any) -> Self:
        return ModelStream(self.model + other)

    def __iter__(self) -> Iterator[Model]:
        """Starts a thread to execute the model and grammar, yielding events as they occur."""

        events = queue.Queue()
        event_queues = _event_queues.get() + (events,)
        token = _event_queues.set(event_queues)

        # Define the target function for the thread
        def target(ctx):
            _event_queues.set(ctx[_event_queues])
            try:
                self.model._run_sync()
                events.put(None)  # mark that we are done
            except BaseException as ex:
                events.put(ex)

        # Start the thread
        thread = threading.Thread(target=target, args=(copy_context(),))
        thread.start()

        # Yield events from the queue as they become available
        while True:
            try:
                # Wait for an event with a timeout to allow for thread termination
                event = events.get(timeout=self.timeout)
                if event is None:
                    break
                elif isinstance(event, BaseException):
                    raise event
                yield event
            except queue.Empty:
                # Check if the thread is still alive
                if not thread.is_alive():
                    break

        # Ensure the thread has completed
        thread.join()

        # Reset the event queues context variable
        _event_queues.reset(token)
