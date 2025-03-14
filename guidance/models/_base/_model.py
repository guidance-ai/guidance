# TODO(nopdive): This module requires a memory review.

import queue
import threading
from contextvars import ContextVar, copy_context
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, Iterator, Optional, TypeVar, Union

from typing_extensions import Self

from ..._ast import ASTNode, Function, _parse_tags
from ..._ast import (
    ASTNode,
    Function,
    GenAudio,
    ImageBlob,
    ImageUrl,
    LiteralNode,
    RoleEnd,
    RoleStart,
    _parse_tags,
)
from ...trace import (
    ImageInput,
    LiteralInput,
    NodeAttr,
    RoleCloserInput,
    RoleOpenerInput,
    StatelessGuidanceInput,
    TextOutput,
    TraceNode,
)
from ...trace._trace import AudioInput
from ...visual import TraceMessage
from ._client import Client
from ._state import State

if TYPE_CHECKING:
    from ...library._block import Block

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


class Model(Generic[S]):
    def __init__(
        self,
        client: Client[S],
        state: S,
        echo: bool = True,
    ) -> None:
        self.echo = echo
        self._client = client
        self._state = state
        self._active_blocks: dict[Block, int] = {}
        self.token_count: int = 0

        self._parent: Optional["Model"] = None
        self._parent_id: Optional[int] = None
        self._id: int = _gen_id()
        self._trace_nodes: set[TraceNode] = set()
        self._update_trace_node(self._id, self._parent_id, None)

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

    def __add__(self, other: Union[str, Function, ASTNode]) -> Self:
        self = self._apply_blocks()
        if isinstance(other, str):
            if other == "":
                return self
            other = _parse_tags(other)
        if isinstance(other, Function):
            return other(self)
        if isinstance(other, ASTNode):
            self = self._apply_node(other)
            self = self._update_open_block_captures()
            return self
        return NotImplemented

    def _apply_node(self, node: ASTNode) -> Self:
        self = self.copy()

        # Input side of trace handler.
        # TODO: StatefulGuidanceInput up in __add__?
        if isinstance(node, RoleStart):
            self._update_trace_node(self._id, self._parent_id, RoleOpenerInput(name=node.role))
        elif isinstance(node, RoleEnd):
            self._update_trace_node(self._id, self._parent_id, RoleCloserInput(name=node.role))
        elif isinstance(node, LiteralNode):
            self._update_trace_node(self._id, self._parent_id, LiteralInput(value=node.value))
        elif isinstance(node, ImageBlob):
            self._update_trace_node(self._id, self._parent_id, ImageInput(value=node.data))
        elif isinstance(node, ImageUrl):
            # TODO -- let's avoid downloading it here
            pass
        elif isinstance(node, GenAudio):
            self._update_trace_node(
                self._id, self._parent_id, AudioInput(value="")
            )  # TODO -- what goes here?
        else:
            self._update_trace_node(self._id, self._parent_id, StatelessGuidanceInput(value=node))

        for i, output_attr in enumerate(self._client.run(self._state, node)):
            if isinstance(output_attr, TextOutput):
                # TODO: put this elsewhere (inside state?)
                self.token_count += output_attr.token_count
            if i != 0:
                # On the first iteration, we already have a fresh trace node
                # TODO: should be allowed to associate multiple output_attrs with a single input node?
                # TODO: put this responsibility on the client in the case that it breaks a single input
                # node into multiple input nodes to be handled sequentially?
                self._parent_id = self._id
                self._id = _gen_id()
            self._update_trace_node(self._id, self._parent_id, output_attr)
            # Stream current model state
            self._send_to_event_queue()
        return self

    def _send_to_event_queue(self) -> None:
        """For streaming"""
        for event_queue in _event_queues.get():
            event_queue.put(self.copy())

    def stream(self) -> "ModelStream":
        """Return a new model stream object that delays execution until it is iterated over."""
        return ModelStream(self)

    def _apply_blocks(self) -> Self:
        self = self.copy()
        global_active_blocks = _active_blocks.get()
        for block, start_index in list(reversed(self._active_blocks.items())):
            # Close blocks that are not globally active anymore
            if block not in global_active_blocks:
                self._active_blocks.pop(block)
                if block.closer is not None:
                    closer = block.closer
                    if isinstance(closer, str):
                        closer = _parse_tags(closer)
                    if isinstance(closer, Function):
                        raise NotImplementedError(
                            "Stateful block opener/closer functions are not yet supported"
                        )
                    self = self._apply_node(closer)
            # Update capture regardless of whether or not it's been closed
            if block.name is not None:
                self = self.set(block.name, str(self)[start_index:])
        for block in global_active_blocks:
            # Open blocks that are not yet locally active
            if block not in self._active_blocks:
                # Set start_index to the current length
                self._active_blocks[block] = len(self)
                if block.opener is not None:
                    opener = block.opener
                    if isinstance(opener, str):
                        opener = _parse_tags(opener)
                    if isinstance(opener, Function):
                        raise NotImplementedError(
                            "Stateful block opener/closer functions are not yet supported"
                        )
                    self = self._apply_node(opener)
        return self

    def _update_open_block_captures(self) -> Self:
        self = self.copy()
        for block, start_index in self._active_blocks.items():
            if block.name is not None:
                self = self.set(block.name, str(self)[start_index:])
        return self

    def copy(self) -> Self:
        obj = object.__new__(self.__class__)
        obj.__dict__.update(self.__dict__)

        obj._state = deepcopy(self._state)
        obj._active_blocks = {**self._active_blocks}
        obj._id = _gen_id()
        obj._parent_id = self._id
        obj._trace_nodes = set()
        obj._parent = self
        obj._update_trace_node(obj._id, obj._parent_id, None)
        return obj

    def __str__(self) -> str:
        return str(self._state)

    def __len__(self):
        return len(str(self))

    def __setitem__(self, key, value):
        raise Exception(
            "Model objects are immutable so you can't use __setitem__! Consider using the .set(key, value) method instead to create a new updated model object."
        )

    def __getitem__(self, key: str) -> Any:
        try:
            captures = self._state.captures[key]
        except KeyError:
            raise KeyError(f"Model does not contain the variable '{key}'")
        if isinstance(captures, list):
            return [c["value"] for c in captures]
        else:
            return captures["value"]

    def __contains__(self, key: str) -> bool:
        return key in self._state.captures

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
            self._state.captures[key] = [{"value": v, "log_prob": None} for v in value]
        else:
            self._state.captures[key] = {"value": value, "log_prob": None}
        return self

    def remove(self, key: str) -> Self:
        """Return a new model with the given variable deleted.

        Parameters
        ----------
        key : str
            The variable name to remove.
        """
        self = self.copy()
        self._state.captures.pop(key)
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
            captures = self._state.captures[key]
        except KeyError:
            return default
        if isinstance(captures, list):
            return [c["log_prob"] for c in captures]
        else:
            return captures["log_prob"]

    def __getattribute__(self, name):
        if name == "engine":
            # For legacy model.engine access (mostly for tests...)
            return getattr(self._client, "engine")
        return super().__getattribute__(name)


class ModelStream:
    def __init__(
        self,
        model: Model,
        grammar: Union["ModelStream", str, ASTNode, Function, None] = None,
        timeout=5,
    ) -> None:
        """Create a model stream object that delays execution until it is iterated over."""
        if model.echo:
            model = model.copy()
            model.echo = False  # turn off display echoing
        self.model = model
        self.grammar = grammar
        self.timeout = timeout

    def __add__(self, grammar: Union[str, ASTNode]) -> Self:
        """Extend this delayed chain of execution with another grammar append."""
        if self.grammar is None:
            return ModelStream(self.model, grammar)
        else:
            return ModelStream(self.model, self.grammar + grammar)

    def _inner_run(self, model):
        """This runs the model stream without iterating, and is only using internally by __iter__."""
        if isinstance(self.grammar, ModelStream):
            model = self.grammar._inner_run(model)
        elif self.grammar is None:
            model = self.model + ""
        else:
            model = self.model + self.grammar

    def __iter__(self) -> Iterator[Model]:
        """Starts a thread to execute the model and grammar, yielding events as they occur."""

        events = queue.Queue()
        event_queues = _event_queues.get() + (events,)
        token = _event_queues.set(event_queues)

        # Define the target function for the thread
        def target(ctx):
            _event_queues.set(ctx[_event_queues])
            try:
                self._inner_run(self.model)
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
