import re
from copy import deepcopy
from typing import Optional

from typing_extensions import Self

from guidance._grammar import Null, RawFunction, _call_pool, _tag_pattern
from guidance._singleton import get_renderer, get_trace_handler
from guidance.trace import (
    NodeAttr,
    RoleCloserInput,
    RoleOpenerInput,
    TextOutput,
    TraceNode,
)
from guidance.visual import TraceMessage

from .ast import MessageChunk, Node, RoleEnd, RoleStart
from .client import Client

_active_role: ContextVar[Optional["RoleStart"]] = ContextVar("active_role", default=None)

_id_counter: int = 0


def _gen_id():
    global _id_counter

    _id = _id_counter
    _id_counter += 1
    return _id


class Model:
    def __init__(
        self,
        client: Client,
        echo: bool = True,
    ) -> None:
        self.client = client
        self.echo = echo
        self._state = client.initial_state()
        self._active_role: Optional["RoleStart"] = None

        self._parent: Optional["Model"] = None
        self._parent_id: Optional[int] = None
        self._id: int = _gen_id()
        self._trace_nodes: set[TraceNode] = set()
        self._update_trace_node(self._id, self._parent_id, None)

    def _update_trace_node(
        self, identifier: int, parent_id: Optional[int], node_attr: Optional[NodeAttr] = None
    ) -> None:
        trace_handler = get_trace_handler()
        trace_node = trace_handler.update_node(identifier, parent_id, node_attr)
        self._trace_nodes.add(trace_node)
        if self.echo:
            get_renderer(trace_handler).update(
                TraceMessage(
                    trace_id=identifier,
                    parent_trace_id=parent_id,
                    node_attr=node_attr,
                )
            )

    def __add__(self, other: Node) -> Self:
        self = self._apply_role_changes()
        if isinstance(other, str):
            if other == "":
                return self
            other = extract_embedded_nodes(other)
        if isinstance(other, RawFunction):
            return other(self)
        self = self._apply_node(other)
        return self

    def _apply_node(self, node: Node) -> Self:
        for chunk in self.client.run(self._state, node):
            self = self._apply_chunk(chunk)
        return self

    def _apply_chunk(self, chunk: MessageChunk) -> Self:
        self = self.copy()
        self._state.apply_chunk(chunk)
        if isinstance(chunk, str):
            self._update_trace_node(self._id, self._parent_id, TextOutput(value=chunk))
        elif isinstance(chunk, RoleStart):
            self._update_trace_node(self._id, self._parent_id, RoleOpenerInput(name=chunk.role))
        elif isinstance(chunk, RoleEnd):
            self._update_trace_node(self._id, self._parent_id, RoleCloserInput())
        else:
            raise NotImplementedError(f"Unsupported chunk type: {type(chunk)}")
        return self

    @contextmanager
    def role(self, role: str) -> Iterator[None]:
        # _apply_chunk will raise an exception via _api_state.apply_chunk if roles are not supported
        role_start = RoleStart(role)
        token = _active_role.set(role_start)
        try:
            yield
        finally:
            _active_role.reset(token)

    def _apply_role_changes(self) -> Self:
        orig_self = self
        active_role = _active_role.get()
        if self._active_role != active_role:
            if self._active_role is not None:
                self = self._apply_chunk(RoleEnd(self._active_role.id))
            if active_role is not None:
                self = self._apply_chunk(active_role)
            if self is orig_self:
                self = self.copy()
            self._active_role = active_role
        return self

    def system(self) -> AbstractContextManager[None]:
        return self.role("system")

    def user(self) -> AbstractContextManager[None]:
        return self.role("user")

    def assistant(self) -> AbstractContextManager[None]:
        return self.role("assistant")

    def __str__(self) -> str:
        return self.client.format_state(self._state)

    def copy(self) -> Self:
        obj = object.__new__(self.__class__)
        obj.__dict__.update(self.__dict__)

        obj._state = deepcopy(self._state)
        obj._id = _gen_id()
        obj._parent_id = self._id
        obj._trace_nodes = set()
        obj._parent = self
        obj._update_trace_node(obj._id, obj._parent_id, None)
        return obj


def extract_embedded_nodes(value: str) -> Node:
    parts: list[str] = re.split(_tag_pattern, value)

    if len(parts) == 1:
        return value

    is_id = False
    grammar = Null()
    for part in parts:
        if is_id:
            call = _call_pool[part]
            grammar += call
        else:
            grammar += part
        is_id = not is_id
    return grammar
