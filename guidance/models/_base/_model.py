# TODO(nopdive): This module requires a memory review.

import re
from abc import ABC, abstractmethod
from base64 import b64encode
from contextvars import ContextVar
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, Iterator, Optional, TypeVar, Union

from typing_extensions import Self, assert_never

from ..._grammar import Null, RawFunction, _call_pool, _tag_pattern
from ..._singleton import get_renderer, get_trace_handler
from ...experimental.ast import ImageBlob, MessageChunk, Node, RoleEnd, RoleStart
from ...trace import (
    CaptureOutput,
    ImageInput,
    LiteralInput,
    NodeAttr,
    RoleCloserInput,
    RoleOpenerInput,
    TextOutput,
    TraceNode,
)
from ...visual import TraceMessage
from ._state import BaseState

_active_role: ContextVar[Optional["RoleStart"]] = ContextVar("active_role", default=None)
_id_counter: int = 0


def _gen_id():
    global _id_counter

    _id = _id_counter
    _id_counter += 1
    return _id


S = TypeVar("S", bound=BaseState)
D = TypeVar("D", bound=Any)


class Model(ABC, Generic[S]):
    def __init__(
        self,
        echo: bool = True,
    ) -> None:
        self.echo = echo
        self._state = self.initial_state()
        self._active_role: Optional["RoleStart"] = None

        self._parent: Optional["Model"] = None
        self._parent_id: Optional[int] = None
        self._id: int = _gen_id()
        self._trace_nodes: set[TraceNode] = set()
        self._update_trace_node(self._id, self._parent_id, None)

    @abstractmethod
    def run(self, state: S, node: Node) -> Iterator[MessageChunk]:
        pass

    @abstractmethod
    def initial_state(self) -> S:
        pass

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
        for chunk in self.run(self._state, node):
            self = self._apply_chunk(chunk)
        return self

    def _apply_chunk(self, chunk: MessageChunk) -> Self:
        self = self.copy()
        self._state.apply_chunk(chunk)
        if isinstance(
            chunk, (LiteralInput, TextOutput, CaptureOutput, RoleOpenerInput, RoleCloserInput)
        ):
            self._update_trace_node(self._id, self._parent_id, chunk)
        elif isinstance(chunk, ImageBlob):
            self._update_trace_node(
                self._id,
                self._parent_id,
                ImageInput(value=b64encode(chunk.image.tobytes()).decode("utf-8")),
            )
        else:
            if TYPE_CHECKING:
                assert_never(chunk)
            raise NotImplementedError(f"Unsupported chunk type: {type(chunk)}")
        return self

    def _apply_role_changes(self) -> Self:
        orig_self = self
        active_role = _active_role.get()
        if self._active_role != active_role:
            if self._active_role is not None:
                self = self._apply_node(RoleEnd(self._active_role.role))
            if active_role is not None:
                self = self._apply_node(active_role)
            if self is orig_self:
                # Make sure we never mutate in place
                self = self.copy()
            self._active_role = active_role
        return self

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


def partial_decode(data: bytes) -> tuple[str, bytes]:
    try:
        return (data.decode("utf-8"), b"")
    except UnicodeDecodeError as e:
        valid_part = data[: e.start].decode("utf-8")
        delayed_part = data[e.start :]
    return (valid_part, delayed_part)
