from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Generic,
    Iterable,
    Iterator,
    Optional,
    TypedDict,
    TypeVar,
    Union,
)
from uuid import UUID, uuid4

from typing_extensions import NotRequired, Self, assert_never

# TODO
Node = str


@dataclass(frozen=True, slots=True)
class RoleStart:
    role: str
    id: UUID


@dataclass(frozen=True, slots=True)
class RoleEnd:
    id: UUID


ContentChunk = str
MessageChunk = Union[ContentChunk, RoleStart, RoleEnd]


@dataclass(slots=True)
class Stream:
    chunks: list[MessageChunk] = field(default_factory=list)

    def apply_chunk(self, chunk: ContentChunk) -> None:
        self.chunks.append(chunk)

    def open_role(self, role: str) -> RoleStart:
        raise TypeError("Roles not supported for completion models")

    def close_role(self, id: UUID) -> None:
        raise TypeError("Roles not supported for completion models")


@dataclass(slots=True)
class ChatStream(Stream):
    active_role: Optional[RoleStart] = None

    def apply_chunk(self, chunk: MessageChunk) -> None:
        match chunk:
            case RoleStart(role, id):
                if self.active_role is not None:
                    raise ValueError(
                        f"Cannot open role {role!r}: {self.active_role.role!r} is already open."
                    )
                self.active_role = RoleStart(role, id)
            case RoleEnd(id):
                if self.active_role is None:
                    raise ValueError("Cannot close role without active role")
                if self.active_role.id != id:
                    raise ValueError("RoleEnd does not match active role")
                self.active_role = None
            case _:
                if self.active_role is None:
                    raise ValueError("Cannot apply node without active role")
        self.chunks.append(chunk)

    def open_role(self, role: str) -> RoleStart:
        role_start = RoleStart(role, uuid4())
        self.apply_chunk(role_start)
        return role_start

    def close_role(self, id: UUID) -> None:
        self.apply_chunk(RoleEnd(id))


class Client(ABC):
    @abstractmethod
    def run(self, stream: Stream, node: Node) -> Iterable[ContentChunk]:
        pass


class Model:
    def __init__(self, client: Client, stream: Stream) -> None:
        self.client = client
        self._stream = stream

    def __iadd__(self, other: Node) -> Self:
        self._apply_node(other)
        return self

    def __add__(self, other: Node) -> None:
        raise TypeError("Use += to add nodes")

    def _apply_node(self, node: Node) -> None:
        for chunk in self.client.run(self._stream, node):
            self._stream.apply_chunk(chunk)

    @contextmanager
    def role(self, role: str) -> Iterator[None]:
        # Stream will handle raising an exception if roles are not supported
        role_start = self._stream.open_role(role)
        try:
            yield
        finally:
            self._stream.close_role(role_start.id)

    def system(self) -> AbstractContextManager[None]:
        return self.role("system")

    def user(self) -> AbstractContextManager[None]:
        return self.role("user")

    def assistant(self) -> AbstractContextManager[None]:
        return self.role("assistant")


S = TypeVar("S")
R = TypeVar("R")


class BaseStreamHandler(Generic[S, R], ABC):
    # TODO: some kind of prefix caching would be nice, especially if we guarantee
    # that a stream handler will be repeatedly called with an append-only stream
    def process_stream(self, stream: Stream) -> R:
        state = self.initialize()
        for chunk in stream.chunks:
            state = self.process_chunk(chunk, state)
        return self._finalize(state)

    def process_chunk(self, chunk: MessageChunk, state: S) -> S:
        match chunk:
            case RoleStart(_, _) as role_start:
                return self.process_role_start(role_start, state)
            case RoleEnd(_) as role_end:
                return self.process_role_end(role_end, state)
            case str(text):
                return self.process_text(text, state)
            case _:
                if TYPE_CHECKING:
                    assert_never(chunk)
                raise NotImplementedError(f"Chunk type {type(chunk)} not supported")

    @abstractmethod
    def initialize(self) -> S:
        pass

    @abstractmethod
    def finalize(self, state: S) -> R:
        pass

    def _finalize(self, state: S) -> R:
        return self.finalize(state)

    def process_role_start(self, role_start: RoleStart, state: S) -> S:
        raise TypeError("Roles not supported for completion models")

    def process_role_end(self, role_end: RoleEnd, state: S) -> S:
        raise TypeError("Roles not supported for completion models")

    def process_text(self, text: str, state: S) -> S:
        raise NotImplementedError("No default text processing implementation")


class ChatStreamHandler(BaseStreamHandler[dict, R], ABC):
    def initialize(self) -> dict:
        return {"active_role": None, "messages": []}

    def process_role_start(self, role_start: RoleStart, state: dict) -> dict:
        if state["active_role"] is not None:
            raise ValueError(
                f"Cannot open role {role_start.role!r}: {state['active_role'].role!r} is already open."
            )
        state["active_role"] = role_start
        return state

    def process_role_end(self, role_end: RoleEnd, state: dict) -> dict:
        if state["active_role"] is None:
            raise ValueError("Cannot close role without active role")
        if state["active_role"].id != role_end.id:
            raise ValueError("RoleEnd does not match active role")
        state = self.finalize_message(state)
        state["active_role"] = None
        return state

    @abstractmethod
    def finalize_message(self, state: dict) -> dict:
        pass

    def _finalize(self, state: dict) -> R:
        if state["active_role"] is not None:
            # Fictitious role end to close the last message
            state = self.finalize_message(state)
        return self.finalize(state)


class CompletionStreamHandler(BaseStreamHandler[str, str]):
    def initialize(self) -> str:
        return ""

    def process_text(self, text: str, state: str) -> str:
        return state + text

    def finalize(self, state: str) -> str:
        return state


class OpenAIStreamHandler(ChatStreamHandler[list[dict]]):
    def finalize_message(self, state: dict) -> dict:
        content = state.get("content")
        audio = state.get("audio")
        if content is not None and audio is not None:
            raise ValueError("OpenAI expects either content or audio, not both")

        if content is not None:
            message = {"role": state["active_role"].role, "content": content}
            state["messages"].append(message)
            del state["content"]

        elif audio is not None:
            message = {"role": state["active_role"].role, "audio": audio}
            state["messages"].append(message)
            del state["audio"]

        return state

    def finalize(self, state: dict) -> list[dict]:
        return state["messages"]

    def process_text(self, text: str, state: dict) -> dict:
        state.setdefault("content", []).append({"type": "text", "value": text})
        return state


class OpenAIClient(Client):
    def __init__(self) -> None:
        self.stream_handler = OpenAIStreamHandler()

    def run(self, stream: Stream, node: Node) -> Iterable[ContentChunk]:
        messages = self.stream_handler.process_stream(stream)
        if isinstance(node, str):
            yield node
        else:
            raise NotImplementedError("Node must be a string")


def chat():
    client = OpenAIClient()
    stream = ChatStream()
    model = Model(client, stream)
    with model.system():
        model += "Talk like a pirate!"
    with model.user():
        model += "Hello, model!"
        model += "How are you?"
    with model.assistant():
        model += "I'm doing well, thank you!"
    return list(client.stream_handler.process_stream(stream))
