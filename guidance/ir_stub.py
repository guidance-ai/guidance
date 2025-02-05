from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, Iterator, Optional, TypeVar, Union
from uuid import UUID, uuid4

from typing_extensions import Self

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

C = TypeVar("C", bound=MessageChunk)


@dataclass(slots=True)
class Stream(Generic[C]):
    chunks: list[C] = field(default_factory=list)

    def apply_chunk(self, chunk: C) -> None:
        self.chunks.append(chunk)


class CompletionStream(Stream[ContentChunk]):
    pass


@dataclass(slots=True)
class MessageStream(Stream[MessageChunk]):
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
        id = uuid4()
        role_start = RoleStart(role, id)
        self.apply_chunk(role_start)
        return role_start

    def close_role(self, id: UUID) -> None:
        self.apply_chunk(RoleEnd(id))


S = TypeVar("S", bound=Stream)


class Client(Generic[S], ABC):
    @abstractmethod
    def run(self, stream: S, node: Node) -> Iterable[ContentChunk]:
        pass


class Model(Generic[S]):
    def __init__(self, client: Client[S], stream: S) -> None:
        self.client = client
        self._stream = stream

    def _apply_node(self, node: Node) -> None:
        for chunk in self.client.run(self._stream, node):
            self._stream.apply_chunk(chunk)

    def __iadd__(self, other: Node) -> Self:
        self._apply_node(other)
        return self

    def __add__(self, other: Node) -> None:
        raise TypeError("Use += to add nodes")


class CompletionModel(Model[CompletionStream]):
    def __init__(self, client: Client) -> None:
        super().__init__(client, CompletionStream())


class ChatModel(Model[MessageStream]):
    def __init__(self, client: Client) -> None:
        super().__init__(client, MessageStream())

    def _apply_node(self, node: Node) -> None:
        if self._stream.active_role is None:
            raise ValueError(f"Cannot apply node to chat model without active role: {node!r}")
        return super()._apply_node(node)

    @contextmanager
    def role(self, role: str) -> Iterator[None]:
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


class StreamHandler(Generic[S], ABC):
    @abstractmethod
    def process_stream(self, stream: S) -> Any:
        pass


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: tuple[str, ...]


class ChatStreamHandler(StreamHandler[MessageStream]):
    def process_stream(self, stream: Stream) -> Iterable[Message]:
        # TODO: add a prefix cache?
        active_role: Optional[RoleStart] = None
        active_content: list[str] = []
        for chunk in stream.chunks:
            if active_role is None:
                if isinstance(chunk, RoleStart):
                    active_role = chunk
                else:
                    raise ValueError("Cannot apply node without active role")
            elif isinstance(chunk, RoleStart):
                raise ValueError("Cannot open role without closing active role")
            elif isinstance(chunk, RoleEnd):
                if active_role.id != chunk.id:
                    raise ValueError("RoleEnd does not match active role")
                yield Message(active_role.role, tuple(active_content))
                active_role = None
                active_content.clear()
            else:
                active_content.append(chunk)
        if active_role is not None:
            yield Message(active_role.role, tuple(active_content))


class ChatClient(Client[MessageStream]):
    def __init__(self, stream_handler: ChatStreamHandler) -> None:
        self.stream_handler = stream_handler

    def run(self, stream: MessageStream, node: Node) -> Iterable[ContentChunk]:
        messages = self.stream_handler.process_stream(stream)
        if isinstance(node, str):
            yield node
        else:
            raise NotImplementedError("Node must be a string")


def main() -> Model:
    chat_client = ChatClient(ChatStreamHandler())
    chat_model = ChatModel(chat_client)
    with chat_model.system():
        chat_model += "Talk like a pirate!"
    with chat_model.user():
        chat_model += "Hello, model!"
        chat_model += "How are you?"
    with chat_model.assistant():
        chat_model += "I'm doing well, thank you!"
    return chat_model
