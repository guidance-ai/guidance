from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    Iterator,
    Optional,
    TypedDict,
    TypeVar,
    Union,
    cast,
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


class BaseChatMessage(TypedDict):
    role: str


M = TypeVar("M", bound=BaseChatMessage)


class BaseChatState(TypedDict, Generic[M]):
    active_role: Optional[RoleStart]
    messages: list[M]


CS = TypeVar("CS", bound=BaseChatState)


class ChatStreamHandler(BaseStreamHandler[CS, R], ABC):
    def initialize(self) -> CS:
        # We can't actually guarantee that BaseChatState is compatible with CS, so this cast is a bit unsafe.
        # So long as sub-classers are careful to not assume that additionally defined fields are present in the
        # state, it should be fine. Having a reasonable default implementation of initialize is probably worth it.
        return cast(CS, {"active_role": None, "messages": []})

    def process_role_start(self, role_start: RoleStart, state: CS) -> CS:
        if state["active_role"] is not None:
            raise ValueError(
                f"Cannot open role {role_start.role!r}: {state['active_role'].role!r} is already open."
            )
        state["active_role"] = role_start
        return state

    def process_role_end(self, role_end: RoleEnd, state: CS) -> CS:
        if state["active_role"] is None:
            raise ValueError("Cannot close role without active role")
        if state["active_role"].id != role_end.id:
            raise ValueError("RoleEnd does not match active role")
        state = self.build_message(state)
        state = self.cleanup_message(state)
        return state

    @abstractmethod
    def build_message(self, state: CS) -> CS:
        pass

    def cleanup_message(self, state: CS) -> CS:
        state["active_role"] = None
        return state

    def _finalize(self, state: CS) -> R:
        if (
            state["active_role"] is not None
        ):  # TODO: should we assert that it actually is NOT None?
            # Bit of a hack to ensure we get the active message
            state = self.build_message(state)
        return self.finalize(state)


class CompletionStreamHandler(BaseStreamHandler[str, str]):
    def initialize(self) -> str:
        return ""

    def process_text(self, text: str, state: str) -> str:
        return state + text

    def finalize(self, state: str) -> str:
        return state


class OpenAIContentMessage(TypedDict):
    role: str
    content: list[dict]


class OpenAIAudioMessage(TypedDict):
    role: str
    audio: dict


OpenAIMessage = Union[OpenAIContentMessage, OpenAIAudioMessage]


class OpenAIState(TypedDict):
    active_role: Optional[RoleStart]
    messages: list[OpenAIMessage]
    content: NotRequired[list[dict]]
    audio: NotRequired[dict]


class OpenAIStreamHandler(ChatStreamHandler[OpenAIState, list[OpenAIMessage]]):
    def build_message(self, state: OpenAIState) -> OpenAIState:
        if state["active_role"] is None:
            raise ValueError("Cannot finalize message without active role")

        content = state.get("content")
        audio = state.get("audio")
        if content is not None and audio is not None:
            raise ValueError("OpenAI expects either content or audio, not both")

        if content is not None:
            content_message = OpenAIContentMessage(
                {"role": state["active_role"].role, "content": content}
            )
            state["messages"].append(content_message)

        elif audio is not None:
            audio_message = OpenAIAudioMessage({"role": state["active_role"].role, "audio": audio})
            state["messages"].append(audio_message)

        return state

    def cleanup_message(self, state: OpenAIState) -> OpenAIState:
        state.pop("content", None)
        state.pop("audio", None)
        return super().cleanup_message(state)

    def finalize(self, state: OpenAIState) -> list[OpenAIMessage]:
        return state["messages"]

    def process_text(self, text: str, state: OpenAIState) -> OpenAIState:
        state.setdefault("content", []).append({"type": "text", "value": text})
        return state


class OpenAIClient(Client):
    def __init__(self) -> None:
        self.stream_handler = OpenAIStreamHandler()

    def run(self, stream: Stream, node: Node) -> Iterable[ContentChunk]:
        if isinstance(node, str):
            yield node
        else:
            raise NotImplementedError("Node must be a string")


# Some transformers models want a string, some want a list of dicts...
TC = TypeVar("TC", bound=Union[str, list[dict]])


class TransformersMessage(TypedDict, Generic[TC]):
    role: str
    content: TC


class TransformersState(TypedDict, Generic[TC]):
    active_role: Optional[RoleStart]
    messages: list[TransformersMessage[TC]]
    content: NotRequired[TC]
    images: NotRequired[list[Any]]
    audio: NotRequired[list[Any]]
    videos: NotRequired[list[Any]]


class TransformersStreamReturn(TypedDict, Generic[TC]):
    messages: list[TransformersMessage[TC]]
    images: list[Any]
    audio: list[Any]
    videos: list[Any]


class BaseTransformersChatStreamHandler(
    ChatStreamHandler[TransformersState[TC], TransformersStreamReturn[TC]]
):
    def build_message(self, state: TransformersState[TC]) -> TransformersState[TC]:
        if state["active_role"] is None:
            raise ValueError("Cannot finalize message without active role")
        content = state.get("content")
        if content is not None:
            message = TransformersMessage({"role": state["active_role"].role, "content": content})
            state["messages"].append(message)
            del state["content"]
        return state

    def cleanup_message(self, state: TransformersState[TC]) -> TransformersState[TC]:
        state.pop("content", None)
        # Don't delete images, audio, or videos, as they are not part of the message
        return super().cleanup_message(state)

    def finalize(self, state: TransformersState[TC]) -> TransformersStreamReturn[TC]:
        return {
            "messages": state["messages"],
            "images": state.get("images", []),
            "audio": state.get("audio", []),
            "videos": state.get("videos", []),
        }


class DefaultTransformersStreamHandler(BaseTransformersChatStreamHandler[str]):
    def process_text(self, text: str, state: TransformersState[str]) -> TransformersState[str]:
        try:
            state["content"] += text
        except KeyError:
            state["content"] = text
        return state


class DefaultTransformersChatClient(Client):
    def __init__(self) -> None:
        self.stream_handler = DefaultTransformersStreamHandler()

    def run(self, stream: Stream, node: Node) -> Iterable[ContentChunk]:
        if isinstance(node, str):
            yield node
        else:
            raise NotImplementedError("Node must be a string")


class Llama3TransformersStreamHandler(BaseTransformersChatStreamHandler[list[dict]]):
    def process_text(
        self, text: str, state: TransformersState[list[dict]]
    ) -> TransformersState[list[dict]]:
        state.setdefault("content", []).append({"type": "text", "text": text})
        return state


def chat():
    client = DefaultTransformersChatClient()
    stream = ChatStream()
    model = Model(client, stream)
    with model.system():
        model += "Talk like a pirate!"
    with model.user():
        model += "Hello, model!"
        model += "\nHow are you?"
    with model.assistant():
        model += "I'm doing well, thank you!"
    return client.stream_handler.process_stream(stream)
