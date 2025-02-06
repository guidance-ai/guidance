from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
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
)
from uuid import UUID, uuid4

from typing_extensions import Self, assert_never

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


R = TypeVar("R")


class BaseStream(Generic[R], ABC):
    @abstractmethod
    def apply_chunk(self, chunk: MessageChunk) -> None:
        pass

    @abstractmethod
    def get_state(self) -> R:
        pass


class BaseStreamAdapter(BaseStream[R]):
    def apply_chunk(self, chunk: MessageChunk) -> None:
        match chunk:
            case RoleStart(_, _) as role_start:
                self.apply_role_start(role_start)
            case RoleEnd(_) as role_end:
                self.apply_role_end(role_end)
            case str(text):
                self.apply_text(text)
            case _:
                if TYPE_CHECKING:
                    assert_never(chunk)
                raise NotImplementedError(f"Chunk type {type(chunk)} not supported")

    @abstractmethod
    def apply_role_start(self, role_start: RoleStart) -> None:
        pass

    @abstractmethod
    def apply_role_end(self, role_end: RoleEnd) -> None:
        pass

    @abstractmethod
    def apply_text(self, text: str) -> None:
        pass


class Stream(BaseStream[list[MessageChunk]]):
    def __init__(self) -> None:
        self.chunks: list[MessageChunk] = []
        self.active_role: Optional[RoleStart] = None

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

    def get_state(self) -> list[MessageChunk]:
        return self.chunks


class Client(ABC):
    @abstractmethod
    def run(self, stream: Stream, node: Node) -> Iterable[ContentChunk]:
        pass


SA = TypeVar("SA", bound="BaseStreamAdapter")


class BaseModel(Generic[SA]):
    def __init__(self, client: Client, stream: Stream, stream_adapter: SA) -> None:
        self.client = client
        self._stream = stream
        self._stream_adapter = stream_adapter

    def __iadd__(self, other: Node) -> Self:
        self._apply_node(other)
        return self

    def __add__(self, other: Node) -> None:
        raise TypeError("Use += to add nodes")

    def _apply_node(self, node: Node) -> None:
        for chunk in self.client.run(self._stream, node):
            self._apply_chunk(chunk)

    def _apply_chunk(self, chunk: MessageChunk) -> None:
        self._stream.apply_chunk(chunk)
        self._stream_adapter.apply_chunk(chunk)


class CompletionModel(BaseModel["CompletionStreamAdapter"]):
    pass


class ChatModel(BaseModel["ChatStreamAdapter"]):
    @contextmanager
    def role(self, role: str) -> Iterator[None]:
        # Stream will handle raising an exception if roles are not supported
        role_start = RoleStart(role, uuid4())
        self._apply_chunk(role_start)
        try:
            yield
        finally:
            role_end = RoleEnd(role_start.id)
            self._apply_chunk(role_end)

    def system(self) -> AbstractContextManager[None]:
        return self.role("system")

    def user(self) -> AbstractContextManager[None]:
        return self.role("user")

    def assistant(self) -> AbstractContextManager[None]:
        return self.role("assistant")


class BaseChatMessage(TypedDict):
    role: str


M = TypeVar("M", bound=BaseChatMessage)


class ChatStreamAdapter(Generic[M, R], BaseStreamAdapter[R], ABC):
    def __init__(self) -> None:
        self.active_role: Optional[RoleStart] = None
        self.messages: list[M] = []

    def apply_role_start(self, role_start: RoleStart) -> None:
        if self.active_role is not None:
            raise ValueError(
                f"Cannot open role {role_start.role!r}: {self.active_role.role!r} is already open."
            )
        self.active_role = role_start

    def apply_role_end(self, role_end: RoleEnd) -> None:
        if self.active_role is None:
            raise ValueError("Cannot close role without active role")
        if self.active_role.id != role_end.id:
            raise ValueError("RoleEnd does not match active role")
        active_message = self.get_active_message()
        self.messages.append(active_message)
        self.reset_active_message()

    def reset_active_message(self) -> None:
        self.active_role = None

    @abstractmethod
    def get_active_message(self) -> M:
        pass


class CompletionStreamAdapter(BaseStreamAdapter[str]):
    def __init__(self) -> None:
        self.state = ""

    def apply_role_start(self, role_start: RoleStart) -> None:
        raise TypeError("Roles not supported for completion models")

    def apply_role_end(self, role_end: RoleEnd) -> None:
        raise TypeError("Roles not supported for completion models")

    def apply_text(self, text: str) -> None:
        self.state += text

    def get_state(self) -> str:
        return self.state


class OpenAIContentMessage(TypedDict):
    role: str
    content: list[dict]


class OpenAIAudioMessage(TypedDict):
    role: str
    audio: dict


OpenAIMessage = Union[OpenAIContentMessage, OpenAIAudioMessage]


class OpenAIStreamAdapter(ChatStreamAdapter[OpenAIMessage, list[OpenAIMessage]]):
    def __init__(self) -> None:
        super().__init__()
        self.content: list[dict] = []
        self.audio: Optional[dict] = None

    def get_active_message(self) -> OpenAIMessage:
        if self.active_role is None:
            raise ValueError("Cannot build message without active role")

        if self.audio:
            if self.content:
                raise ValueError("Expected either content or audio in OpenAI message, not both")
            return OpenAIAudioMessage({"role": self.active_role.role, "audio": self.audio})

        return OpenAIContentMessage({"role": self.active_role.role, "content": self.content})

    def reset_active_message(self) -> None:
        self.content = []
        self.audio = None
        super().reset_active_message()

    def apply_text(self, text: str) -> None:
        self.content.append({"type": "text", "text": text})

    def get_state(self) -> list[OpenAIMessage]:
        return self.messages


TC = TypeVar("TC")


class TransformersMessage(TypedDict, Generic[TC]):
    role: str
    content: TC


class TransformersStreamReturn(TypedDict, Generic[TC]):
    messages: list[TransformersMessage[TC]]
    images: list[Any]
    audio: list[Any]
    videos: list[Any]


class BaseTransformersChatStreamAdapter(
    ChatStreamAdapter[TransformersMessage[TC], TransformersStreamReturn[TC]]
):
    content: TC
    images: list[Any]
    audio: list[Any]
    videos: list[Any]

    def get_active_message(self) -> TransformersMessage[TC]:
        if self.active_role is None:
            raise ValueError("Cannot build message without active role")

        return TransformersMessage({"role": self.active_role.role, "content": self.content})

    def get_state(self) -> TransformersStreamReturn[TC]:
        return {
            "messages": self.messages,
            "images": self.images,
            "audio": self.audio,
            "videos": self.videos,
        }


class TransformersStructuredStreamAdapter(BaseTransformersChatStreamAdapter[list[dict]]):
    def __init__(self) -> None:
        super().__init__()
        self.content = []
        self.images = []
        self.audio = []
        self.videos = []

    def reset_active_message(self) -> None:
        self.content = []
        # Don't delete images, audio, or videos, as they are not part of the message
        super().reset_active_message()


class TransformersUnstructuredStreamAdapter(BaseTransformersChatStreamAdapter[str]):
    def __init__(self) -> None:
        super().__init__()
        self.content = ""
        self.images = []
        self.audio = []
        self.videos = []

    def reset_active_message(self) -> None:
        self.content = ""
        # Don't delete images, audio, or videos, as they are not part of the message
        super().reset_active_message()

    def apply_text(self, text: str) -> None:
        if self.content is None:
            self.content = ""
        self.content += text


class Llama3TransformersStreamAdapter(TransformersStructuredStreamAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.content = []

    def apply_text(self, text: str) -> None:
        if self.content is None:
            self.content = []
        self.content.append({"type": "text", "text": text})


class DummyClient(Client):
    def run(self, stream: Stream, node: Node) -> Iterable[ContentChunk]:
        if isinstance(node, str):
            yield node
        else:
            raise NotImplementedError("Node must be a string")


def chat():
    import json

    for sa in [
        OpenAIStreamAdapter,
        TransformersUnstructuredStreamAdapter,
        Llama3TransformersStreamAdapter,
    ]:
        model = ChatModel(DummyClient(), Stream(), sa())
        with model.system():
            model += "Talk like a pirate!"
        with model.user():
            model += "Hello, model!"
            model += "\nHow are you?"
        with model.assistant():
            model += "I'm doing well, thank you!"
        print("-" * 80)
        print(sa.__name__)
        print(json.dumps(model._stream_adapter.get_state(), indent=2))
