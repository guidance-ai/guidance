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


class BaseState(Generic[R], ABC):
    def __init__(self) -> None:
        self.active_role: Optional[RoleStart] = None

    def apply_chunk(self, chunk: MessageChunk) -> None:
        match chunk:
            case RoleStart(_, _) as role_start:
                self.apply_role_start(role_start)
            case RoleEnd(_) as role_end:
                self.apply_role_end(role_end)
            case _:
                self.apply_content_chunk(chunk)

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
        self.active_role = None

    @abstractmethod
    def get_state(self) -> R:
        pass

    @abstractmethod
    def apply_content_chunk(self, chunk: ContentChunk) -> None:
        pass


class InternalState(BaseState[list[MessageChunk]]):
    def __init__(self) -> None:
        self.chunks: list[MessageChunk] = []
        self.active_role: Optional[RoleStart] = None

    def apply_content_chunk(self, chunk: str) -> None:
        super().apply_content_chunk(chunk)
        self.chunks.append(chunk)

    def apply_role_start(self, role_start: RoleStart) -> None:
        super().apply_role_start(role_start)
        self.chunks.append(role_start)

    def apply_role_end(self, role_end: RoleEnd) -> None:
        super().apply_role_end(role_end)
        self.chunks.append(role_end)

    def get_state(self) -> list[MessageChunk]:
        return self.chunks


class APIState(BaseState[R]):
    def apply_content_chunk(self, chunk: ContentChunk) -> None:
        match chunk:
            case str(text):
                self.apply_text(text)
            case _:
                if TYPE_CHECKING:
                    assert_never(chunk)
                raise NotImplementedError(f"Chunk type {type(chunk)} not supported")

    @abstractmethod
    def apply_text(self, text: str) -> None:
        pass


class Client(ABC):
    @abstractmethod
    def run(self, state: APIState, node: Node) -> Iterable[ContentChunk]:
        pass


class Model:
    def __init__(
        self,
        client: Client,
        internal_state: InternalState,
        api_state: APIState,
    ) -> None:
        self.client = client
        self._internal_state = internal_state
        self._api_state = api_state

    def __iadd__(self, other: Node) -> Self:
        self._apply_node(other)
        return self

    def __add__(self, other: Node) -> None:
        raise TypeError("Use += to add nodes")

    def _apply_node(self, node: Node) -> None:
        for chunk in self.client.run(self._api_state, node):
            self._apply_chunk(chunk)

    def _apply_chunk(self, chunk: MessageChunk) -> None:
        # Apply to _api_state first, so that it can raise an exception if the chunk is not supported
        self._api_state.apply_chunk(chunk)
        self._internal_state.apply_chunk(chunk)

    @contextmanager
    def role(self, role: str) -> Iterator[None]:
        # _apply_chunk will raise an exception via _api_state.apply_chunk if roles are not supported
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


class ChatState(Generic[M, R], APIState[R], ABC):
    def __init__(self) -> None:
        self.active_role: Optional[RoleStart] = None
        self.messages: list[M] = []

    def apply_content_chunk(self, chunk: ContentChunk) -> None:
        if self.active_role is None:
            raise ValueError(
                "Cannot add to chat model outside of a role block (use model.system(), model.user(), or model.assistant() context managers)"
            )
        super().apply_content_chunk(chunk)

    def apply_role_end(self, role_end: RoleEnd) -> None:
        # Get active message before closing it with apply_role_end and reset_active_message
        active_message = self.get_active_message()
        # Use apply_role_end to hit any exceptions before other state mutations
        super().apply_role_end(role_end)
        self.messages.append(active_message)
        self.reset_active_message()

    @abstractmethod
    def reset_active_message(self) -> None:
        pass

    @abstractmethod
    def get_active_message(self) -> M:
        pass


class CompletionState(APIState[str]):
    def __init__(self) -> None:
        self.state = ""

    def apply_role_start(self, role_start: RoleStart) -> None:
        raise TypeError("Role blocks not supported for completion models")

    def apply_role_end(self, role_end: RoleEnd) -> None:
        raise TypeError("Role blocks not supported for completion models")

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


class OpenAIState(ChatState[OpenAIMessage, list[OpenAIMessage]]):
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
        super().reset_active_message()
        self.content = []
        self.audio = None

    def apply_text(self, text: str) -> None:
        self.content.append({"type": "text", "text": text})

    def get_state(self) -> list[OpenAIMessage]:
        return self.messages


TC = TypeVar("TC")


class TransformersMessage(TypedDict, Generic[TC]):
    role: str
    content: TC


class TransformersChatObj(TypedDict, Generic[TC]):
    messages: list[TransformersMessage[TC]]
    images: list[Any]
    audio: list[Any]
    videos: list[Any]


class BaseTransformersChatState(ChatState[TransformersMessage[TC], TransformersChatObj[TC]]):
    def __init__(self) -> None:
        super().__init__()
        self.content: TC = self._default_content_factory()
        self.images: list[Any] = []
        self.audio: list[Any] = []
        self.videos: list[Any] = []

    def get_active_message(self) -> TransformersMessage[TC]:
        if self.active_role is None:
            raise ValueError("Cannot build message without active role")

        return TransformersMessage({"role": self.active_role.role, "content": self.content})

    def get_state(self) -> TransformersChatObj[TC]:
        return {
            "messages": self.messages,
            "images": self.images,
            "audio": self.audio,
            "videos": self.videos,
        }

    def reset_active_message(self) -> None:
        super().reset_active_message()
        # Don't delete images, audio, or videos, as they are not part of the message
        self.content = self._default_content_factory()

    @abstractmethod
    def _default_content_factory(self) -> TC:
        pass


class TransformersStructuredState(BaseTransformersChatState[list[dict]]):
    def _default_content_factory(self) -> list[dict]:
        return []


class TransformersUnstructuredState(BaseTransformersChatState[str]):
    def _default_content_factory(self) -> str:
        return ""

    def apply_text(self, text: str) -> None:
        if self.content is None:
            self.content = ""
        self.content += text


class Llama3TransformersState(TransformersStructuredState):
    def apply_text(self, text: str) -> None:
        if self.content is None:
            self.content = []
        self.content.append({"type": "text", "text": text})


class DummyClient(Client):
    def run(self, state: APIState, node: Node) -> Iterable[ContentChunk]:
        if isinstance(node, str):
            yield node
        else:
            raise NotImplementedError("Node must be a string")


def chat():
    import json

    for s in [
        OpenAIState,
        TransformersUnstructuredState,
        Llama3TransformersState,
    ]:
        model = Model(DummyClient(), InternalState(), s())
        with model.system():
            model += "Talk like a pirate!"
        with model.user():
            model += "Hello, model!"
            model += "\nHow are you?"
        with model.assistant():
            model += "I'm doing well, thank you!"
        print("-" * 80)
        print(s.__name__)
        print("-" * 80)
        print(json.dumps(model._api_state.get_state(), indent=2))


def completion():
    for s in [
        CompletionState,
    ]:
        model = Model(DummyClient(), InternalState(), s())
        model += "<|system|>\nTalk like a pirate!\n<|end_of_turn|>\n"
        model += "<|user|>\nHello, model!\n<|end_of_turn|>\n"
        model += "<|user|>\nHow are you?\n<|end_of_turn|>\n"
        model += "<|assistant|>\nI'm doing well, thank you!\n<|end_of_turn|>\n"
        print("-" * 80)
        print(s.__name__)
        print("-" * 80)
        print(model._api_state.get_state())
