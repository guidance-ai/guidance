from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Optional, TypedDict, TypeVar

from typing_extensions import Self, assert_never

from ..ast import ContentChunk, MessageChunk, RoleEnd, RoleStart


class BaseChatMessage(TypedDict):
    role: str


# Return type of BaseState.get_state
R = TypeVar("R")
# Message type of BaseChatState
M = TypeVar("M", bound=BaseChatMessage)


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


class APIState(BaseState[R], ABC):
    @classmethod
    def from_internal_state(cls, internal_state: InternalState) -> Self:
        self = cls()
        for chunk in internal_state.get_state():
            self.apply_chunk(chunk)
        return self

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


class CompletionState(APIState[R]):
    def __init__(self) -> None:
        self.prompt = ""

    def apply_role_start(self, role_start: RoleStart) -> None:
        raise TypeError("Role blocks not supported for completion models")

    def apply_role_end(self, role_end: RoleEnd) -> None:
        raise TypeError("Role blocks not supported for completion models")

    def apply_text(self, text: str) -> None:
        self.prompt += text
