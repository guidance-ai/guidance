from abc import ABC, abstractmethod
from typing import Any, Generic, TypedDict, TypeVar

from .base import ChatState

# Type of content in a message
TC = TypeVar("TC")


class TransformersMessage(TypedDict, Generic[TC]):
    role: str
    content: TC


class TransformersChatObj(TypedDict, Generic[TC]):
    messages: list[TransformersMessage[TC]]
    images: list[Any]
    audio: list[Any]
    videos: list[Any]


class BaseTransformersChatState(ChatState[TransformersMessage[TC], TransformersChatObj[TC]], ABC):
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
