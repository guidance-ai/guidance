from typing import Any, Optional, Sequence, TypedDict, Union

from .._base import BaseChatState, BaseCompletionState


class EngineCompletionPrompt(TypedDict):
    prompt: str
    images: list[Any]
    audio: list[Any]
    videos: list[Any]


class EngineCompletionState(BaseCompletionState):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[Any] = []
        self.audio: list[Any] = []
        self.videos: list[Any] = []

    def get_prompt(self) -> EngineCompletionPrompt:
        return {
            "prompt": self.content,
            "images": self.images,
            "audio": self.audio,
            "videos": self.videos,
        }


class EngineMessage(TypedDict):
    role: str
    content: str


class EngineChatPrompt(TypedDict):
    prompt: Sequence[EngineMessage]
    images: list[Any]
    audio: list[Any]
    videos: list[Any]


class EngineChatState(BaseChatState):
    def __init__(self) -> None:
        super().__init__()
        self.content: str = ""
        self.images: list[Any] = []
        self.audio: list[Any] = []
        self.videos: list[Any] = []

    def get_active_message(self) -> Optional[EngineMessage]:
        if self.active_role is None:
            return None

        return EngineMessage({"role": self.active_role.role, "content": self.content})

    def get_prompt(self) -> EngineChatPrompt:
        messages = self.messages
        active_message = self.get_active_message()
        if active_message is not None:
            messages = (*messages, active_message)
        return {
            "prompt": messages,
            "images": self.images,
            "audio": self.audio,
            "videos": self.videos,
        }

    def reset_active_message(self) -> None:
        super().reset_active_message()
        # Don't delete images, audio, or videos, as they are not part of the message
        self.content = ""

    def apply_text(self, text: str) -> None:
        self.content += text


EnginePrompt = Union[EngineChatPrompt, EngineCompletionPrompt]
EngineState = Union[EngineChatState, EngineCompletionState]
