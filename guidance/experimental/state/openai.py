from typing import Optional, TypedDict, Union

from .base import BaseChatStateObj, ChatState


class OpenAIContentMessage(TypedDict):
    role: str
    content: list[dict]


class OpenAIAudioMessage(TypedDict):
    role: str
    audio: dict


OpenAIMessage = Union[OpenAIContentMessage, OpenAIAudioMessage]


class OpenAIState(ChatState[OpenAIMessage]):
    def __init__(self) -> None:
        super().__init__()
        self.content: list[dict] = []
        self.audio: Optional[dict] = None

    def get_active_message(self) -> Optional[OpenAIMessage]:
        if self.active_role is None:
            return None

        if self.audio:
            if self.content:
                raise ValueError("Expected either content or audio in OpenAI message, not both")
            return OpenAIAudioMessage({"role": self.active_role.role, "audio": self.audio})

        if self.content:
            return OpenAIContentMessage({"role": self.active_role.role, "content": self.content})

        return None

    def reset_active_message(self) -> None:
        super().reset_active_message()
        self.content = []
        self.audio = None

    def apply_text(self, text: str) -> None:
        self.content.append({"type": "text", "text": text})
