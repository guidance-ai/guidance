import re
from typing import Any, Optional, Sequence, TypedDict

from ...experimental.ast import ImageBlob
from ..base import ChatState


class TransformersMessage(TypedDict):
    role: str
    content: str


class TransformersChatObj(TypedDict):
    messages: Sequence[TransformersMessage]
    prefill: Optional[TransformersMessage]
    active_role: Optional[str]
    images: list[Any]
    audio: list[Any]
    videos: list[Any]


class TransformersChatState(ChatState[TransformersMessage]):
    def __init__(self) -> None:
        super().__init__()
        self.active_message_content: str = ""
        self.images: list[Any] = []
        self.audio: list[Any] = []
        self.videos: list[Any] = []

    @classmethod
    def from_model_id(cls, model_id: str) -> "TransformersChatState":
        if "Phi-3-vision" in model_id:
            return Phi3VisionState()
        if re.search("Llama-3.*-Vision", model_id):
            return Llama3TransformersState()
        # Fallback
        return TransformersChatState()

    def get_active_message(self) -> Optional[TransformersMessage]:
        if self.active_role is None:
            return None

        return TransformersMessage(
            {"role": self.active_role.role, "content": self.active_message_content}
        )

    def get_state(self) -> TransformersChatObj:
        return {
            "messages": self.messages,
            "prefill": self.get_active_message(),
            "active_role": self.get_active_role(),
            "images": self.images,
            "audio": self.audio,
            "videos": self.videos,
        }

    def reset_active_message(self) -> None:
        super().reset_active_message()
        # Don't delete images, audio, or videos, as they are not part of the message
        self.active_message_content = ""

    def apply_text(self, text: str) -> None:
        self.active_message_content += text


class Llama3TransformersState(TransformersChatState):
    def apply_image(self, image: ImageBlob) -> None:
        self.images.append(image.image)
        self.active_message_content += "<|image|>"


class Phi3VisionState(TransformersChatState):
    def apply_image(self, image: ImageBlob) -> None:
        pil_image = image.image
        if pil_image in self.images:
            ix = self.images.index(pil_image) + 1
        else:
            self.images.append(pil_image)
            ix = len(self.images)
        self.active_message_content += f"<|image_{ix}|>"
