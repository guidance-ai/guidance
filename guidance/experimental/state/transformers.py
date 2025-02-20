import re
from typing import Any, Optional, Sequence, TypedDict

from ..ast import ImageBlob
from ...models._base import BaseChatState


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


class BaseTransformersChatState(BaseChatState[TransformersMessage]):
    def __init__(self) -> None:
        super().__init__()
        self.content: str = ""
        self.images: list[Any] = []
        self.audio: list[Any] = []
        self.videos: list[Any] = []

    @classmethod
    def from_model_id(cls, model_id: str) -> "BaseTransformersChatState":
        if "Phi-3-vision" in model_id:
            return Phi3VisionState()
        if re.search("Llama-3.*-Vision", model_id):
            return Llama3TransformersState()
        # Fallback
        return BaseTransformersChatState()

    def get_active_message(self) -> Optional[TransformersMessage]:
        if self.active_role is None:
            return None

        return TransformersMessage({"role": self.active_role.role, "content": self.content})

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
        self.content = ""

    def apply_text(self, text: str) -> None:
        self.content += text


class Llama3TransformersState(BaseTransformersChatState):
    def apply_image(self, image: ImageBlob) -> None:
        self.images.append(image.image)
        self.content += "<|image|>"


class Phi3VisionState(BaseTransformersChatState):
    def apply_image(self, image: ImageBlob) -> None:
        pil_image = image.image
        if pil_image in self.images:
            self.content += f"<|image_{self.images.index(pil_image) + 1}|>"
        else:
            self.images.append(image.image)
            self.content += f"<|image_{len(self.images)}|>"
