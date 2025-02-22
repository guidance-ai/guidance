from typing import Any

from ...experimental.ast import ImageBlob
from .._base import State


class EngineState(State):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[Any] = []
        self.audio: list[Any] = []
        self.videos: list[Any] = []

    def apply_text(self, text: str) -> None:
        if self.active_message["data"].get("content") is None:
            self.active_message["data"]["content"] = text
        else:
            self.active_message["data"]["content"] += text
        self.text += text


class Llama3VisionState(EngineState):
    def apply_image(self, image: ImageBlob) -> None:
        self.images.append(image.image)
        text = "<|image|>"
        EngineState.apply_text(self, text)


class Phi3VisionState(EngineState):
    def apply_image(self, image: ImageBlob) -> None:
        pil_image = image.image
        if pil_image in self.images:
            ix = self.images.index(pil_image) + 1
        else:
            self.images.append(pil_image)
            ix = len(self.images)
        EngineState.apply_text(self, f"<|image_{ix}|>")
