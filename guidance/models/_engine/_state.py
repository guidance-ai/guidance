import base64
from io import BytesIO
from typing import Any

from ...trace._trace import ImageOutput
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
    def apply_image(self, image: ImageOutput) -> None:
        try:
            import PIL.Image
        except ImportError:
            raise Exception(
                "Please install the Pillow package `pip install Pillow` in order to use images with Llama3!"
            )

        image_bytes = base64.b64decode(image.value)
        pil_image = PIL.Image.open(BytesIO(image_bytes))
        self.images.append(pil_image)

        text = "<|image|>"
        EngineState.apply_text(self, text)


class Phi3VisionState(EngineState):
    def apply_image(self, image: ImageOutput) -> None:
        try:
            import PIL.Image
        except ImportError:
            raise Exception(
                "Please install the Pillow package `pip install Pillow` in order to use images with Phi 3!"
            )

        image_bytes = base64.b64decode(image.value)
        pil_image = PIL.Image.open(BytesIO(image_bytes))

        if pil_image in self.images:
            ix = self.images.index(pil_image) + 1
        else:
            self.images.append(pil_image)
            ix = len(self.images)
        EngineState.apply_text(self, f"<|image_{ix}|>")
