from typing import Any

from guidance.models._engine._state import EngineChatState, EngineCompletionState

from ...experimental.ast import ImageBlob


# Mixins that should work for both EngineChatState and EngineCompletionState
class _Llama3:
    images: list[Any]
    content: str

    def apply_image(self, image: ImageBlob) -> None:
        self.images.append(image.image)
        self.content += "<|image|>"


class _Phi3:
    images: list[Any]
    content: str

    def apply_image(self, image: ImageBlob) -> None:
        pil_image = image.image
        if pil_image in self.images:
            ix = self.images.index(pil_image) + 1
        else:
            self.images.append(pil_image)
            ix = len(self.images)
        self.content += f"<|image_{ix}|>"


class Llama3ChatState(EngineChatState, _Llama3):
    pass


class Llama3CompletionState(EngineCompletionState, _Llama3):
    pass


class Phi3VisionChatState(EngineChatState, _Phi3):
    pass


class Phi3VisionCompletionState(EngineCompletionState, _Phi3):
    pass
