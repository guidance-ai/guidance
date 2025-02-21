from guidance.models._engine import EngineState

from ...experimental.ast import ImageBlob


class Llama3(EngineState):
    def apply_image(self, image: ImageBlob) -> None:
        self.images.append(image.image)
        text = "<|image|>"
        EngineState.apply_text(self, text)


class Phi3(EngineState):
    def apply_image(self, image: ImageBlob) -> None:
        pil_image = image.image
        if pil_image in self.images:
            ix = self.images.index(pil_image) + 1
        else:
            self.images.append(pil_image)
            ix = len(self.images)
        EngineState.apply_text(self, f"<|image_{ix}|>")
