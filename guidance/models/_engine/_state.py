from typing import Any

from .._base import BaseState

class EngineState(BaseState):
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
