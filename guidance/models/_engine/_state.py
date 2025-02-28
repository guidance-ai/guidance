from typing import Any

from .._base import State


class EngineState(State):
    def __init__(self) -> None:
        super().__init__()
        self.prompt: str = ""
        self.images: list[Any] = []
        self.audio: list[Any] = []
        self.videos: list[Any] = []

    def __str__(self) -> str:
        return self.prompt
