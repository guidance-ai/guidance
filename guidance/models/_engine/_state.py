from typing import Any

from ..._types import TokenUsage
from .._base import State


class EngineState(State):
    def __init__(self) -> None:
        # Initialize with zero token usage rather than default None
        # since engine can fast-forward tokens
        super().__init__(token_usage=TokenUsage(ff_tokens=0))
        self.prompt: str = ""
        self.images: list[Any] = []
        self.audio: list[Any] = []
        self.videos: list[Any] = []

    def __str__(self) -> str:
        return self.prompt
