from typing import Any
from copy import deepcopy

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
    
    def copy(self) -> "State":
        """Create a copy of the current state."""
        new_state = self.__class__()
        # deepcopy everything using __dict__ to ensure we copy the structure
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                new_state.__dict__[key] = deepcopy(value)
            elif isinstance(value, list):
                new_state.__dict__[key] = [deepcopy(item) for item in value]
            else:
                new_state.__dict__[key] = deepcopy(value)
        return new_state
