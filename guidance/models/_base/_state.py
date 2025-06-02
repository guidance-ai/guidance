from abc import ABC, abstractmethod
from typing import Optional, TypedDict, Union
from copy import deepcopy

from ...trace import CaptureOutput


class CaptureVar(TypedDict):
    value: str
    log_prob: Optional[float]


class State(ABC):
    def __init__(self) -> None:
        self.captures: dict[str, Union[CaptureVar, list[CaptureVar]]] = {}
        self.active_role: Optional[str] = None

    @abstractmethod
    def __str__(self) -> str:
        pass

    def apply_capture(
        self, name: str, value: Optional[str], log_prob=Optional[float], is_append: bool = False
    ) -> CaptureOutput:
        if value is None:
            # A "reset" signal
            self.captures.pop(name)
        else:
            var = CaptureVar(value=value, log_prob=log_prob)
            if is_append:
                vars = self.captures.get(name, [])
                if not isinstance(vars, list):
                    vars = [vars]
                vars.append(var)
                self.captures[name] = vars
            else:
                self.captures[name] = var

        return CaptureOutput(
            name=name,
            value=value,
            log_probs=log_prob or float("nan"),
            is_append=is_append,
        )
        
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
