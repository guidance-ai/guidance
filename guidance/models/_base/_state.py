from abc import ABC, abstractmethod
from typing import Optional, TypedDict, Union

from ...trace import CaptureOutput


class CaptureVar(TypedDict):
    value: str
    log_prob: Optional[float]


class State(ABC):
    def __init__(self) -> None:
        self.captures: dict[str, Union[CaptureVar, list[CaptureVar]]] = {}
        self.active_role: Optional[str] = None
        self.open_capture_blocks: dict[str, int] = {}

    def open_capture(self, name: str) -> None:
        self.open_capture_blocks[name] = len(str(self))

    def close_capture(self, name: str) -> CaptureOutput:
        start_index = self.open_capture_blocks.pop(name)
        value = str(self)[start_index:]
        return self.apply_capture(
            name=name,
            value=value,
            log_prob=None,
            is_append=False,
        )

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
