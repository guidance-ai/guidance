from abc import ABC, abstractmethod
from typing import Optional, TypedDict, Union

from ...trace import CaptureOutput
from ..._schema import TokenUsage
from ...metrics import emit_usage


class CaptureVar(TypedDict):
    value: str
    log_prob: Optional[float]


class State(ABC):
    def __init__(self, token_usage: Optional[TokenUsage] = None) -> None:
        self.captures: dict[str, Union[CaptureVar, list[CaptureVar]]] = {}
        self.active_role: Optional[str] = None
        self._token_usage: TokenUsage = token_usage or TokenUsage()

    def add_usage(self, usage: TokenUsage) -> None:
        """Add token usage to the current state."""
        self._token_usage += usage
        # TODO: need to do this conditionally? Or is this essentially zero-cost?
        emit_usage(self._token_usage)

    def get_usage(self) -> TokenUsage:
        """Get the current token usage."""
        return self._token_usage

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
