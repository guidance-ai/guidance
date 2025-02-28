from typing import Optional, TypedDict, Union

from ...trace import CaptureOutput


class CaptureVar(TypedDict):
    value: str
    log_prob: Optional[float]


class State:
    def __init__(self) -> None:
        self.text: str = ""
        self.captures: dict[str, Union[CaptureVar, list[CaptureVar]]] = {}
        self.active_role: Optional[str] = None

    def __str__(self) -> str:
        return self.text

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
