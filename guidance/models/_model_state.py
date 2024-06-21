from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import base64
import html

@dataclass
class Object:
    def _html(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError


@dataclass
class Text(Object):
    text: str
    probability: Optional[float] = None

    def __str__(self) -> str:
        return self.text

    def _html(self) -> str:
        escaped_text = html.escape(self.text)
        if self.probability is not None:
            style = f"background-color: rgba({165*(1-self.probability) + 0}, {165*self.probability + 0}, 0, {0.15}); border-radius: 3px;"
            return f"<span style='{style}' title='{self.probability}'>{escaped_text}</span>"
        return escaped_text


@dataclass
class Image(Object):
    id: str
    data: bytes

    def __str__(self) -> str:
        raise NotImplementedError
    
    def _html(self) -> str:
        return f"""<img src="data:image/png;base64,'{base64.b64encode(self.data).decode()}'" style="max-width: 400px; vertical-align: middle; margin: 4px;">"""


class ModelState(list[Object]):
    # TODO: opened blocks
    def __str__(self) -> str:
        out = ""
        for obj in self:
            out += str(obj)
        return out

    def _html(self) -> str:
        out = "<pre style='margin: 0px; padding: 0px; vertical-align: middle; padding-left: 8px; margin-left: -8px; border-radius: 0px; border-left: 1px solid rgba(127, 127, 127, 0.2); white-space: pre-wrap; font-family: ColfaxAI, Arial; font-size: 15px; line-height: 23px;'>"
        for obj in self:
            out += obj._html()
        out += "</pre>"
        return out
