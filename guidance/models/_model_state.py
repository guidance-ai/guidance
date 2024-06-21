from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, overload
import base64
import html


@dataclass(frozen=True, slots=True)
class Object:
    def _html(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class Image(Object):
    id: str
    data: bytes

    def __str__(self) -> str:
        raise NotImplementedError

    def _html(self) -> str:
        return f"""<img src="data:image/png;base64,'{base64.b64encode(self.data).decode()}'" style="max-width: 400px; vertical-align: middle; margin: 4px;">"""


@dataclass(frozen=True, slots=True)
class RoleOpener(Object):
    role_name: str
    text: str
    indent: bool

    def _html(self) -> str:
        out = ""
        if self.indent:
            out += f"<div style='display: flex; border-bottom: 1px solid rgba(127, 127, 127, 0.2);  justify-content: center; align-items: center;'><div style='flex: 0 0 80px; opacity: 0.5;'>{self.role_name.lower()}</div><div style='flex-grow: 1; padding: 5px; padding-top: 10px; padding-bottom: 10px; margin-top: 0px; white-space: pre-wrap; margin-bottom: 0px;'>"
        else:
            out += "<span style='background-color: rgba(255, 180, 0, 0.3); border-radius: 3px;'>"
            out += self.text
            out += "</span>"
        return out

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class RoleCloser(Object):
    role_name: str
    text: str
    indent: bool

    def _html(self) -> str:
        out = ""
        if self.indent:
            out += "</div></div>"
        else:
            out += "<span style='background-color: rgba(255, 180, 0, 0.3); border-radius: 3px;'>"
            out += self.text
            out += "</span>"
        return out

    def __str__(self) -> str:
        return self.text


class ModelState:
    __slots__ = ["objects"]

    def __init__(self) -> None:
        self.objects: list[Object] = []

    def copy(self) -> ModelState:
        new = ModelState()
        new.objects = self.objects.copy()
        return new

    @overload
    def __getitem__(self, index: int) -> Object: ...

    @overload
    def __getitem__(self, index: slice) -> ModelState: ...

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.objects[index]
        elif isinstance(index, slice):
            new = ModelState()
            new.objects = self.objects[index]
            return new
        raise TypeError(f"Index must be int or slice, not {type(index)}")

    def append(self, obj: Object) -> None:
        self.objects.append(obj)

    def __len__(self) -> int:
        return len(self.objects)

    # TODO: opened blocks
    def __str__(self) -> str:
        out = ""
        for obj in self.objects:
            out += str(obj)
        return out

    def _html(self) -> str:
        out = "<pre style='margin: 0px; padding: 0px; vertical-align: middle; padding-left: 8px; margin-left: -8px; border-radius: 0px; border-left: 1px solid rgba(127, 127, 127, 0.2); white-space: pre-wrap; font-family: ColfaxAI, Arial; font-size: 15px; line-height: 23px;'>"
        for obj in self.objects:
            out += obj._html()
        out += "</pre>"
        return out
