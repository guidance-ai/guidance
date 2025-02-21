from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Sequence, TypedDict, Union

from typing_extensions import Self, assert_never
from ...trace import (
    CaptureOutput,
    LiteralInput,
    TextOutput,
    RoleCloserInput,
    RoleOpenerInput,
)
from ...experimental.ast import (
    ContentChunk,
    ImageBlob,
    MessageChunk,
)

class Message(TypedDict):
    role: Optional[str]
    data: dict[str, Any]

class CaptureVar(TypedDict):
    value: str
    log_prob: Optional[float]

class BaseState(ABC):
    def __init__(self) -> None:
        self.chunks: list[MessageChunk] = []
        self.captures: dict[str, Union[CaptureVar, list[CaptureVar]]] = {}
        self.text: str = ""
        self.messages: list[Message] = []
        self.active_message: Message = {"role": None, "data": {}}

    def __str__(self) -> str:
        return self.text

    @classmethod
    def from_chunks(cls, chunks: Sequence[MessageChunk]) -> Self:
        self = cls()
        for chunk in chunks:
            self.apply_chunk(chunk)
        return self

    def apply_chunk(self, chunk: MessageChunk) -> None:
        self.chunks.append(chunk)
        if isinstance(chunk, RoleOpenerInput):
            self.apply_role_start(chunk)
        elif isinstance(chunk, RoleCloserInput):
            self.apply_role_end(chunk)
        elif isinstance(chunk, CaptureOutput):
            if chunk.value is None:
                # A "reset" signal
                self.captures.pop(chunk.name)
            else:
                var = CaptureVar(value=chunk.value, log_prob=chunk.log_probs)
                if chunk.is_append:
                    vars = self.captures.get(chunk.name, [])
                    if not isinstance(vars, list):
                        vars = [vars]
                    vars.append(var)
                    self.captures[chunk.name] = vars
                else:
                    self.captures[chunk.name] = var
        else:
            self.apply_content_chunk(chunk)

    def apply_content_chunk(self, chunk: ContentChunk) -> None:
        if isinstance(chunk, (LiteralInput, TextOutput)):
            self.apply_text(chunk.value)
        elif isinstance(chunk, ImageBlob):
            self.apply_image(chunk)
        else:
            if TYPE_CHECKING:
                assert_never(chunk)
            raise NotImplementedError(f"Chunk type {type(chunk)} not supported")

    def apply_role_start(self, role_opener: RoleOpenerInput) -> None:
        active_message = self.active_message
        active_role = active_message["role"]
        if active_role is not None:
            raise ValueError(
                f"Cannot open role {role_opener.name!r}: {active_role!r} is already open."
            )
        if active_message["data"]:
            # There's an active message without a role... let's just trust the user even if this bites us
            self.messages = (*self.messages, active_message)
        self.active_message = {"role": role_opener.name, "data": {}}
        self.text += role_opener.text

    def apply_role_end(self, role_closer: RoleCloserInput) -> None:
        active_message = self.active_message
        active_role = active_message["role"]
        if active_role is None:
            raise ValueError("Cannot close role without active role")
        if active_role != role_closer.name:
            raise ValueError(
                f"Cannot close role {role_closer.name!r}: {active_role!r} is open."
            )
        self.messages = (*self.messages, active_message)
        self.active_message = {"role": None, "data": {}} 
        self.text += role_closer.text

    @abstractmethod
    def apply_text(self, text: str) -> None:
        pass

    def apply_image(self, image: ImageBlob) -> None:
        # TODO: raise custom exception so we can catch it and raise a better error
        # where we have the model's name, etc.
        raise TypeError(f"Image blobs not supported by {self.__class__.__name__}")
