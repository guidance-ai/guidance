from dataclasses import dataclass, field
from typing import Union
from uuid import UUID, uuid4

# TODO
Node = str
ContentChunk = str


@dataclass(frozen=True, slots=True)
class RoleStart:
    role: str
    id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True, slots=True)
class RoleEnd:
    id: UUID


MessageChunk = Union[ContentChunk, RoleStart, RoleEnd]
