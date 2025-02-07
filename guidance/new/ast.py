from dataclasses import dataclass, field
from typing import Union
from uuid import UUID, uuid4

from guidance._grammar import GrammarFunction, RawFunction

# TODO
Node = Union[str, GrammarFunction, RawFunction]
ContentChunk = str


@dataclass(frozen=True, slots=True)
class RoleStart:
    role: str
    id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True, slots=True)
class RoleEnd:
    id: UUID


MessageChunk = Union[ContentChunk, RoleStart, RoleEnd]
