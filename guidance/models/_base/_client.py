from abc import ABC, abstractmethod
from typing import Generic, Iterator, TypeVar

from ..._ast import ASTNode
from ._state import MessageChunk, State

S = TypeVar("S", bound=State)


class Client(ABC, Generic[S]):
    @abstractmethod
    def run(self, state: S, node: ASTNode) -> Iterator[MessageChunk]:
        pass
