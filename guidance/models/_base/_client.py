from abc import ABC, abstractmethod
from typing import Generic, Iterator, TypeVar

from ...experimental.ast import MessageChunk, Node
from ._state import State

S = TypeVar("S", bound=State)


class Client(ABC, Generic[S]):
    @abstractmethod
    def run(self, state: S, node: Node) -> Iterator[MessageChunk]:
        pass
