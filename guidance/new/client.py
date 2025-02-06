from abc import ABC, abstractmethod
from typing import Iterable

from .ast import ContentChunk, Node
from .state import APIState


class Client(ABC):
    @abstractmethod
    def run(self, state: APIState, node: Node) -> Iterable[ContentChunk]:
        pass
