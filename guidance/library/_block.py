from contextlib import contextmanager
from typing import Optional
from .._ast import ASTNode
from ..models._base._model import _active_blocks

class Block:
    def __init__(self, name: Optional[str], opener: Optional[ASTNode], closer: Optional[ASTNode]):
        self.name = name
        self.opener = opener
        self.closer = closer


@contextmanager
def block(name=None, opener=None, closer=None):
    current_blocks = _active_blocks.get()
    new_block = Block(name=name, opener=opener, closer=closer)
    token = _active_blocks.set(current_blocks + (new_block,))
    try:
        yield
    finally:
        _active_blocks.reset(token)
