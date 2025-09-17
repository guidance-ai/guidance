from contextlib import contextmanager

from .._ast import ASTNode, Function
from .._guidance import _in_stateless_context
from ..models._base._model import _active_blocks


class Block:
    def __init__(
        self, name: str | None, opener: str | Function | ASTNode, closer: str | Function | ASTNode
    ):
        self.name = name
        self.opener = opener
        self.closer = closer


@contextmanager
def block(name=None, opener=None, closer=None):
    if _in_stateless_context.get():
        raise RuntimeError("Cannot use roles or other blocks when stateless=True")
    current_blocks = _active_blocks.get()
    new_block = Block(name=name, opener=opener, closer=closer)
    token = _active_blocks.set(current_blocks + (new_block,))
    try:
        yield
    finally:
        _active_blocks.reset(token)
