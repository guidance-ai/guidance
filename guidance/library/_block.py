from contextlib import contextmanager
from typing import Optional, Union
from .._ast import ASTNode, Function
from ..models._base._model import _active_blocks
from .._guidance import _in_stateless_context

class Block:
    def __init__(self, name: Optional[str], opener: Union[str, Function, ASTNode], closer: Union[str, Function, ASTNode]):
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
