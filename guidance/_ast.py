from typing import Union
from ._grammar import GrammarFunction, RawFunction, RoleEnd, RoleStart
from .trace import ImageOutput


ASTNode = Union[str, GrammarFunction, RawFunction, ImageOutput, RoleStart, RoleEnd]
