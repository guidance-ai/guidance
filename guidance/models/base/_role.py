from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

from ...experimental.ast import RoleStart

_active_role: ContextVar[Optional["RoleStart"]] = ContextVar("active_role", default=None)


# TODO HN: Add a docstring to better describe arbitrary role functions
@contextmanager
def role(role: str) -> Iterator[None]:
    role_start = RoleStart(role)
    token = _active_role.set(role_start)
    try:
        yield
    finally:
        _active_role.reset(token)
