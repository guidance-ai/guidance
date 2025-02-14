from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

from .ast import RoleStart

_active_role: ContextVar[Optional["RoleStart"]] = ContextVar("active_role", default=None)


@contextmanager
def role(role: str) -> Iterator[None]:
    # _apply_chunk will raise an exception via _api_state.apply_chunk if roles are not supported
    role_start = RoleStart(role)
    token = _active_role.set(role_start)
    try:
        yield
    finally:
        _active_role.reset(token)


def system() -> AbstractContextManager[None]:
    return role("system")


def user() -> AbstractContextManager[None]:
    return role("user")


def assistant() -> AbstractContextManager[None]:
    return role("assistant")
