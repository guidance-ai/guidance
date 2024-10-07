import sys
from typing import (
    Any,
    Callable,
    Literal,
    TypeVar,
    Union,
    overload,
)
if sys.version_info >= (3, 10):
    from typing import ParamSpec, TypeAlias, Concatenate
else:
    from typing_extensions import ParamSpec, TypeAlias, Concatenate

from ._grammar import GrammarFunction, RawFunction
from .models import Model

P = ParamSpec("P")
M: TypeAlias = Any # sort of Union[Model, GrammarFunction]?
R = TypeVar("R", bound = Union[RawFunction, GrammarFunction])
GuidanceWrappable = Callable[Concatenate[M, P], M]
GuidanceFunction = Callable[P, R]
StatefulGuidanceFunction = GuidanceFunction[P, RawFunction]
StatelessGuidanceFunction = GuidanceFunction[P, GrammarFunction]

@overload
def guidance(
    f: GuidanceWrappable[P],
    *,
    stateless: Literal[False] = False,
    cache: bool = ...,
    dedent: bool = ...,
    model: type[Model] = ...,
) -> StatefulGuidanceFunction[P]:
    ...


@overload
def guidance(
    f: None = None,
    *,
    stateless: Literal[False] = False,
    cache: bool = ...,
    dedent: bool = ...,
    model: type[Model] = ...,
) -> Callable[[GuidanceWrappable[P]], StatefulGuidanceFunction[P]]:
    ...


@overload
def guidance(
    f: GuidanceWrappable[P],
    *,
    stateless: Literal[True],
    cache: bool = ...,
    dedent: bool = ...,
    model: type[Model] = ...,
) -> StatelessGuidanceFunction[P]:
    ...


@overload
def guidance(
    f: None = None,
    *,
    stateless: Literal[True],
    cache: bool = ...,
    dedent: bool = ...,
    model: type[Model] = ...,
) -> Callable[[GuidanceWrappable[P]], StatelessGuidanceFunction[P]]:
    ...


@overload
def guidance(
    f: GuidanceWrappable[P],
    *,
    stateless: Callable[..., bool],
    cache: bool = ...,
    dedent: bool = ...,
    model: type[Model] = ...,
) -> GuidanceFunction[P, Union[RawFunction, GrammarFunction]]:
    ...


@overload
def guidance(
    f: None = None,
    *,
    stateless: Callable[..., bool],
    cache: bool = ...,
    dedent: bool = ...,
    model: type[Model] = ...,
) -> Callable[[GuidanceWrappable[P]], GuidanceFunction[P, Union[RawFunction, GrammarFunction]]]:
    ...
