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

from ._ast import RuleNode, Function
from .models import Model

P = ParamSpec("P")
M: TypeAlias = Any # sort of Union[Model, GrammarNode]?
R = TypeVar("R", bound = Union[Function, RuleNode])
GuidanceWrappable = Callable[Concatenate[M, P], M]
GuidanceFunction = Callable[P, R]
StatefulGuidanceFunction = GuidanceFunction[P, Function]
StatelessGuidanceFunction = GuidanceFunction[P, RuleNode]

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
) -> GuidanceFunction[P, Union[Function, RuleNode]]:
    ...


@overload
def guidance(
    f: None = None,
    *,
    stateless: Callable[..., bool],
    cache: bool = ...,
    dedent: bool = ...,
    model: type[Model] = ...,
) -> Callable[[GuidanceWrappable[P]], GuidanceFunction[P, Union[Function, RuleNode]]]:
    ...
