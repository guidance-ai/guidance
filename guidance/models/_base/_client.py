from typing import Generic, Iterator, TypeVar

from ..._ast import (
    ASTNode,
    GrammarNode,
    ImageNode,
    JoinNode,
    JsonNode,
    LiteralNode,
    RegexNode,
    RepeatNode,
    RoleEnd,
    RoleStart,
    RuleNode,
    SelectNode,
    SubgrammarNode,
    SubstringNode,
)
from ...trace import OutputAttr, RoleOpenerInput, TextOutput
from ._state import State

S = TypeVar("S", bound=State)


class Client(Generic[S]):
    def run(self, state: S, node: ASTNode, **kwargs) -> Iterator[OutputAttr]:
        node = node.simplify()
        for attr in node._run(self, state, **kwargs):
            yield attr
            if isinstance(attr, RoleOpenerInput):
                # TODO: this is a hotfix / workaround -- the vis front-end expects a string corresponding
                # to the just-opened role.
                yield TextOutput(value=attr.text, input=True)

    def _role_start(self, state: S, node: RoleStart, **kwargs) -> Iterator[OutputAttr]:
        if state.active_role is not None:
            raise ValueError(
                f"Cannot open role {node.role!r}: {state.active_role!r} is already open."
            )
        return self.role_start(state, node, **kwargs)

    def role_start(self, state: S, node: RoleStart, **kwargs) -> Iterator[OutputAttr]:
        raise UnsupportedNodeError(client=self, node=node)

    def _role_end(self, state: S, node: RoleEnd, **kwargs) -> Iterator[OutputAttr]:
        if state.active_role is None:
            raise ValueError("Cannot close role without active role")
        if state.active_role != node.role:
            raise ValueError(f"Cannot close role {node.role!r}: {state.active_role!r} is open.")
        return self.role_end(state, node, **kwargs)

    def role_end(self, state: S, node: RoleEnd, **kwargs) -> Iterator[OutputAttr]:
        raise UnsupportedNodeError(client=self, node=node)

    def text(self, state: S, node: LiteralNode, **kwargs) -> Iterator[OutputAttr]:
        state.text += node.value
        yield TextOutput(value=node.value, input=True)

    def image(self, state: S, node: ImageNode, **kwargs) -> Iterator[OutputAttr]:
        raise UnsupportedNodeError(client=self, node=node)

    def grammar(self, state: S, node: GrammarNode, **kwargs) -> Iterator[OutputAttr]:
        raise UnsupportedNodeError(client=self, node=node)

    def regex(self, state: S, node: RegexNode, **kwargs) -> Iterator[OutputAttr]:
        return self.grammar(state, node, **kwargs)

    def select(self, state: S, node: SelectNode, **kwargs) -> Iterator[OutputAttr]:
        return self.grammar(state, node, **kwargs)

    def join(self, state: S, node: JoinNode, **kwargs) -> Iterator[OutputAttr]:
        return self.grammar(state, node, **kwargs)

    def repeat(self, state: S, node: RepeatNode, **kwargs) -> Iterator[OutputAttr]:
        return self.grammar(state, node, **kwargs)

    def substring(self, state: S, node: SubstringNode, **kwargs) -> Iterator[OutputAttr]:
        return self.grammar(state, node, **kwargs)

    def rule(self, state: S, node: RuleNode, **kwargs) -> Iterator[OutputAttr]:
        return self.grammar(state, node, **kwargs)

    def subgrammar(self, state: S, node: SubgrammarNode, **kwargs) -> Iterator[OutputAttr]:
        return self.grammar(state, node, **kwargs)

    def json(self, state: S, node: JsonNode, **kwargs) -> Iterator[OutputAttr]:
        return self.grammar(state, node, **kwargs)


class UnsupportedNodeError(ValueError):
    def __init__(self, client: Client, node: ASTNode):
        super().__init__(f"{client} does not support {node!r} of type {type(node)}")
        self.client = client
        self.node = node
