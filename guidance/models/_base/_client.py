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
from ...trace import ImageOutput, TextOutput
from ._state import MessageChunk, State

S = TypeVar("S", bound=State)


class Client(Generic[S]):
    def run(self, state: S, node: ASTNode, **kwargs) -> Iterator[MessageChunk]:
        return node.simplify()._run(self, state, **kwargs)

    def role_start(self, state: S, node: RoleStart, **kwargs) -> Iterator[MessageChunk]:
        raise UnsupportedNodeError(client=self, node=node)

    def role_end(self, state: S, node: RoleEnd, **kwargs) -> Iterator[MessageChunk]:
        raise UnsupportedNodeError(client=self, node=node)

    def literal_str(self, state: S, node: LiteralNode, **kwargs) -> Iterator[MessageChunk]:
        yield TextOutput(value=node.value)

    def literal_image(self, state: S, node: ImageNode, **kwargs) -> Iterator[MessageChunk]:
        yield ImageOutput(value=node.value, is_input=True)

    def grammar(self, state: S, node: GrammarNode, **kwargs) -> Iterator[MessageChunk]:
        raise UnsupportedNodeError(client=self, node=node)

    def regex(self, state: S, node: RegexNode, **kwargs) -> Iterator[MessageChunk]:
        return self.grammar(state, node, **kwargs)

    def select(self, state: S, node: SelectNode, **kwargs) -> Iterator[MessageChunk]:
        return self.grammar(state, node, **kwargs)

    def join(self, state: S, node: JoinNode, **kwargs) -> Iterator[MessageChunk]:
        return self.grammar(state, node, **kwargs)

    def repeat(self, state: S, node: RepeatNode, **kwargs) -> Iterator[MessageChunk]:
        return self.grammar(state, node, **kwargs)

    def substring(self, state: S, node: SubstringNode, **kwargs) -> Iterator[MessageChunk]:
        return self.grammar(state, node, **kwargs)

    def rule(self, state: S, node: RuleNode, **kwargs) -> Iterator[MessageChunk]:
        return self.grammar(state, node, **kwargs)

    def subgrammar(self, state: S, node: SubgrammarNode, **kwargs) -> Iterator[MessageChunk]:
        return self.grammar(state, node, **kwargs)

    def json(self, state: S, node: JsonNode, **kwargs) -> Iterator[MessageChunk]:
        return self.grammar(state, node, **kwargs)


class UnsupportedNodeError(ValueError):
    def __init__(self, client: Client, node: ASTNode):
        super().__init__(f"{client} does not support {node!r} of type {type(node)}")
        self.client = client
        self.node = node
