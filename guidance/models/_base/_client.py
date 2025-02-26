from typing import Generic, Iterator, TypeVar

from ..._ast import (
    ASTNode,
    GenNode,
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
from ...trace import ImageOutput, LiteralInput
from ._state import MessageChunk, State

S = TypeVar("S", bound=State)


class Client(Generic[S]):
    def run(self, state: S, node: ASTNode) -> Iterator[MessageChunk]:
        return node.run(self, state)

    def role_start(self, state: S, node: RoleStart) -> Iterator[MessageChunk]:
        raise UnsupportedNodeError(node)

    def role_end(self, state: S, node: RoleEnd) -> Iterator[MessageChunk]:
        raise UnsupportedNodeError(node)

    def literal_str(self, state: S, node: LiteralNode) -> Iterator[MessageChunk]:
        yield LiteralInput(value=node.value)

    def literal_image(self, state: S, node: ImageNode) -> Iterator[MessageChunk]:
        yield ImageOutput(value=node.value, is_input=True)

    def grammar(self, state: S, node: GrammarNode) -> Iterator[MessageChunk]:
        raise UnsupportedNodeError(node)

    def regex(self, state: S, node: RegexNode) -> Iterator[MessageChunk]:
        return self.grammar(state, node)

    def select(self, state: S, node: SelectNode) -> Iterator[MessageChunk]:
        return self.grammar(state, node)

    def join(self, state: S, node: JoinNode) -> Iterator[MessageChunk]:
        return self.grammar(state, node)

    def repeat(self, state: S, node: RepeatNode) -> Iterator[MessageChunk]:
        return self.grammar(state, node)

    def substring(self, state: S, node: SubstringNode) -> Iterator[MessageChunk]:
        return self.grammar(state, node)

    def rule(self, state: S, node: RuleNode) -> Iterator[MessageChunk]:
        return self.grammar(state, node)

    def gen(self, state: S, node: GenNode) -> Iterator[MessageChunk]:
        return self.grammar(state, node)

    def subgrammar(self, state: S, node: SubgrammarNode) -> Iterator[MessageChunk]:
        return self.grammar(state, node)

    def json(self, state: S, node: JsonNode) -> Iterator[MessageChunk]:
        return self.grammar(state, node)


class UnsupportedNodeError(ValueError):
    def __init__(self, client: Client, node: ASTNode):
        super().__init__(f"{client} does not support {node!r} of type {type(node)}")
        self.client = client
        self.node = node
