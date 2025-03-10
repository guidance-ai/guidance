import base64
from typing import Generic, Iterator, TypeVar

from ..._ast import (
    ASTNode,
    AudioBlob,
    GenAudio,
    GrammarNode,
    ImageBlob,
    ImageUrl,
    JoinNode,
    JsonNode,
    LarkNode,
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
from ..._utils import bytes_from
from ...trace import OutputAttr
from ._state import State

S = TypeVar("S", bound=State)


class Client(Generic[S]):
    def run(self, state: S, node: ASTNode, **kwargs) -> Iterator[OutputAttr]:
        yield from node.simplify()._run(self, state, **kwargs)

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
        raise UnsupportedNodeError(client=self, node=node)

    def image_blob(self, state: S, node: ImageBlob, **kwargs) -> Iterator[OutputAttr]:
        raise UnsupportedNodeError(client=self, node=node)

    def image_url(self, state: S, node: ImageUrl, **kwargs) -> Iterator[OutputAttr]:
        image_bytes = bytes_from(node.url, allow_local=False)
        base64_string = base64.b64encode(image_bytes).decode("utf-8")
        return self.image_blob(state, ImageBlob(data=base64_string), **kwargs)

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

    def lark(self, state: S, node: LarkNode, **kwargs) -> Iterator[OutputAttr]:
        return self.grammar(state, node, **kwargs)

    def audio_blob(self, state: S, node: AudioBlob, **kwargs) -> Iterator[OutputAttr]:
        raise UnsupportedNodeError(client=self, node=node)

    def gen_audio(self, state: S, node: GenAudio, **kwargs) -> Iterator[OutputAttr]:
        raise UnsupportedNodeError(client=self, node=node)


class UnsupportedNodeError(ValueError):
    def __init__(self, client: Client, node: ASTNode):
        super().__init__(f"{client} does not support {node!r} of type {type(node)}")
        self.client = client
        self.node = node
