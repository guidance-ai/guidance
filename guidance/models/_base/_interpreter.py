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


class Interpreter(Generic[S]):
    def __init__(self, state: S):
        self.state = state

    def run(self, node: ASTNode, **kwargs) -> Iterator[OutputAttr]:
        yield from node.simplify()._run(self, **kwargs)

    def _role_start(self, node: RoleStart, **kwargs) -> Iterator[OutputAttr]:
        if self.state.active_role is not None:
            raise ValueError(
                f"Cannot open role {node.role!r}: {self.state.active_role!r} is already open."
            )
        return self.role_start(node, **kwargs)

    def role_start(self, node: RoleStart, **kwargs) -> Iterator[OutputAttr]:
        raise UnsupportedNodeError(interpreter=self, node=node)

    def _role_end(self, node: RoleEnd, **kwargs) -> Iterator[OutputAttr]:
        if self.state.active_role is None:
            raise ValueError("Cannot close role without active role")
        if self.state.active_role != node.role:
            raise ValueError(f"Cannot close role {node.role!r}: {self.state.active_role!r} is open.")
        return self.role_end(node, **kwargs)

    def role_end(self, node: RoleEnd, **kwargs) -> Iterator[OutputAttr]:
        raise UnsupportedNodeError(interpreter=self, node=node)

    def text(self, node: LiteralNode, **kwargs) -> Iterator[OutputAttr]:
        raise UnsupportedNodeError(interpreter=self, node=node)

    def image_blob(self, node: ImageBlob, **kwargs) -> Iterator[OutputAttr]:
        raise UnsupportedNodeError(interpreter=self, node=node)

    def image_url(self, node: ImageUrl, **kwargs) -> Iterator[OutputAttr]:
        image_bytes = bytes_from(node.url, allow_local=False)
        return self.image_blob(ImageBlob(data=base64.b64encode(image_bytes)), **kwargs)

    def grammar(self, node: GrammarNode, **kwargs) -> Iterator[OutputAttr]:
        raise UnsupportedNodeError(interpreter=self, node=node)

    def regex(self, node: RegexNode, **kwargs) -> Iterator[OutputAttr]:
        return self.grammar(node, **kwargs)

    def select(self, node: SelectNode, **kwargs) -> Iterator[OutputAttr]:
        return self.grammar(node, **kwargs)

    def join(self, node: JoinNode, **kwargs) -> Iterator[OutputAttr]:
        return self.grammar(node, **kwargs)

    def repeat(self, node: RepeatNode, **kwargs) -> Iterator[OutputAttr]:
        return self.grammar(node, **kwargs)

    def substring(self, node: SubstringNode, **kwargs) -> Iterator[OutputAttr]:
        return self.grammar(node, **kwargs)

    def rule(self, node: RuleNode, **kwargs) -> Iterator[OutputAttr]:
        return self.grammar(node, **kwargs)

    def subgrammar(self, node: SubgrammarNode, **kwargs) -> Iterator[OutputAttr]:
        return self.grammar(node, **kwargs)

    def json(self, node: JsonNode, **kwargs) -> Iterator[OutputAttr]:
        return self.grammar(node, **kwargs)

    def lark(self, node: LarkNode, **kwargs) -> Iterator[OutputAttr]:
        return self.grammar(node, **kwargs)

    def audio_blob(self, node: AudioBlob, **kwargs) -> Iterator[OutputAttr]:
        raise UnsupportedNodeError(interpreter=self, node=node)

    def gen_audio(self, node: GenAudio, **kwargs) -> Iterator[OutputAttr]:
        raise UnsupportedNodeError(interpreter=self, node=node)


class UnsupportedNodeError(ValueError):
    def __init__(self, interpreter: Interpreter, node: ASTNode):
        super().__init__(f"{interpreter} does not support {node!r} of type {type(node)}")
        self.interpreter = interpreter
        self.node = node
