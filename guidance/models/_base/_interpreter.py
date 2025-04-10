import base64
from abc import ABC
from typing import Generic, Iterable, AsyncIterable, TypeVar, Union

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
R = TypeVar("R", bound=Union[Iterable[OutputAttr], AsyncIterable[OutputAttr]])

class BaseInterpreter(Generic[S, R], ABC):
    def __init__(self, state: S):
        self.state = state

    def run(self, node: ASTNode, **kwargs) -> R:
        return node.simplify()._run(self, **kwargs)

    def _role_start(self, node: RoleStart, **kwargs) -> R:
        if self.state.active_role is not None:
            raise ValueError(
                f"Cannot open role {node.role!r}: {self.state.active_role!r} is already open."
            )
        return self.role_start(node, **kwargs)

    def role_start(self, node: RoleStart, **kwargs) -> R:
        raise UnsupportedNodeError(interpreter=self, node=node)

    def _role_end(self, node: RoleEnd, **kwargs) -> R:
        if self.state.active_role is None:
            raise ValueError("Cannot close role without active role")
        if self.state.active_role != node.role:
            raise ValueError(f"Cannot close role {node.role!r}: {self.state.active_role!r} is open.")
        return self.role_end(node, **kwargs)

    def role_end(self, node: RoleEnd, **kwargs) -> R:
        raise UnsupportedNodeError(interpreter=self, node=node)

    def text(self, node: LiteralNode, **kwargs) -> R:
        raise UnsupportedNodeError(interpreter=self, node=node)

    def image_blob(self, node: ImageBlob, **kwargs) -> R:
        raise UnsupportedNodeError(interpreter=self, node=node)

    def image_url(self, node: ImageUrl, **kwargs) -> R:
        image_bytes = bytes_from(node.url, allow_local=False)
        base64_string = base64.b64encode(image_bytes).decode("utf-8")
        return self.image_blob(ImageBlob(data=base64_string), **kwargs)

    def grammar(self, node: GrammarNode, **kwargs) -> R:
        raise UnsupportedNodeError(interpreter=self, node=node)

    def regex(self, node: RegexNode, **kwargs) -> R:
        return self.grammar(node, **kwargs)

    def select(self, node: SelectNode, **kwargs) -> R:
        return self.grammar(node, **kwargs)

    def join(self, node: JoinNode, **kwargs) -> R:
        return self.grammar(node, **kwargs)

    def repeat(self, node: RepeatNode, **kwargs) -> R:
        return self.grammar(node, **kwargs)

    def substring(self, node: SubstringNode, **kwargs) -> R:
        return self.grammar(node, **kwargs)

    def rule(self, node: RuleNode, **kwargs) -> R:
        return self.grammar(node, **kwargs)

    def subgrammar(self, node: SubgrammarNode, **kwargs) -> R:
        return self.grammar(node, **kwargs)

    def json(self, node: JsonNode, **kwargs) -> R:
        return self.grammar(node, **kwargs)

    def lark(self, node: LarkNode, **kwargs) -> R:
        return self.grammar(node, **kwargs)

    def audio_blob(self, node: AudioBlob, **kwargs) -> R:
        raise UnsupportedNodeError(interpreter=self, node=node)

    def gen_audio(self, node: GenAudio, **kwargs) -> R:
        raise UnsupportedNodeError(interpreter=self, node=node)

class Interpreter(BaseInterpreter[S, Iterable[OutputAttr]]):
    pass

class AsyncInterpreter(BaseInterpreter[S, AsyncIterable[OutputAttr]]):
    pass

class UnsupportedNodeError(ValueError):
    def __init__(self, interpreter: BaseInterpreter, node: ASTNode):
        super().__init__(f"{interpreter} does not support {node!r} of type {type(node)}")
        self.interpreter = interpreter
        self.node = node
