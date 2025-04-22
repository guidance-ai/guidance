import base64
from abc import ABC
from typing import Generic, Iterable, AsyncIterable, TypeVar, Union, Optional

from ..._ast import (
    ASTNode,
    AudioBlob,
    Concatenate,
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
    CaptureStart,
    CaptureEnd,
)
from ..._utils import bytes_from
from ...trace import OutputAttr
from ._state import State

S = TypeVar("S", bound=State)

class Interpreter(Generic[S], ABC):
    def __init__(self, state: S):
        self.state = state

    def run(self, node: ASTNode, **kwargs) -> AsyncIterable[OutputAttr]:
        return node.simplify()._run(self, **kwargs)

    async def capture_start(self, node: CaptureStart, **kwargs) -> AsyncIterable[OutputAttr]:
        self.state.open_capture(node.name)
        if False:
            # Yes, this is intentional.
            yield

    async def capture_end(self, node: CaptureEnd, **kwargs) -> AsyncIterable[OutputAttr]:
        yield self.state.close_capture(node.name)

    async def concatenate(self, node: Concatenate, **kwargs) -> AsyncIterable[OutputAttr]:
        buffer: Optional[GrammarNode] = None
        for child in node:
            assert not isinstance(child, Concatenate) # iter should be flat
            if isinstance(child, GrammarNode):
                if buffer is None:
                    buffer = child
                else:
                    buffer = buffer + child
            else:
                if buffer is not None:
                    async for attr in self.run(buffer, **kwargs):
                        yield attr
                    buffer = None
                async for attr in self.run(child, **kwargs):
                    yield attr
        if buffer is not None:
            async for attr in self.run(buffer, **kwargs):
                yield attr

    def _role_start(self, node: RoleStart, **kwargs) -> AsyncIterable[OutputAttr]:
        if self.state.active_role is not None:
            raise ValueError(
                f"Cannot open role {node.role!r}: {self.state.active_role!r} is already open."
            )
        return self.role_start(node, **kwargs)

    def role_start(self, node: RoleStart, **kwargs) -> AsyncIterable[OutputAttr]:
        raise UnsupportedNodeError(interpreter=self, node=node)

    def _role_end(self, node: RoleEnd, **kwargs) -> AsyncIterable[OutputAttr]:
        if self.state.active_role is None:
            raise ValueError("Cannot close role without active role")
        if self.state.active_role != node.role:
            raise ValueError(f"Cannot close role {node.role!r}: {self.state.active_role!r} is open.")
        return self.role_end(node, **kwargs)

    def role_end(self, node: RoleEnd, **kwargs) -> AsyncIterable[OutputAttr]:
        raise UnsupportedNodeError(interpreter=self, node=node)

    def text(self, node: LiteralNode, **kwargs) -> AsyncIterable[OutputAttr]:
        raise UnsupportedNodeError(interpreter=self, node=node)

    def image_blob(self, node: ImageBlob, **kwargs) -> AsyncIterable[OutputAttr]:
        raise UnsupportedNodeError(interpreter=self, node=node)

    def image_url(self, node: ImageUrl, **kwargs) -> AsyncIterable[OutputAttr]:
        # TODO: we should be using something like httpx to fetch the image
        image_bytes = bytes_from(node.url, allow_local=False)
        base64_string = base64.b64encode(image_bytes).decode("utf-8")
        return self.image_blob(ImageBlob(data=base64_string), **kwargs)

    def grammar(self, node: GrammarNode, **kwargs) -> AsyncIterable[OutputAttr]:
        raise UnsupportedNodeError(interpreter=self, node=node)

    def regex(self, node: RegexNode, **kwargs) -> AsyncIterable[OutputAttr]:
        return self.grammar(node, **kwargs)

    def select(self, node: SelectNode, **kwargs) -> AsyncIterable[OutputAttr]:
        return self.grammar(node, **kwargs)

    def join(self, node: JoinNode, **kwargs) -> AsyncIterable[OutputAttr]:
        return self.grammar(node, **kwargs)

    def repeat(self, node: RepeatNode, **kwargs) -> AsyncIterable[OutputAttr]:
        return self.grammar(node, **kwargs)

    def substring(self, node: SubstringNode, **kwargs) -> AsyncIterable[OutputAttr]:
        return self.grammar(node, **kwargs)

    def rule(self, node: RuleNode, **kwargs) -> AsyncIterable[OutputAttr]:
        return self.grammar(node, **kwargs)

    def subgrammar(self, node: SubgrammarNode, **kwargs) -> AsyncIterable[OutputAttr]:
        return self.grammar(node, **kwargs)

    def json(self, node: JsonNode, **kwargs) -> AsyncIterable[OutputAttr]:
        return self.grammar(node, **kwargs)

    def lark(self, node: LarkNode, **kwargs) -> AsyncIterable[OutputAttr]:
        return self.grammar(node, **kwargs)

    def audio_blob(self, node: AudioBlob, **kwargs) -> AsyncIterable[OutputAttr]:
        raise UnsupportedNodeError(interpreter=self, node=node)

    def gen_audio(self, node: GenAudio, **kwargs) -> AsyncIterable[OutputAttr]:
        raise UnsupportedNodeError(interpreter=self, node=node)

class UnsupportedNodeError(ValueError):
    def __init__(self, interpreter: Interpreter, node: ASTNode):
        super().__init__(f"{interpreter} does not support {node!r} of type {type(node)}")
        self.interpreter = interpreter
        self.node = node
