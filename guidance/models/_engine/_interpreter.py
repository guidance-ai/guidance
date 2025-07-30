import warnings
from base64 import b64decode, b64encode
from copy import deepcopy
from io import BytesIO
from typing import TYPE_CHECKING, Iterator, Optional

from ..._ast import GrammarNode, ImageBlob, LiteralNode, RoleEnd, RoleStart, ToolCallNode
from ..._schema import GenTokenExtra, TokenUsage
from ..._utils import partial_decode, recode_special_tokens, text_to_grammar, to_utf8_or_bytes_string
from ...trace import Backtrack, ImageOutput, OutputAttr, Token, TokenOutput
from .._base import Interpreter
from ._engine import Engine
from ._state import EngineState

if TYPE_CHECKING:
    from ...tools import ToolCallHandler


class EngineInterpreter(Interpreter[EngineState]):
    def __init__(self, engine: Engine, tool_call_handler_cls: Optional[type["ToolCallHandler"]] = None):
        super().__init__(state=EngineState())
        self.engine = engine
        if not issubclass(tool_call_handler_cls, ToolCallHandler):
            if isinstance(tool_call_handler_cls, ToolCallHandler):
                raise TypeError(
                    f"tool_call_handler_cls must be a subclass of ToolCallHandler, got instance {tool_call_handler_cls}"
                )
            raise TypeError(f"tool_call_handler_cls must be a subclass of ToolCallHandler, got {tool_call_handler_cls}")
        self.tool_call_handler_cls = tool_call_handler_cls
        self.chat_template = self.engine.get_chat_template()

    def __deepcopy__(self, memo):
        """Custom deepcopy to ensure engine is not copied."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "engine":
                # Don't copy the engine
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    def get_role_start(self, role: str) -> str:
        if self.chat_template is None:
            raise ValueError("Cannot use roles without a chat template")
        return self.chat_template.get_role_start(role)

    def get_role_end(self, role: str) -> str:
        if self.chat_template is None:
            raise ValueError("Cannot use roles without a chat template")
        return self.chat_template.get_role_end(role)

    def role_start(self, node: RoleStart, **kwargs) -> Iterator[OutputAttr]:
        self.state.active_role = node.role
        text = self.get_role_start(node.role)
        # TODO: it's probably somewhat wasteful to trigger engine calls here,
        # so we can maybe add this as "pending text" to the state instead,
        # accumulating it until the next engine call..?
        yield from self.run(text_to_grammar(self.engine.tokenizer, text))

    def role_end(self, node: RoleEnd, **kwargs) -> Iterator[OutputAttr]:
        self.state.active_role = None
        text = self.get_role_end(node.role)
        # TODO: it's probably somewhat wasteful to trigger engine calls here,
        # so we can maybe add this as "pending text" to the state instead,
        # accumulating it until the next engine call..?
        yield from self.run(text_to_grammar(self.engine.tokenizer, text))

    def text(self, node: LiteralNode, **kwargs) -> Iterator[OutputAttr]:
        yield from self.grammar(node, **kwargs)

    def grammar(self, node: GrammarNode, **kwargs) -> Iterator[OutputAttr]:
        engine_gen = self.engine(
            state=self.state,
            grammar=node.ll_grammar(),
            ensure_bos_token=True,
            sampling_params=kwargs.pop("sampling_params", None),
        )
        delayed_bytes = b""
        while True:
            try:
                chunk = next(engine_gen)
            except StopIteration as e:
                if not isinstance(e.value, TokenUsage):
                    raise e
                self.state.add_usage(e.value)
                break

            new_bytes = recode_special_tokens(self.engine.tokenizer, chunk.new_bytes)
            new_text, delayed_bytes = partial_decode(delayed_bytes + new_bytes)
            self.state.prompt += new_text

            if chunk.backtrack:
                yield Backtrack(
                    n_tokens=chunk.backtrack,
                    bytes=b64encode(chunk.backtrack_bytes),
                )

            for token in chunk.tokens:
                if isinstance(token, GenTokenExtra):
                    top_k = [
                        Token(
                            token=to_utf8_or_bytes_string(t.bytes),
                            bytes=b64encode(t.bytes),
                            prob=t.prob,
                            masked=t.is_masked,
                        )
                        for t in token.top_k
                    ]
                else:
                    top_k = None

                token_value = to_utf8_or_bytes_string(token.bytes)
                yield TokenOutput(
                    value=token_value,
                    token=Token(token=token_value, bytes=b64encode(token.bytes), prob=token.prob),
                    latency_ms=token.latency_ms,
                    is_input=token.is_input,
                    is_generated=token.is_generated,
                    is_force_forwarded=token.is_force_forwarded,
                    top_k=top_k,
                )
                if token.is_backtracked:
                    yield Backtrack(
                        n_tokens=1,
                        bytes=b64encode(token.bytes),
                    )

            for name in chunk.capture_groups.keys():
                values = chunk.capture_groups[name]
                log_probs = chunk.capture_group_log_probs[name]
                if isinstance(values, list):
                    assert isinstance(log_probs, list)
                    assert len(values) == len(log_probs)
                    for value, log_prob in zip(values, log_probs):
                        yield self.state.apply_capture(name, value.decode("utf-8"), log_prob=log_prob, is_append=True)
                else:
                    assert isinstance(log_probs, float)
                    yield self.state.apply_capture(name, values.decode("utf-8"), log_prob=log_probs, is_append=False)

        if delayed_bytes:
            raise RuntimeError("Shouldn't have any delayed bytes left...")

    def tool_call(self, node: ToolCallNode, **kwargs) -> Iterator[OutputAttr]:
        if self.tool_call_handler_cls is None:
            from ...tools import LegacyToolCallHandler

            tool_call_handler_cls = LegacyToolCallHandler
            warnings.warn(
                "Tool calling without a ToolCallHandler is deprecated and will be removed in a future version."
                "Please specify a ToolCallHandler subclass in the model's constructor via the `tool_call_handler` argument.",
                DeprecationWarning,
            )
        else:
            tool_call_handler_cls = self.tool_call_handler_cls

        from uuid import uuid4

        from guidance import capture
        from guidance._ast import LiteralNode

        capture_id = f"_tool_call_{uuid4().hex}"
        handler = tool_call_handler_cls(tools=node.tools)
        grm = handler.build_grammar()

        yield from self.run(capture(grm, name=capture_id))

        tool_call_text = self.state.captures[capture_id]["value"]
        tool_calls = handler.parse_tool_calls(tool_call_text)
        if len(tool_calls) > 1 and not node.parallel_tool_calls:
            raise ValueError("Multiple tool calls detected, but parallel_tool_calls is set to False. ")
        if tool_calls:
            for tool_call in tool_calls:
                response = handler.invoke_tool(tool_call)
                yield from self.run(LiteralNode(handler.format_return_value(response)))


class Llama3VisionInterpreter(EngineInterpreter):
    def image_blob(self, node: ImageBlob, **kwargs) -> Iterator[OutputAttr]:
        try:
            import PIL.Image
        except ImportError as ie:
            raise Exception(
                "Please install the Pillow package `pip install Pillow` in order to use images with Llama3!"
            ) from ie

        image_bytes = b64decode(node.data)
        pil_image = PIL.Image.open(BytesIO(image_bytes))
        self.state.images.append(pil_image)
        self.state.prompt += "<|image|>"

        yield ImageOutput(value=node.data, is_input=True)


class Phi3VisionInterpreter(EngineInterpreter):
    def image_blob(self, node: ImageBlob, **kwargs) -> Iterator[OutputAttr]:
        try:
            import PIL.Image
        except ImportError as ie:
            raise Exception(
                "Please install the Pillow package `pip install Pillow` in order to use images with Llama3!"
            ) from ie

        image_bytes = b64decode(node.data)
        pil_image = PIL.Image.open(BytesIO(image_bytes))

        if pil_image in self.state.images:
            ix = self.state.images.index(pil_image) + 1
        else:
            self.state.images.append(pil_image)
            ix = len(self.state.images)
        self.state.prompt += f"<|image_{ix}|>"

        yield ImageOutput(value=node.data, is_input=True)
