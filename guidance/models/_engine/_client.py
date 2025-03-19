from base64 import b64decode
from io import BytesIO
from typing import Iterator

from ..._ast import GrammarNode, ImageBlob, LiteralNode, RoleEnd, RoleStart
from ..._utils import to_utf8_or_bytes_string
from ...trace import ImageOutput, OutputAttr, BacktrackMessage, TokenOutput
from .._base import Client
from ._engine import Engine
from ._state import EngineState
from ..._schema import GenTokenExtra


class EngineClient(Client[EngineState]):
    def __init__(self, engine: Engine):
        self.engine = engine
        self.chat_template = self.engine.get_chat_template()

    def get_role_start(self, role: str) -> str:
        if self.chat_template is None:
            raise ValueError("Cannot use roles without a chat template")
        return self.chat_template.get_role_start(role)

    def get_role_end(self, role: str) -> str:
        if self.chat_template is None:
            raise ValueError("Cannot use roles without a chat template")
        return self.chat_template.get_role_end(role)

    def role_start(self, state: EngineState, node: RoleStart, **kwargs) -> Iterator[OutputAttr]:
        state.active_role = node.role
        text = self.get_role_start(node.role)
        yield from self.run(
            state,
            # TODO: special tokens aren't literal text, but this HAPPENS to work (may be fragile)
            LiteralNode(text)
        )

    def role_end(self, state: EngineState, node: RoleEnd, **kwargs) -> Iterator[OutputAttr]:
        state.active_role = None
        text = self.get_role_end(node.role)
        yield from self.run(
            state,
            # TODO: special tokens aren't literal text, but this HAPPENS to work (may be fragile)
            LiteralNode(text)
        )

    def text(self, state: EngineState, node: LiteralNode, **kwargs) -> Iterator[OutputAttr]:
        yield from self.grammar(state, node, **kwargs)

    def grammar(self, state: EngineState, node: GrammarNode, **kwargs) -> Iterator[OutputAttr]:
        engine_gen = self.engine(
            state,
            node.ll_grammar(),
            ensure_bos_token=True,
            echo=False,
        )

        delayed_bytes = b""
        for chunk in engine_gen:
            new_text, delayed_bytes = partial_decode(delayed_bytes + chunk.new_bytes)
            state.prompt += new_text

            if chunk.backtrack:
                yield BacktrackMessage(
                    n_tokens=chunk.backtrack,
                    bytes=chunk.backtrack_bytes,
                )

            for token in chunk.tokens:
                if isinstance(token, GenTokenExtra):
                    top_k = [
                        {
                            "token": to_utf8_or_bytes_string(t.bytes),
                            "bytes": t.bytes,
                            "prob": t.prob,
                            "is_masked": t.is_masked,
                        }
                        for t in token.top_k
                    ]
                else:
                    top_k = None

                token_value = to_utf8_or_bytes_string(token.bytes)
                yield TokenOutput(
                    value=token_value,
                    token={"token": token_value, "bytes": token.bytes, "prob": token.prob},
                    latency_ms=token.latency_ms,
                    is_input = token.is_input,
                    is_generated = token.is_generated,
                    is_force_forwarded = token.is_force_forwarded,
                    top_k=top_k,
                )
                if token.is_backtracked:
                    yield BacktrackMessage(
                        n_tokens=1,
                        bytes=token.bytes,
                    )

            for name in chunk.capture_groups.keys():
                values = chunk.capture_groups[name]
                log_probs = chunk.capture_group_log_probs[name]
                if isinstance(values, list):
                    assert isinstance(log_probs, list)
                    assert len(values) == len(log_probs)
                    for value, log_prob in zip(values, log_probs):
                        yield state.apply_capture(
                            name, value.decode("utf-8"), log_prob=log_prob, is_append=True
                        )
                else:
                    assert isinstance(log_probs, float)
                    yield state.apply_capture(
                        name, values.decode("utf-8"), log_prob=log_probs, is_append=False
                    )

        if delayed_bytes:
            raise RuntimeError("Shouldn't have any delayed bytes left...")


class Llama3VisionClient(EngineClient):
    def image_blob(self, state: EngineState, node: ImageBlob, **kwargs) -> Iterator[OutputAttr]:
        try:
            import PIL.Image
        except ImportError:
            raise Exception(
                "Please install the Pillow package `pip install Pillow` in order to use images with Llama3!"
            )

        image_bytes = b64decode(node.data)
        pil_image = PIL.Image.open(BytesIO(image_bytes))
        state.images.append(pil_image)
        state.prompt += "<|image|>"

        yield ImageOutput(value=node.data, input=True)


class Phi3VisionClient(EngineClient):
    def image_blob(self, state: EngineState, node: ImageBlob, **kwargs) -> Iterator[OutputAttr]:
        try:
            import PIL.Image
        except ImportError:
            raise Exception(
                "Please install the Pillow package `pip install Pillow` in order to use images with Llama3!"
            )

        image_bytes = b64decode(node.data)
        pil_image = PIL.Image.open(BytesIO(image_bytes))

        if pil_image in state.images:
            ix = state.images.index(pil_image) + 1
        else:
            state.images.append(pil_image)
            ix = len(state.images)
        state.prompt += f"<|image_{ix}|>"

        yield ImageOutput(value=node.data, input=True)


def partial_decode(data: bytes) -> tuple[str, bytes]:
    try:
        return (data.decode("utf-8"), b"")
    except UnicodeDecodeError as e:
        valid_part = data[: e.start].decode("utf-8")
        delayed_part = data[e.start :]
    return (valid_part, delayed_part)
