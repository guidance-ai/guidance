from base64 import b64decode
from io import BytesIO
from typing import Iterator

from ..._ast import GrammarNode, ImageBlob, LiteralNode, RoleEnd, RoleStart
from ...trace import ImageOutput, OutputAttr, TextOutput
from .._base import Client
from ._engine import Engine
from ._state import EngineState


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
        # TODO: mark these as special tokens / apply as grammar so we get token probs
        text = self.get_role_start(node.role)
        state.prompt += text
        yield TextOutput(value=text, is_input=True)

    def role_end(self, state: EngineState, node: RoleEnd, **kwargs) -> Iterator[OutputAttr]:
        state.active_role = None
        # TODO: mark these as special tokens / apply as grammar so we get token probs
        text = self.get_role_end(node.role)
        state.prompt += text
        yield TextOutput(value=text, is_input=True)

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
        new_bytes_buffer = b""
        backtracked_bytes_buffer = b""
        token_buffer = []
        for chunk in engine_gen:
            if chunk.backtrack_bytes:
                assert len(new_bytes_buffer) == 0  # otherwise we'll be out of order...
                backtracked_bytes_buffer += chunk.backtrack_bytes
            new_bytes_buffer += chunk.new_bytes
            token_buffer += chunk.tokens

            i = 0
            for token in token_buffer:
                if (backtracked_bytes_buffer + new_bytes_buffer).startswith(token.bytes):
                    data = {
                        "token_id": token.token_id,
                        "bytes": token.bytes,
                        "new_bytes": token.bytes[len(backtracked_bytes_buffer) :],
                    }
                    # Update the state
                    new_text, delayed_bytes = partial_decode(delayed_bytes + data["new_bytes"])
                    state.prompt += new_text
                    # Emit a message
                    yield TextOutput(
                        value=new_text,
                        is_input=token.is_input,
                        is_generated=token.is_generated,
                        is_force_forwarded=token.is_force_forwarded,
                        token_count=1,
                        prob=token.prob,
                        tokens=[token],
                    )
                    if len(token.bytes) > len(backtracked_bytes_buffer):
                        new_bytes_buffer = new_bytes_buffer[
                            len(token.bytes) - len(backtracked_bytes_buffer) :
                        ]
                    backtracked_bytes_buffer = backtracked_bytes_buffer[len(token.bytes) :]
                else:
                    break
                i += 1
            token_buffer = token_buffer[i:]

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

        if new_bytes_buffer:
            # TODO: Fix the related error in tool call (nontrivial backtrack)
            raise RuntimeError("Shouldn't have any new bytes left...")
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
