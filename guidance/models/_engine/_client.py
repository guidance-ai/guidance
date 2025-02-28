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
        # TODO: mark these as special tokens..?
        yield from self.run(state, LiteralNode(value=self.get_role_start(node.role)), **kwargs)

    def role_end(self, state: EngineState, node: RoleEnd, **kwargs) -> Iterator[OutputAttr]:
        state.active_role = None
        # TODO: mark these as special tokens..?
        yield from self.run(state, LiteralNode(value=self.get_role_end(node.role)), **kwargs)

    def text(self, state: EngineState, node: LiteralNode, **kwargs) -> Iterator[OutputAttr]:
        state.prompt += node.value
        yield TextOutput(value=node.value, is_input=True)

    def grammar(self, state: EngineState, node: GrammarNode, **kwargs) -> Iterator[OutputAttr]:
        engine_gen = self.engine(
            state,
            node.ll_grammar(),
            ensure_bos_token=True,
            echo=False,
        )

        delayed_bytes = b""
        for chunk in engine_gen:
            new_bytes = chunk.new_bytes
            new_text, delayed_bytes = partial_decode(new_bytes)

            # Update the state
            state.prompt += new_text
            yield TextOutput(value=new_text, token_count=chunk.new_token_count, is_generated=True)

            # TODO -- rewrite engine internals to make sure chunk.{generated,fast_forwarded}_tokens aren't empty...
            # # TODO: GenTokenExtra
            # for token in chunk.generated_tokens:
            #     yield TextOutput(
            #         value=token.text,  # TODO: this should really be the token bytes
            #         is_generated=True,
            #         token_count=1,
            #         prob=token.prob,
            #         tokens=[token],  # TODO: drop this
            #     )
            # for token in chunk.force_forwarded_tokens:
            #     yield TextOutput(
            #         value=token.text,  # TODO: this should really be the token bytes
            #         is_generated=False,
            #         token_count=1,
            #         prob=token.prob,
            #         tokens=[token],  # TODO: drop this
            #     )

            # # TODO: yield some kind of backtrack signal?

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
