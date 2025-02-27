from typing import Iterator

from ..._ast import GrammarNode, ImageNode, LiteralNode, RoleEnd, RoleStart
from ...trace import (
    CaptureOutput,
    ImageOutput,
    OutputAttr,
    RoleCloserInput,
    RoleOpenerInput,
    TextOutput,
)
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

    def text(self, state: EngineState, node: LiteralNode, **kwargs) -> Iterator[OutputAttr]:
        output = TextOutput(value=node.value, input=True)
        state.apply_chunk(output)
        yield output

    def image(self, state: EngineState, node: ImageNode, **kwargs) -> Iterator[OutputAttr]:
        output = ImageOutput(value=node.value, input=True)
        state.apply_chunk(output)
        yield output

    def role_start(self, state: EngineState, node: RoleStart, **kwargs) -> Iterator[OutputAttr]:
        opener = RoleOpenerInput(
            name=node.role,
            text=self.get_role_start(node.role),
            closer_text=self.get_role_end(node.role),
        )
        state.apply_chunk(opener)
        yield opener

    def role_end(self, state: EngineState, node: RoleEnd, **kwargs) -> Iterator[OutputAttr]:
        closer = RoleCloserInput(
            name=node.role,
            text=self.get_role_end(node.role),
        )
        state.apply_chunk(closer)
        yield closer

    def grammar(self, state: EngineState, node: GrammarNode, **kwargs) -> Iterator[OutputAttr]:
        engine_gen = self.engine(
            state,
            node.ll_grammar(),
            ensure_bos_token=True,
            echo=False,
        )

        delayed_bytes = b""
        for chunk in engine_gen:
            generated_bytes = delayed_bytes + chunk.generated_bytes
            generated_text, delayed_bytes = partial_decode(generated_bytes)
            ff_bytes = delayed_bytes + chunk.force_forwarded_bytes
            ff_text, delayed_bytes = partial_decode(ff_bytes)

            ff_token_count = chunk.new_token_count
            if generated_bytes:
                ff_token_count -= 1
                output = TextOutput(
                    value=generated_text,
                    is_generated=True,
                    prob=chunk.new_bytes_prob,
                    token_count=1,  # len(chunk.generated_tokens),
                    tokens=chunk.generated_tokens,
                )
                state.apply_chunk(output)
                yield output

            if ff_bytes:
                output = TextOutput(
                    value=ff_text,
                    is_generated=False,
                    prob=chunk.new_bytes_prob,
                    token_count=ff_token_count,  # len(chunk.force_forwarded_tokens),
                    tokens=chunk.force_forwarded_tokens,
                )
                state.apply_chunk(output)
                yield output

            for name in chunk.capture_groups.keys():
                values = chunk.capture_groups[name]
                log_probs = chunk.capture_group_log_probs[name]
                if isinstance(values, list):
                    assert isinstance(log_probs, list) and len(log_probs) == len(values)
                    list_append = True
                else:
                    values = [values]
                    log_probs = [log_probs]
                    list_append = False

                for value, log_prob in zip(values, log_probs):
                    capture = CaptureOutput(
                        name=name,
                        value=value,
                        is_append=list_append,
                        # TODO: let this be Optional?
                        log_probs=log_prob,
                    )
                    state.apply_chunk(capture)
                    yield capture

        if delayed_bytes:
            raise RuntimeError("Shouldn't have any delayed bytes left...")


def partial_decode(data: bytes) -> tuple[str, bytes]:
    try:
        return (data.decode("utf-8"), b"")
    except UnicodeDecodeError as e:
        valid_part = data[: e.start].decode("utf-8")
        delayed_part = data[e.start :]
    return (valid_part, delayed_part)
