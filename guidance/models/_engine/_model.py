from typing import Iterator

from ..._grammar import Function
from ...experimental.ast import ImageBlob, MessageChunk, Node, RoleEnd, RoleStart
from ...trace import (
    CaptureOutput,
    LiteralInput,
    RoleCloserInput,
    RoleOpenerInput,
    TextOutput,
)
from .._base._model import Model, partial_decode
from ._engine import Engine
from ._state import EngineState


class ModelWithEngine(Model[EngineState]):
    def __init__(self, engine: Engine, echo: bool = True):
        self.engine = engine
        super().__init__(echo=echo)

    def run(self, state: EngineState, node: Node) -> Iterator[MessageChunk]:
        if isinstance(node, str):
            yield LiteralInput(value=node)

        elif isinstance(node, RoleStart):
            chat_template = self.engine.get_chat_template()
            if chat_template is None:
                raise ValueError("Cannot use roles without a chat template")
            yield RoleOpenerInput(
                name=node.role,
                text=chat_template.get_role_start(node.role),
                closer_text=chat_template.get_role_end(node.role),
            )

        elif isinstance(node, RoleEnd):
            chat_template = self.engine.get_chat_template()
            if chat_template is None:
                raise ValueError("Cannot use roles without a chat template")
            yield RoleCloserInput(
                name=node.role,
                text=chat_template.get_role_end(node.role),
            )

        elif isinstance(node, ImageBlob):
            yield node

        elif isinstance(node, Function):
            engine_gen = self.engine(
                state,
                node,
                ensure_bos_token=True,
                echo=False,
            )

            delayed_bytes = b""
            for chunk in engine_gen:
                generated_bytes = delayed_bytes + chunk.generated_bytes
                generated_text, delayed_bytes = partial_decode(generated_bytes)
                ff_bytes = delayed_bytes + chunk.force_forwarded_bytes
                ff_text, delayed_bytes = partial_decode(ff_bytes)

                if generated_bytes:
                    yield TextOutput(
                        value=generated_text,
                        is_generated=True,
                        prob=chunk.new_bytes_prob,
                        token_count=len(chunk.generated_tokens),
                        tokens=chunk.generated_tokens,
                    )
                if ff_bytes:
                    yield TextOutput(
                        value=ff_text,
                        is_generated=False,
                        prob=chunk.new_bytes_prob,
                        token_count=len(chunk.force_forwarded_tokens),
                        tokens=chunk.force_forwarded_tokens,
                    )

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
                        yield CaptureOutput(
                            name=name,
                            value=value,
                            is_append=list_append,
                            # TODO: let this be Optional?
                            log_probs=log_prob,
                        )

            if delayed_bytes:
                raise RuntimeError("Shouldn't have any delayed bytes left...")

        else:
            raise NotImplementedError(f"Unknown node: {node}")

    def initial_state(self) -> EngineState:
        # TODO: for llama_cpp and transformers, we need to provide an interface
        # for getting these from something like a model id..?
        return EngineState()
