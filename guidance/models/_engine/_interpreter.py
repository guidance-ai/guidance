from base64 import b64decode
from io import BytesIO
from typing import Iterator
from copy import deepcopy
import re

from ..._ast import GrammarNode, ImageBlob, LiteralNode, RoleEnd, RoleStart
from ...trace import ImageOutput, OutputAttr, TextOutput
from .._base import Interpreter
from ._engine import Engine, Tokenizer
from ._state import EngineState

class EngineInterpreter(Interpreter[EngineState]):
    def __init__(self, engine: Engine):
        self.state = EngineState()
        self.engine = engine
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
        # TODO: mark these as special tokens..?
        yield from self.run(LiteralNode(value=self.get_role_start(node.role)), **kwargs)

    def role_end(self, node: RoleEnd, **kwargs) -> Iterator[OutputAttr]:
        self.state.active_role = None
        # TODO: mark these as special tokens..?
        yield from self.run(LiteralNode(value=self.get_role_end(node.role)), **kwargs)

    def text(self, node: LiteralNode, **kwargs) -> Iterator[OutputAttr]:
        self.state.prompt += node.value
        yield TextOutput(value=node.value, is_input=True)

    def grammar(self, node: GrammarNode, **kwargs) -> Iterator[OutputAttr]:
        engine_gen = self.engine(
            state=self.state,
            grammar=node.ll_grammar(),
            ensure_bos_token=True,
            echo=False,
        )
        delayed_bytes = b""
        for chunk in engine_gen:
            new_bytes = parse_special_tokens(self.engine.tokenizer, chunk.new_bytes)
            new_text, delayed_bytes = partial_decode(new_bytes)
            # Update the state
            self.state.prompt += new_text
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
                        yield self.state.apply_capture(
                            name, value.decode("utf-8"), log_prob=log_prob, is_append=True
                        )
                else:
                    assert isinstance(log_probs, float)
                    yield self.state.apply_capture(
                        name, values.decode("utf-8"), log_prob=log_probs, is_append=False
                    )

        if delayed_bytes:
            raise RuntimeError("Shouldn't have any delayed bytes left...")


class Llama3VisionInterpreter(EngineInterpreter):
    def image_blob(self, node: ImageBlob, **kwargs) -> Iterator[OutputAttr]:
        try:
            import PIL.Image
        except ImportError:
            raise Exception(
                "Please install the Pillow package `pip install Pillow` in order to use images with Llama3!"
            )

        image_bytes = b64decode(node.data)
        pil_image = PIL.Image.open(BytesIO(image_bytes))
        self.state.images.append(pil_image)
        self.state.prompt += "<|image|>"

        yield ImageOutput(value=node.data, input=True)


class Phi3VisionInterpreter(EngineInterpreter):
    def image_blob(self, node: ImageBlob, **kwargs) -> Iterator[OutputAttr]:
        try:
            import PIL.Image
        except ImportError:
            raise Exception(
                "Please install the Pillow package `pip install Pillow` in order to use images with Llama3!"
            )

        image_bytes = b64decode(node.data)
        pil_image = PIL.Image.open(BytesIO(image_bytes))

        if pil_image in self.state.images:
            ix = self.state.images.index(pil_image) + 1
        else:
            self.state.images.append(pil_image)
            ix = len(self.state.images)
        self.state.prompt += f"<|image_{ix}|>"

        yield ImageOutput(value=node.data, input=True)


def partial_decode(data: bytes) -> tuple[str, bytes]:
    try:
        return (data.decode("utf-8"), b"")
    except UnicodeDecodeError as e:
        valid_part = data[: e.start].decode("utf-8")
        delayed_part = data[e.start :]
    return (valid_part, delayed_part)

_SPECIAL_TOKEN_PAT = re.compile(rb"\xff\[([0-9]+)\]")
def parse_special_tokens(tokenizer: Tokenizer, bts: bytes) -> bytes:
    """
    llguidance will emit special tokens in the form of _SPECIAL_TOKEN_PAT,
    and we need to parse them back into the original text.
    """
    out = b""
    isspecial = False
    for match in _SPECIAL_TOKEN_PAT.split(bts):
        if isspecial:
            token_id = int(match.decode("utf-8"))
            out += tokenizer.decode([token_id])
        else:
            out += match
        isspecial = not isspecial
    return out
