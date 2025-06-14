from base64 import b64decode, b64encode
from io import BytesIO
from typing import Iterator
from copy import deepcopy
import re

from ..._ast import GrammarNode, ImageBlob, LiteralNode, RoleEnd, RoleStart, SpecialToken, JoinNode
from ..._utils import to_utf8_or_bytes_string
from ...trace import ImageOutput, OutputAttr, Backtrack, TokenOutput, Token
from .._base import Interpreter
from ._engine import Engine, Tokenizer
from ._state import EngineState
from ..._schema import GenTokenExtra


class EngineInterpreter(Interpreter[EngineState]):
    def __init__(self, engine: Engine):
        self.state = EngineState()
        self.engine = engine

    def state_str(self) -> str:
        msgs = self.state.messages
        if self.state.active_message is not None:
            msgs = msgs + [self.state.active_message]
        return self.engine.apply_chat_template(
            msgs,
        )

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

    def role_start(self, node: RoleStart, **kwargs) -> Iterator[OutputAttr]:
        self.state.active_role = node.role
        # TODO: something for vis?
        yield from ()

    def role_end(self, node: RoleEnd, **kwargs) -> Iterator[OutputAttr]:
        self.state.active_role = None
        # TODO: something for vis?
        yield from ()

    def text(self, node: LiteralNode, **kwargs) -> Iterator[OutputAttr]:
        yield from self.grammar(node, **kwargs)

    def grammar(self, node: GrammarNode, **kwargs) -> Iterator[OutputAttr]:
        engine_gen = self.engine(
            state=self.state,
            grammar=node.ll_grammar(),
            ensure_bos_token=True,
            echo=False,
        )

        delayed_bytes = b""
        for chunk in engine_gen:
            new_bytes = recode_special_tokens(self.engine.tokenizer, chunk.new_bytes)
            new_text, delayed_bytes = partial_decode(delayed_bytes + new_bytes)
            self.state.add_text(new_text)

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
                    is_input = token.is_input,
                    is_generated = token.is_generated,
                    is_force_forwarded = token.is_force_forwarded,
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
        self.state.add_media(
            media_type="image",
            media=pil_image,
            text_representation="<|image|>",
            allow_ref=False,
        )
        yield ImageOutput(value=node.data, is_input=True)


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
        self.state.add_media(
            media_type="image",
            media=pil_image,
            text_representation=lambda ix: f"<|image_{ix}|>",
            allow_ref=True,
        )
        yield ImageOutput(value=node.data, is_input=True)


def partial_decode(data: bytes) -> tuple[str, bytes]:
    try:
        return (data.decode("utf-8"), b"")
    except UnicodeDecodeError as e:
        valid_part = data[: e.start].decode("utf-8")
        delayed_part = data[e.start :]
    return (valid_part, delayed_part)

LLG_SPECIAL_TOKEN_PAT = re.compile(br"\xff\[([0-9]+)\]")
def recode_special_tokens(tokenizer: Tokenizer, data: bytes) -> bytes:
    """Recode a byte string with special tokens in llguidance format to their actual byte representation."""
    return LLG_SPECIAL_TOKEN_PAT.sub(
        lambda m: tokenizer.decode([int(m.group(1).decode("utf-8"))]),
        data
    )

def text_to_grammar(tokenizer: Tokenizer, text: str) -> GrammarNode:
    """
    Convert a text string into a GrammarNode that can be used in the grammar.
    This is useful for converting static text into a grammar node that can be processed by the engine.
    """
    grammar_bits: list[GrammarNode] = []
    delayed_bytes = b""
    for token_id in tokenizer.encode(text.encode("utf-8"), parse_special=True):
        if tokenizer.is_special_token(token_id):
            assert not delayed_bytes, "Should not have any delayed bytes when encountering a special token"
            grammar_bits.append(SpecialToken(id=token_id))
        else:
            new_bytes = tokenizer.decode([token_id])
            new_text, delayed_bytes = partial_decode(delayed_bytes + new_bytes)
            if new_text:
                grammar_bits.append(LiteralNode(new_text))
    assert not delayed_bytes, "Should not have any delayed bytes left after processing the text"
    if len(grammar_bits) == 1:
        return grammar_bits[0]
    return JoinNode(tuple(grammar_bits))
