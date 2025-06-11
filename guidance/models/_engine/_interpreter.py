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
        yield from self.run(
            text_to_grammar(self.engine.tokenizer, text)
        )

    def role_end(self, node: RoleEnd, **kwargs) -> Iterator[OutputAttr]:
        self.state.active_role = None
        text = self.get_role_end(node.role)
        # TODO: it's probably somewhat wasteful to trigger engine calls here,
        # so we can maybe add this as "pending text" to the state instead,
        # accumulating it until the next engine call..?
        yield from self.run(
            text_to_grammar(self.engine.tokenizer, text)
        )

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
        self.state.images.append(pil_image)
        self.state.prompt += "<|image|>"

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

        if pil_image in self.state.images:
            ix = self.state.images.index(pil_image) + 1
        else:
            self.state.images.append(pil_image)
            ix = len(self.state.images)
        self.state.prompt += f"<|image_{ix}|>"

        yield ImageOutput(value=node.data, is_input=True)


def partial_decode(data: bytes) -> tuple[str, bytes]:
    try:
        return (data.decode("utf-8"), b"")
    except UnicodeDecodeError as e:
        valid_part = data[: e.start].decode("utf-8")
        delayed_part = data[e.start :]
    return (valid_part, delayed_part)

LLG_SPECIAL_TOKEN_PAT = re.compile(b"\xff\[([0-9]+)\]")
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
