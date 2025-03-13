import logging
from typing import Optional, Sequence

import numpy as np

from .._schema import EngineOutput, GenToken, GenTokenExtra
from .._utils import softmax
from ..trace import TraceHandler
from ..visual._renderer import DoNothingRenderer
from ._base import Model
from ._engine import Engine, EngineClient, EngineState, Tokenizer

logger = logging.getLogger(__name__)

# TODO: this import pattern happens in a few places, should be cleaned up
try:
    from .. import cpp  # type: ignore[attr-defined]
except ImportError:
    logger.warn("Failed to load guidance.cpp, falling back to Python mirror implementations...")
    from .. import _cpp as cpp


class MockTokenizer(Tokenizer):
    def __init__(self, tokens: Sequence[bytes], special_token_ids: Optional[list[int]] = None):
        super().__init__(tokens, chat_template=None, bos_token_id=0, eos_token_id=0, special_token_ids=special_token_ids)
        self.byte_trie = cpp.ByteTrie(self.tokens, np.arange(len(self.tokens)))

    def encode(self, byte_string: bytes) -> list[int]:
        """Simple greedy tokenizer
        TODO: could be a method on ByteTrie if we want to reuse it
        """
        pos = 0
        tokens = []
        while pos < len(byte_string):
            current_node = self.byte_trie
            last_match = None
            match_pos = pos

            while match_pos < len(byte_string) and current_node.has_child(
                byte_string[match_pos : match_pos + 1]
            ):
                current_node = current_node.child(byte_string[match_pos : match_pos + 1])
                if current_node.value >= 0:
                    last_match = (current_node.value, match_pos + 1)
                match_pos += 1

            if last_match is not None:
                tokens.append(last_match[0])
                pos = last_match[1]
            else:
                raise ValueError(
                    f"Could not find a match for byte {byte_string[pos]} at position {pos}"
                )

        return tokens

    def recode(self, tokens: Sequence[int]) -> list[int]:
        # Make a no-op for now
        return list(tokens)


class MockEngine(Engine):
    def __init__(self, tokenizer, byte_patterns, compute_log_probs, force):
        renderer = DoNothingRenderer(trace_handler=TraceHandler())
        super().__init__(tokenizer, compute_log_probs=compute_log_probs)

        self._valid_mask = np.zeros(len(tokenizer.tokens))
        for i, t in enumerate(tokenizer.tokens):
            try:
                t.decode("utf8")
                self._valid_mask[i] = 1.0
            except:
                pass
        self.force = force
        self.called_temperatures = []

        # allow a single byte pattern to be passed
        if isinstance(byte_patterns, (bytes, str)):
            byte_patterns = [byte_patterns]

        # allow for strings to be passed
        for i, pattern in enumerate(byte_patterns):
            if isinstance(pattern, str):
                byte_patterns[i] = pattern.encode("utf8")

        self.byte_patterns = byte_patterns

        # seed the random number generator
        self._rand_generator = np.random.default_rng(seed=42)

    def sample_with_temperature(self, logits, mask, temperature):
        self.called_temperatures.append(temperature)
        return super().sample_with_temperature(logits, mask, temperature)

    def get_next_token_with_top_k(
        self,
        logits: Optional[np.ndarray],
        logits_lat_ms: Optional[float],
        token_ids: list[int],
        mask: Optional[bytes],
        temperature: float,
        k: int = 1,
        force_return_unmasked_probs: bool = False,
    ) -> EngineOutput:
        self.called_temperatures.append(temperature)
        return super().get_next_token_with_top_k(
            logits, logits_lat_ms, token_ids, mask, temperature, k, force_return_unmasked_probs
        )

    def get_logits(self, token_ids: list[int]) -> np.ndarray:
        """Pretends to compute the logits for the given token state."""
        # build the byte strings
        byte_string = b"".join(self.tokenizer.tokens[i] for i in token_ids)

        # if we are forcing the bytes patterns then don't allow other tokens
        if self.force:
            logits = np.ones(len(self.tokenizer.tokens)) * -np.inf

        # otherwise we randomly generate valid unicode bytes
        else:
            logits = (
                self._rand_generator.standard_normal(len(self.tokenizer.tokens)) * self._valid_mask
            )

        # if we have a pattern that matches then force the next token
        bias = 100.0
        if self.byte_patterns is not None:
            byte_string
            for p in self.byte_patterns:
                if p.startswith(byte_string) and len(p) > len(byte_string):
                    for i in self._get_next_tokens(p[len(byte_string) :]):
                        logits[i] += bias
                    bias /= 2  # if we have multiple matches then they apply with decreasing bias

        return logits

    def get_per_token_topk_probs(
        self, token_ids: list[int], top_k: int = 5
    ) -> list[GenTokenExtra]:
        result_list = []
        if len(token_ids) == 0:
            return result_list

        added_bos = False
        if self.tokenizer.bos_token is not None and token_ids[0] != self.tokenizer.bos_token_id:
            token_ids = [self.tokenizer.bos_token_id] + token_ids
            added_bos = True

        # assume the first token has probability 1.0 because it is the input token
        result_list.append(
            GenTokenExtra(
                token_id=token_ids[0],
                prob=1.0,
                text=self.tokenizer.decode([token_ids[0]]).decode("utf8"),
                top_k=[
                    GenToken(
                        token_id=token_ids[0],
                        prob=1.0,
                        text=self.tokenizer.decode([token_ids[0]]).decode("utf8"),
                    )
                ],
            )
        )

        for i in range(1, len(token_ids)):
            token_id = token_ids[i]
            _logits = self.get_logits(token_ids[:i])
            _probs = softmax(_logits)
            top_k_indices = np.argsort(_logits)[-top_k:][::-1]

            top_k_indices = top_k_indices.tolist()
            if token_ids[i] not in top_k_indices:
                top_k_indices.append(token_id)

            top_k_result = []
            for token_id in top_k_indices:
                top_k_result.append(
                    GenToken(
                        token_id=token_id,
                        prob=_probs[token_id],
                        text=self.tokenizer.decode([token_id]).decode("utf8"),
                    )
                )

            result_list.append(
                GenTokenExtra(
                    token_id=token_id,
                    prob=_probs[token_id],
                    text=self.tokenizer.decode([token_id]).decode("utf-8"),
                    top_k=top_k_result,
                )
            )

        if added_bos:
            result_list = result_list[1:]

        return result_list

    def _get_next_tokens(self, byte_string):
        special_tokens = [
            (self.tokenizer.bos_token_id, self.tokenizer.bos_token),
            (self.tokenizer.eos_token_id, self.tokenizer.eos_token),
        ]
        for i, t in special_tokens:
            # if the byte string starts with a special token then make sure we don't yield any other tokens
            if byte_string.startswith(t):
                yield i
                return
        for i, t in enumerate(self.tokenizer.tokens):
            if byte_string.startswith(t):
                yield i


class Mock(Model):
    def __init__(
        self,
        byte_patterns=[],
        echo=False,
        compute_log_probs=False,
        force=False,
        **kwargs,
    ):
        """Build a new Mock model object that represents a model in a given state."""

        # Our tokens are all bytes and all lowercase letter pairs
        all_lc_pairs = [
            bytes([i, j]) for i in range(ord("a"), ord("z")) for j in range(ord("a"), ord("z"))
        ]
        all_bytes = [bytes([i]) for i in range(256)]
        tokens = [b"<s>"] + all_lc_pairs + all_bytes

        tokenizer = MockTokenizer(tokens, special_token_ids=[0])
        engine = MockEngine(tokenizer, byte_patterns, compute_log_probs, force)

        super().__init__(
            client=EngineClient(engine),
            state=EngineState(),
            echo=echo,
        )


# class MockChat(Mock, Chat):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
