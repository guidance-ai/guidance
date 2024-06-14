from typing import Sequence

import numpy as np

from ._model import Engine, Model, Chat
from ._remote import RemoteEngine
from ._tokenizer import Tokenizer


class MockTokenizer(Tokenizer):
    def __init__(self, tokens: Sequence[bytes]):

        super().__init__(tokens, chat_template=None, bos_token_id=0, eos_token_id=0)

    def recode(self, tokens: Sequence[int]) -> Sequence[int]:
        # Make a no-op for now
        return tokens


class MockEngine(Engine):
    def __init__(self, tokenizer, byte_patterns, compute_log_probs, force):
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

    def get_logits(self, token_ids, forced_bytes, current_temp):
        """Pretends to compute the logits for the given token state."""
        self.called_temperatures.append(current_temp)

        # build the byte strings
        byte_string = b"".join(self.tokenizer.tokens[i] for i in token_ids)

        # if we are forcing the bytes patterns then don't allow other tokens
        if self.force:
            logits = np.ones(len(self.tokenizer.tokens)) * -np.inf

        # otherwise we randomly generate valid unicode bytes
        else:
            logits = (
                self._rand_generator.standard_normal(len(self.tokenizer.tokens))
                * self._valid_mask
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

    def _get_next_tokens(self, byte_string):
        for i, t in enumerate(self.tokenizer.tokens):
            if byte_string.startswith(t):
                yield i


class Mock(Model):
    def __init__(
        self,
        byte_patterns=[],
        echo=True,
        compute_log_probs=False,
        force=False,
        **kwargs,
    ):
        """Build a new Mock model object that represents a model in a given state."""

        if isinstance(byte_patterns, str) and byte_patterns.startswith("http"):
            engine = RemoteEngine(byte_patterns, **kwargs)
        else:
            # Our tokens are all bytes and all lowercase letter pairs
            all_lc_pairs = [
                bytes([i, j])
                for i in range(ord("a"), ord("z"))
                for j in range(ord("a"), ord("z"))
            ]
            all_bytes = [bytes([i]) for i in range(256)]
            tokens = [b"<s>"] + all_lc_pairs + all_bytes

            tokenizer = MockTokenizer(tokens)
            engine = MockEngine(tokenizer, byte_patterns, compute_log_probs, force)

        super().__init__(engine, echo=echo)


class MockChat(Mock, Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
