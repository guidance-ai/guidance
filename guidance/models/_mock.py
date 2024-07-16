from typing import Sequence

import numpy as np
import logging

from ._model import Engine, Model, Chat
from ._remote import RemoteEngine
from ._tokenizer import Tokenizer
from ._byte_tokenizer import ByteTokenizer

logger = logging.getLogger(__name__)

# TODO: this import pattern is neded for both Grammarless and Mock, but not the Model base class.
#   we should refactor this to prevent the need for this import pattern
try:
    from .. import cpp  # type: ignore[attr-defined]
except ImportError:
    logger.warn(
        "Failed to load guidance.cpp, falling back to Python mirror implementations..."
    )
    from .. import _cpp as cpp


class MockEngine(Engine):
    def __init__(self, tokenizer, byte_patterns, compute_log_probs, force):
        super().__init__(tokenizer, compute_log_probs=compute_log_probs)

        self._token_trie = cpp.ByteTrie(
            self.tokenizer.tokens, np.arange(len(self.tokenizer.tokens))
        )

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

    def get_next_token(self, token_ids: list[int], mask: np.ndarray, temperature: float) -> int:
        self.called_temperatures.append(temperature)
        return super().get_next_token(token_ids, mask, 0.)

    def get_logits(self, token_ids):
        """Pretends to compute the logits for the given token state."""
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
                    i = self._get_next_token(p[len(byte_string) :])
                    logits[i] += bias

        return logits

    def _get_next_token(self, byte_string) -> int:
        """Tokenize the prefix of a byte string and return the token id."""
        trie = self._token_trie
        pos = 0
        token_id = None
        while (next_byte := byte_string[pos: pos + 1]):
            if trie.has_child(next_byte):
                trie = trie.child(next_byte)
                pos += 1
                if trie.value >= 0:
                    token_id = trie.value
            else:
                break
        if token_id is None:
            raise ValueError(f"Could not tokenize byte string: {byte_string!r}")
        return token_id

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
            tokenizer = ByteTokenizer()
            engine = MockEngine(tokenizer, byte_patterns, compute_log_probs, force)

        super().__init__(engine, echo=echo)


class MockChat(Mock, Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
