from typing import Sequence

import numpy as np

from ._model import Engine, Model, Chat
from ._remote import RemoteEngine
from ._tokenizer import Tokenizer
from ._byte_tokenizer import ByteTokenizer

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
                    next_token = self.tokenizer.encode(p[len(byte_string) :])[0]
                    logits[next_token] += bias

        return logits

    def sample_with_temperature(self, logits: np.ndarray, mask: np.ndarray, temperature: float):
        self.called_temperatures.append(temperature)
        return super().sample_with_temperature(logits, mask, 0.)

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
