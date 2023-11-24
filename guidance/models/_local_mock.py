import numpy as np

from ._model import Chat
from ._local import Local


class LocalMock(Local):
    def __init__(self, byte_patterns=[], echo=True):
        
        super().__init__(
            # our tokens are all bytes and all lowercase letter pairs
            [b"<s>"] + [bytes([i,j]) for i in range(ord('a'), ord('z')) for j in range(ord('a'), ord('z'))] + [bytes([i]) for i in range(256)],
            0,
            0,
            echo=echo
        )

        self._valid_mask = np.zeros(len(self.tokens))
        for i,t in enumerate(self.tokens):
            try:
                t.decode("utf8")
                self._valid_mask[i] = 1.0
            except:
                pass

        # allow a single byte pattern to be passed
        if isinstance(byte_patterns, (bytes, str)):
            byte_patterns = [byte_patterns]

        # allow for strings to be passed
        for i,pattern in enumerate(byte_patterns):
            if isinstance(pattern, str):
                byte_patterns[i] = pattern.encode("utf8")

        self.byte_patterns = byte_patterns
        self._rand_generator = np.random.default_rng(seed=42)

        self._cache_state["past_key_values"] = None
        self._cache_state["logits"] = None
        self.eos_token = b"<s>"

    def _get_logits(self, token_ids, forced_bytes):
        '''Pretends to compute the logits for the given token state.
        '''

        # build the byte strings
        byte_string = b"".join(self.tokens[i] for i in token_ids)

        # we randomly generate valid unicode bytes
        logits = self._rand_generator.standard_normal(len(self.tokens)) * self._valid_mask

        # if we have a pattern that matches then force the next token
        bias = 100.0
        if self.byte_patterns is not None:
            byte_string
            for p in self.byte_patterns:
                if p.startswith(byte_string) and len(p) > len(byte_string):
                    for i in self._get_next_tokens(p[len(byte_string):]):
                        logits[i] += bias
                    bias /= 2 # if we have multiple matches then they apply with decreasing bias
        
        return logits
    
    def _get_next_tokens(self, byte_string):
        for i,t in enumerate(self.tokens):
            if byte_string.startswith(t):
                yield i
        

class LocalMockChat(LocalMock, Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)