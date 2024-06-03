from typing import Any, Sequence, Union

import numpy as np

from ..chat import load_template_class, ChatTemplate


class Tokenizer:
    """This is the standardized tokenizer interface used by guidance models.

    This class should be subclassed by specific implementations and then used as the
    tokenizer in the corresponding Engine subclass.
    """

    def __init__(
        self,
        tokens: Union[list, np.ndarray],
        chat_template: Union[str, ChatTemplate, None],
        bos_token_id: Union[int, None] = None,
        eos_token_id: Union[int, None] = None,
    ):

        # a numpy array of token byte strings indexed by their token id
        if isinstance(tokens, list):
            # note that we need np.bytes_ so zero bytes are not treated as null terminations
            self._tokens = np.array(tokens, dtype="object")

        # a numpy array of token byte strings indexed by their token id
        elif isinstance(tokens, np.ndarray):
            self._tokens = tokens

        else:
            raise ValueError("Unknown tokenizer was passed!")

        assert isinstance(
            self.tokens[0], bytes
        ), "The tokens need to be provided as bytes!"

        # This method supports None, a huggingface style jinja2_template_str, or a ChatTemplate subclass
        # Defaults to ChatML if nothing is found
        self._chat_template = load_template_class(chat_template)

        self._bos_token_id = bos_token_id
        self._bos_token = (
            None if self.bos_token_id is None else self.tokens[self.bos_token_id]
        )
        self._eos_token_id = eos_token_id if eos_token_id is not None else bos_token_id
        self._eos_token = (
            None if self.eos_token_id is None else self.tokens[self.eos_token_id]
        )

        # track which tokens are duplicates
        self._duplicate_tokens = []
        found = {}
        for i, t in enumerate(self.tokens):
            if t in found:
                self._duplicate_tokens.append((i, found[t]))
            else:
                found[t] = i

    @property
    def tokens(self) -> np.ndarray:
        return self._tokens

    @property
    def bos_token_id(self) -> Union[int, None]:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> Union[int, None]:
        return self._eos_token_id

    @property
    def bos_token(self) -> Union[bytes, None]:
        return self._bos_token

    @property
    def eos_token(self) -> Union[bytes, None]:
        return self._eos_token

    @property
    def chat_template(self) -> Union[Any, None]:
        return self._chat_template

    def __call__(self, byte_string: bytes):
        return self.bytes_to_tokens(byte_string)

    def bytes_to_tokens(self, byte_string: bytes) -> Sequence[int]:
        """Returns a list of tokens that represent the given byte string."""
        raise NotImplementedError(
            "You need to use a Tokenize subclass that overrides the bytes_to_tokens method"
        )

    def tokens_to_bytes(self, tokens: Sequence[int]) -> bytes:
        """Returns the bytes represented by the given list of tokens."""
        raise NotImplementedError(
            "You need to use a Tokenize subclass that overrides the tokens_to_bytes method"
        )

    def clean_duplicate_tokens(self, probs):
        """This moves all the probability mass from duplicate positons on to their primary index."""
        for i, j in self._duplicate_tokens:
            probs[j] += probs[i]
            probs[i] = 0
