import numpy as np
from ._tokenizer import Tokenizer
from ..chat import load_template_class
import typing

class ByteTokenizer(Tokenizer):
    def __init__(self, chat_template=None):
        # directly map integer values to byte strings
        tokens = np.array([bytes([i]) for i in range(256)], dtype="object")
        chat_template = load_template_class(chat_template)
        super().__init__(tokens, chat_template)

    def __call__(self, byte_string) -> typing.List[int]:
        """Returns a list of tokens that represent the given byte string."""
        return list(byte_string)
