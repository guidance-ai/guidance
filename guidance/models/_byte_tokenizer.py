import numpy as np
from ._engine import Tokenizer
from ..chat import load_template_class
from typing import List

class ByteTokenizer(Tokenizer):
    def __init__(self, chat_template=None):
        # directly map integer values to byte strings
        all_bytes = [bytes([i]) for i in range(256)]
        bos = b"<s>"
        tokens = np.array(all_bytes + [bos], dtype="object")
        chat_template = load_template_class(chat_template)
        super().__init__(tokens, chat_template, bos_token_id=256)

    def encode(self, byte_string: bytes) -> List[int]:
        """Returns a list of tokens that represent the given byte string."""
        if isinstance(byte_string, str):
            byte_string = byte_string.encode("utf8")
        i = 0
        result = []
        while i < len(byte_string):
            if byte_string[i:i+3] == b'<s>':
                result.append(256)
                i += 3  # Skip the next two characters as part of '<s>'
            else:
                result.append(byte_string[i])
                i += 1
        return result
