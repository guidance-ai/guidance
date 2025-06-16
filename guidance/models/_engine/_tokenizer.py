from typing import Any, Optional, Sequence, Union, Callable
from dataclasses import dataclass
from functools import cached_property


from ...chat import ChatTemplate, load_template_class
import llguidance

@dataclass
class TokenizerWrappable:
    eos_token_id: int
    bos_token_id: Optional[int]
    tokens: list[bytes]
    special_token_ids: list[int]
    encode_callable: Callable[[bytes], list[int]]

    def __call__(self, byte_string: bytes) -> list[int]:
        return self.encode_callable(byte_string)

    def as_ll_tokenizer(self) -> "llguidance.LLTokenizer":
        """Returns an LLTokenizer that can be used by llguidance."""
        return llguidance.LLTokenizer(
            llguidance.TokenizerWrapper(self)
        )

class Tokenizer:
    """This is the standardized tokenizer interface used by guidance models.

    This class should be subclassed by specific implementations and then used as the
    tokenizer in the corresponding Engine subclass.
    """

    def __init__(
        self,
        ll_tokenizer: llguidance.LLTokenizer,
        chat_template: Union[str, ChatTemplate, None],
        bos_token_id: Optional[int] = None,
    ):
        self._ll_tokenizer = ll_tokenizer
        # This method supports None, a huggingface style jinja2_template_str, or a ChatTemplate subclass
        # Defaults to ChatML if nothing is found
        self._chat_template = load_template_class(chat_template)
        self._bos_token_id = bos_token_id

    def is_special_token(self, token_id: int) -> bool:
        """Returns True if the given token ID is a special token."""
        return self._ll_tokenizer.is_special_token(token_id)

    @property
    def bos_token_id(self) -> Union[int, None]:
        # Currently, lltokenizer does not have a bos_token attribute,
        # so we have to store our own if we want to use it
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._ll_tokenizer.eos_token

    @cached_property
    def bos_token(self) -> Union[bytes, None]:
        if self.bos_token_id is None:
            return None
        return self.decode([self.bos_token_id])

    @cached_property
    def eos_token(self) -> bytes:
        return self.decode([self.eos_token_id])

    @property
    def chat_template(self) -> Union[Any, None]:
        return self._chat_template

    def __call__(self, byte_string: bytes):
        return self.encode(byte_string)

    def encode(self, byte_string: bytes, *, parse_special: bool = True) -> list[int]:
        """Returns a list of tokens that represent the given byte string."""
        return self._ll_tokenizer.tokenize_bytes(byte_string, parse_special=parse_special)

    def decode(self, tokens: Sequence[int]) -> bytes:
        """Returns the bytes represented by the given list of tokens."""
        return self._ll_tokenizer.decode_bytes(list(tokens))

    def recode(self, tokens: Sequence[int]) -> list[int]:
        """Redoes a tokenization.

        Encoding a string into tokens does not distribute over concatenation.
        That is, in general, `encode(A)+encode(B) != encode(A+B)` (although it
        it may in some cases).
        An LLM will generate token-by-token, but it is possible (even likely) that
        when the generation is considered as a whole, a better tokenization may
        be possible.
        This method takes in a sequence of tokens, and returns an 'improved' sequence.
        """

        # This is the notional behavior
        # It may need to be overridden in particular cases because
        # we are dealing with LLM ecosystems in the real world
        return self.encode(self.decode(tokens))
