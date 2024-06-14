from typing import List, Optional
import json
import numpy as np
import llguidance

from ._grammar import GrammarFunction, Terminal, Join
from .models._tokenizer import Tokenizer


class ParserException(Exception):
    def __init__(self, *args, **kwargs):
        self.current_byte = kwargs.pop("current_byte", None)
        self.allowed_bytes = kwargs.pop("allowed_bytes", None)
        self.consumed_bytes = kwargs.pop("consumed_bytes", None)
        super().__init__(*args, **kwargs)

class Parser:
    """An abstract base class for guidance parsers."""

    pass

class LLParser(Parser):

    def __init__(self, grammar: GrammarFunction, tokenizer: Tokenizer):
        # we can't have a terminal as the root
        if isinstance(grammar, Terminal):
            grammar = Join([grammar])

        self.grammar = grammar
        self.tokenizer = tokenizer

        self.ll_tokenizer = llguidance.LLTokenizer(
            llguidance.TokenizerWrapper(tokenizer)
        )
        self.ll_parser = llguidance.LLInterpreter(
            self.ll_tokenizer,
            json.dumps(grammar.ll_serialize()),
            log_level=2,
        )

    def start(self, prompt: bytes = b'', ensure_bos_token=True):
        # add the beginning of sequence token if needed
        if (
            ensure_bos_token
            and self.tokenizer.bos_token is not None
            and not prompt.startswith(self.tokenizer.bos_token)
        ):
            prompt = self.tokenizer.bos_token + prompt

        prompt_tokens = self.tokenizer.encode(prompt)
        self.tokens: List[int] = self.ll_parser.process_prompt(prompt_tokens)
        self.ff_tokens: List[int] = []
        self.backtrack: int = 0
        self.mask: Optional[bytes] = None
        self.done: bool = False
        self.next_token_temperature: float = -1.
        self.progress: List[dict] = []
        self.can_consume_token: bool = False

    def advance(self):
        if self.done:
            raise ParserException("Attempted to advance a parser that is already done!")
        
        mask, resp = self.ll_parser.mid_process(self.backtrack, self.ff_tokens)
        r = json.loads(resp)

        if r["stop"]:
            self.can_consume_token = False
            self.done = True
            return
        
        self.backtrack = r["backtrack"]
        self.ff_tokens = r["ff_tokens"]
        self.progress = r["progress"]
        self.next_token_temperature = r["temperature"]
        self.mask = mask

        if mask is not None:
            assert self.backtrack == 0
            assert len(self.ff_tokens) == 0
            self.can_consume_token = True
        else:
            if self.backtrack:
                del self.tokens[-self.backtrack:]
            self.tokens += self.ff_tokens
            self.can_consume_token = False

    def consume_token(self, tok_id: int):
        assert self.can_consume_token

        if tok_id not in self.valid_next_tokens():
            raise ParserException(
                "Attempted to consume a token that was not a valid next token!"
            )
        self.tokens.append(tok_id)
        self.ff_tokens = [tok_id]

        self.can_consume_token = False
    
    def next_token_mask(self):
        return np.frombuffer(self.mask, dtype=np.uint8)

    def valid_next_tokens(self):
        [tok_ids] = np.nonzero(self.next_token_mask())
        return tok_ids
