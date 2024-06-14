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

    @property
    def progress(self) -> dict:
        return self._handle_progress(self._progress)

    def start(self, prompt: bytes = b'', ensure_bos_token=True):
        # add the beginning of sequence token if needed
        if (
            ensure_bos_token
            and self.tokenizer.bos_token is not None
            and not prompt.startswith(self.tokenizer.bos_token)
        ):
            prompt = self.tokenizer.bos_token + prompt
        prompt_tokens = self.tokenizer.encode(prompt)

        self._tokens: List[int] = self.ll_parser.process_prompt(prompt_tokens)
        self._ff_tokens: List[int] = []
        self._backtrack: int = 0
        self._mask: Optional[bytes] = None
        self._progress: List[dict] = []
        
        self.done: bool = False
        self.next_token_temperature: float = -1.
        self.can_consume_token: bool = False

    def advance(self):
        if self.done:
            raise ParserException("Attempted to advance a parser that is already done!")
        
        mask, resp = self.ll_parser.mid_process(self._backtrack, self._ff_tokens)
        r = json.loads(resp)

        if r["stop"]:
            self.can_consume_token = False
            self.done = True
            return
        
        self._backtrack = r["backtrack"]
        self._ff_tokens = r["ff_tokens"]
        self._progress = r["progress"]
        self.next_token_temperature = r["temperature"]
        self._mask = mask

        if mask is not None:
            assert self._backtrack == 0
            assert len(self._ff_tokens) == 0
            self.can_consume_token = True
        else:
            if self._backtrack:
                del self._tokens[-self._backtrack:]
            self._tokens += self._ff_tokens
            self.can_consume_token = False

    def consume_token(self, tok_id: int):
        assert self.can_consume_token

        if tok_id not in self.valid_next_tokens():
            raise ParserException(
                "Attempted to consume a token that was not a valid next token!"
            )
        self._tokens.append(tok_id)
        self._ff_tokens = [tok_id]

        self.can_consume_token = False
    
    def next_token_mask(self):
        return np.frombuffer(self._mask, dtype=np.uint8)

    def valid_next_tokens(self):
        [tok_ids] = np.nonzero(self.next_token_mask())
        return tok_ids
    
    @staticmethod
    def _handle_progress(progress: List[dict]) -> dict:
        # TODO: schema obj

        new_bytes = b""
        new_token_count = 0
        new_bytes_prob = 0.0
        is_generated = False
        capture_groups = {}
        capture_group_log_probs = {}
        num_text_entries = 0

        for j in progress:
            tag = j.get("object", "")
            if tag == "capture":
                is_generated = True
                cname: str = j["name"]
                data = bytes.fromhex(j["hex"])
                if cname.startswith("__LIST_APPEND:"):
                    cname = cname[14:]
                    if cname not in capture_groups or \
                        not isinstance(capture_groups[cname], list):
                        capture_groups[cname] = []
                        capture_group_log_probs[cname] = []
                    capture_groups[cname].append(data)
                    capture_group_log_probs[cname].append(j["log_prob"])
                else:
                    capture_groups[cname] = data
                    capture_group_log_probs[cname] = j["log_prob"]
            elif tag == "text":
                # it actually should only happen once per round...
                new_bytes += bytes.fromhex(j["hex"])
                new_token_count += j["num_tokens"]
                new_bytes_prob += j["log_prob"]
                is_generated |= j["is_generated"]
                num_text_entries += 1
        if num_text_entries > 0:
            new_bytes_prob /= num_text_entries

        return {
            "new_bytes": new_bytes,
            "new_token_count": new_token_count,
            "new_bytes_prob": new_bytes_prob,
            "is_generated": is_generated,
            "capture_groups": capture_groups,
            "capture_group_log_probs": capture_group_log_probs,
        }