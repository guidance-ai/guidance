import json
import os
from dataclasses import dataclass
from typing import Any, Generator, Optional, Tuple, Union

import llguidance  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from ._grammar import GrammarFunction, Join, Terminal
from ._schema import EngineCallResponse, LLInterpreterResponse
from .models._byte_tokenizer import ByteTokenizer
from .models._tokenizer import Tokenizer


@dataclass
class GenData:
    tokens: list[int]
    mask: NDArray[np.uint8]
    temperature: float

    def valid_next_tokens(self) -> list[int]:
        return np.where(self.mask)[0].tolist()


class LLParserException(Exception):
    pass


class InvalidTokenException(LLParserException):
    def __init__(self, token: int, valid_tokens: list[int], prompt_tokens: list[int]):
        self.token = token
        self.valid_tokens = valid_tokens
        self.prompt_tokens = prompt_tokens
        super().__init__(
            f"Invalid token {token}, expected one of {valid_tokens} after {prompt_tokens}"
        )


class LLParser:

    def __init__(
        self,
        grammar: Union[GrammarFunction, str],
        tokenizer: Tokenizer,
        prompt: bytes = b"",
        ensure_bos_token: bool = True,
    ):
        if isinstance(grammar, GrammarFunction):
            # we can't have a terminal as the root
            if isinstance(grammar, Terminal):
                grammar = Join([grammar])
            serialized_grammar = json.dumps(grammar.ll_serialize())
        else:
            serialized_grammar = grammar

        self.tokenizer = tokenizer
        self.ll_tokenizer = llguidance.LLTokenizer(
            llguidance.TokenizerWrapper(tokenizer)
        )
        self.ll_interpreter = llguidance.LLInterpreter(
            self.ll_tokenizer,
            serialized_grammar,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )
        self._generator = self._parse(prompt, ensure_bos_token)
        self._done = False

    def matched(self) -> bool:
        return self.ll_interpreter.is_accepting()

    def done(self) -> bool:
        return self._done

    def advance(
        self, token: Optional[int]
    ) -> Tuple[Optional[GenData], EngineCallResponse]:
        # TODO: return something lower level than EngineCallResponse?
        return self._generator.send(token)

    def _process_prompt(self, prompt: bytes, ensure_bos_token: bool) -> list[int]:
        prompt_tokens = self.ll_interpreter.process_prompt(
            self.tokenizer.encode(prompt)
        )
        if (
            ensure_bos_token
            and self.tokenizer.bos_token is not None
            and prompt_tokens[:1] != [self.tokenizer.bos_token_id]
        ):
            # add the beginning of sequence token if needed
            prompt_tokens = [self.tokenizer.bos_token_id] + prompt_tokens

        return prompt_tokens

    def _parse(
        self,
        prompt: bytes,
        ensure_bos_token: bool,
    ) -> Generator[Tuple[Optional[GenData], EngineCallResponse], Optional[int], None]:
        tokens = self._process_prompt(prompt=prompt, ensure_bos_token=ensure_bos_token)

        while not self._done:
            mask, resp = self.ll_interpreter.mid_process()
            r = LLInterpreterResponse.model_validate_json(resp)
            self._done = r.stop
            response = r.progress.to_engine_call_response()

            if mask is not None:
                assert not self._done
                assert r.temperature is not None
                gen_data = GenData(
                    # TODO: be careful and return a copy of tokens?
                    tokens=tokens,
                    mask=np.frombuffer(mask, dtype=np.uint8),
                    temperature=r.temperature,
                )
                # Send caller the mask and response; wait for token
                token = yield (gen_data, response)
                # TODO: better exception handling (ParserException?)
                if token is None:
                    raise LLParserException("Expected token, got None")
                if not mask[token]:
                    # Note: we could punt this probem to ll_interpreter.post_process,
                    # but it's a bit clearer to handle it here
                    raise InvalidTokenException(token, gen_data.valid_next_tokens(), tokens)
            else:
                gen_data = None
                token = yield (gen_data, response)
                if token is not None:
                    raise LLParserException(f"Expected None, got token {token}")

            backtrack, ff_tokens = self.ll_interpreter.post_process(token)
            if backtrack:
                tokens = tokens[:-backtrack]
            tokens = tokens + ff_tokens


class ByteParserException(Exception):
    def __init__(self, *args, **kwargs):
        self.current_byte = kwargs.pop("current_byte", None)
        self.allowed_bytes = kwargs.pop("allowed_bytes", None)
        self.consumed_bytes = kwargs.pop("consumed_bytes", None)
        super().__init__(*args, **kwargs)


class ByteParser:
    def __init__(
        self,
        grammar: GrammarFunction,
        prompt: bytes = b"",
        ensure_bos_token: bool = True,
    ):
        self.tokenizer = ByteTokenizer()
        self.ll_parser = LLParser(grammar, self.tokenizer, prompt, ensure_bos_token)
        self.bytes = b""
        self.gen_data: Optional[GenData] = None
        self.pos = 0
        self._variables: dict[str, Any] = {}
        self._variables_log_probs: dict[str, Any] = {}
        self.consume_bytes(prompt)

    def matched(self) -> bool:
        if self.pos < len(self.bytes):
            return False
        return self.ll_parser.matched()

    def valid_next_bytes(self) -> set[bytes]:
        if self.pos < len(self.bytes):
            return {self.bytes[self.pos : self.pos + 1]}
        if self.gen_data is None:
            return set()
        return {
            bytes([t])
            for t in self.gen_data.valid_next_tokens()
            if t != self.tokenizer.eos_token_id
        }

    def next_byte_mask(self) -> NDArray[np.uint8]:
        mask = np.zeros(256, dtype=np.uint8)
        for t in self.valid_next_bytes():
            mask[t[0]] = 1
        return mask

    def consume_bytes(self, bts: bytes) -> None:
        # Run underlying ll_parser and fast-forward all of our bytes
        # until we have a "choice" (generation step) to make
        while self.gen_data is None and not self.ll_parser.done():
            self.gen_data, response = self.ll_parser.advance(None)
            self._update_capture(response)
            self.bytes += response.new_bytes

        if not bts:
            return

        b = bts[0]
        # If the current position is less than the length of the bytes, then we are in fast_forward mode
        # and we need to make sure that the byte we are consuming is the same as the byte at the current
        # position
        if self.pos < len(self.bytes):
            if b != self.bytes[self.pos]:
                next_byte = self.bytes[self.pos : self.pos + 1]
                raise ByteParserException(
                    f"Expected byte {next_byte!r} (fast_forward), got {bytes([b])!r}",
                    current_byte=bytes([b]),
                    allowed_bytes={next_byte},
                    consumed_bytes=self.bytes[: self.pos],
                )
            # Byte was good, move to the next byte
            self.pos += 1
            self.consume_bytes(bts[1:])
        else:
            # If we are here, then we are either in generation mode or we are done.
            if self.gen_data is None:
                # TODO: may run into trouble here if we need to backtrack
                assert self.ll_parser.done()
                assert not self.valid_next_bytes()
                raise ByteParserException(
                    f"Expected end of input, got {bytes([b])!r}",
                    current_byte=bytes([b]),
                    allowed_bytes=set(),
                    consumed_bytes=self.bytes[: self.pos],
                )
            # We're in generation mode. Assure that the byte is one of the valid next bytes
            valid_next_tokens = self.gen_data.valid_next_tokens()
            if b not in valid_next_tokens:
                valid_next_bytes = self.valid_next_bytes()
                raise ByteParserException(
                    f"Expected one of the following bytes: {valid_next_bytes!r}, got {bytes([b])!r}",
                    current_byte=bytes([b]),
                    allowed_bytes=valid_next_bytes,
                    consumed_bytes=self.bytes[: self.pos],
                )
            # Byte was good, have ll_parser consume it so we can advance further
            self.gen_data, response = self.ll_parser.advance(b)
            self._update_capture(response)
            self.bytes += response.new_bytes

            # Run consume_bytes to advance ll_parser and consume the next byte
            self.consume_bytes(bts)

    def force_done(self):
        if not self.matched():
            raise ByteParserException("Hit end of input before reaching a valid state")
        if self.ll_parser.done():
            return

        self.gen_data, response = self.ll_parser.advance(self.tokenizer.eos_token_id)
        self._update_capture(response)
        self.bytes += response.new_bytes
        if not self.ll_parser.done() or not self.matched():
            raise ByteParserException("Hit end of input before reaching a valid state")

    def get_captures(self):
        return self._variables, self._variables_log_probs

    def _update_capture(self, response: EngineCallResponse):
        # Stolen from model. TODO: refactor to share code
        for k in response.capture_groups:
            v = response.capture_groups[k]

            # see if we are in a list_append mode
            if isinstance(v, list):
                for i, inner_v in enumerate(v):
                    # convert to a string if possible
                    # TODO: will need to not just always do this once we support images etc.
                    try:
                        inner_v = (
                            inner_v.decode("utf8")
                            if isinstance(inner_v, bytes)
                            else inner_v
                        )
                    except UnicodeDecodeError:
                        pass

                    if k not in self._variables or not isinstance(
                        self._variables[k], list
                    ):
                        self._variables[k] = []
                        self._variables_log_probs[k] = []
                    self._variables[k].append(inner_v)
                    self._variables_log_probs[k].append(
                        response.capture_group_log_probs[k][i]
                    )

            # ...or standard assignment mode
            else:
                # convert to a string if possible
                # TODO: will need to not just always do this once we support images etc.
                try:
                    v = v.decode("utf8") if isinstance(v, bytes) else v
                except UnicodeDecodeError:
                    pass
                self._variables[k] = v
                self._variables_log_probs[k] = response.capture_group_log_probs[k]
