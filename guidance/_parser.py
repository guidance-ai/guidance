import os
from typing import Any, Generator, Optional, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, Future

import llguidance  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from ._schema import EngineOutput, GenData, EngineCallResponse, GenToken, LLInterpreterResponse, LLGrammar

if TYPE_CHECKING:
    from .models._engine import Tokenizer

class TokenParserException(Exception):
    pass


class InvalidTokenException(TokenParserException):
    def __init__(self, token: int, valid_tokens: list[int], prompt_tokens: list[int]):
        self.token = token
        self.valid_tokens = valid_tokens
        self.prompt_tokens = prompt_tokens
        super().__init__(
            f"Invalid token {token}, expected one of {valid_tokens} after {prompt_tokens}"
        )


class TokenParser:

    def __init__(
        self,
        grammar: LLGrammar,
        tokenizer: "Tokenizer",
        prompt: bytes = b"",
        ensure_bos_token: bool = True,
        enable_backtrack: bool = True,
        enable_ff_tokens: bool = True,
    ):
        self.tokenizer = tokenizer
        self.ll_tokenizer = llguidance.LLTokenizer(llguidance.TokenizerWrapper(tokenizer))
        self.ll_interpreter = llguidance.LLInterpreter(
            self.ll_tokenizer,
            grammar.model_dump_json(),
            enable_backtrack,
            enable_ff_tokens,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )
        self._threadpool = ThreadPoolExecutor(max_workers=1)
        self._generator = self._parse(prompt, ensure_bos_token)
        self._done = False
        self._has_pending_stop = False

    def is_accepting(self) -> bool:
        return self.ll_interpreter.is_accepting()

    def done(self) -> bool:
        return self._done

    def advance(
        self, engine_output: Optional[EngineOutput]
    ) -> tuple[
        list[int], 
        Future[tuple[Optional[bytes], LLInterpreterResponse]], 
        int
    ]:
        if self.done():
            raise TokenParserException("Cannot advance on a done parser")
        
        return self._generator.send(engine_output)
    
    def has_pending_stop(self) -> bool:
        return self._has_pending_stop

    def _process_prompt(self, prompt: bytes, ensure_bos_token: bool) -> tuple[list[int], int]:
        _prompt_tokens = self.tokenizer.encode(prompt)
        prompt_tokens = self.ll_interpreter.process_prompt(_prompt_tokens)
        if (
            ensure_bos_token
            and self.tokenizer.bos_token is not None
            and prompt_tokens[:1] != [self.tokenizer.bos_token_id]
        ):
            # add the beginning of sequence token if needed
            prompt_tokens = [self.tokenizer.bos_token_id] + prompt_tokens

        return self.tokenizer.recode(prompt_tokens)

    def compute_mask(self) -> tuple[Optional[bytes], LLInterpreterResponse]:
        mask, ll_response_string = self.ll_interpreter.compute_mask()
        ll_response = LLInterpreterResponse.model_validate_json(ll_response_string)
        return mask, ll_response

    def _parse(
        self,
        prompt: bytes,
        ensure_bos_token: bool,
    ) -> Generator[
        tuple[
            list[int], 
            Future[tuple[Optional[bytes], LLInterpreterResponse]],
            int
        ], Optional[EngineOutput], None
    ]:
        tokens = self._process_prompt(prompt=prompt, ensure_bos_token=ensure_bos_token)

        backtrack = 0
        engine_output = None
        ff_tokens = []
        while True:
            # Note: need to call/set has_pending_stop before spinning up the compute mask 
            # future as the two methods cannot be called concurrently
            self._has_pending_stop = self.ll_interpreter.has_pending_stop()
            compute_mask_future = self._threadpool.submit(self.compute_mask)

            # Send caller the mask and response; wait for token
            engine_output = yield (tokens, compute_mask_future, backtrack)

            # Upstairs should have already waited on this future
            mask, r = compute_mask_future.result()

            if r.stop:
                # This is the only case in which the mask is None
                assert mask is None
                # If we're done, our caller should NOT send us a token
                if engine_output is not None:
                    raise TokenParserException(f"Expected None, got token {engine_output.issued_token.token_id}")
                self._done = True
                break

            assert mask is not None
            assert r.temperature is not None

            if engine_output is None:
                raise TokenParserException("Expected token, got None")
            
            if not mask[engine_output.issued_token.token_id]:
                # Note: we could punt this probem to ll_interpreter.post_process,
                # but it's a bit clearer to handle it here
                raise InvalidTokenException(
                    token=engine_output.issued_token.token_id,
                    valid_tokens=[i for i in range(len(mask)) if mask[i]],
                    prompt_tokens=tokens
                )            

            backtrack, ff_tokens = self.ll_interpreter.commit_token(
                engine_output.issued_token.token_id
            )
            if backtrack:
                tokens = tokens[:-backtrack]
            tokens = tokens + ff_tokens

    def cleanup(self):
        # Rather than having our caller send us None at the end, we'll handle that internally
        # so we can (1) verify that the generator actually stops and (2) check the stop reason
        # and raise if needed
        if not self.done():
            try:
                self._generator.send(None)
            except StopIteration:
                pass
        if not self.done():
            raise TokenParserException("Tried to cleanup but parser is not done")
        stop_reason = self.ll_interpreter.stop_reason()
        if stop_reason not in {"NoExtension", "EndOfSentence"}:
            # Will raise if there is some "bad" stop reason (like hit token limit) OR we're NOT stopped.
            # TODO: raise specific exceptions for reasons such as MaxTokensTotal
            raise TokenParserException(f"Unexpected stop reason: {stop_reason}")

class ByteParserException(Exception):
    def __init__(self, *args, **kwargs):
        self.current_byte = kwargs.pop("current_byte", None)
        self.allowed_bytes = kwargs.pop("allowed_bytes", None)
        self.consumed_bytes = kwargs.pop("consumed_bytes", None)
        super().__init__(*args, **kwargs)


class ByteParser:
    def __init__(
        self,
        grammar: LLGrammar,
        prompt: bytes = b"",
        ensure_bos_token: bool = True,
    ):
        # TODO: figure out this circular import
        from .models._byte_tokenizer import ByteTokenizer
        self.tokenizer = ByteTokenizer()
        self.token_parser = TokenParser(grammar, self.tokenizer, prompt, ensure_bos_token)
        self.bytes = b""
        self.gen_data: Optional[GenData] = None
        self.pos = 0
        self._variables: dict[str, Any] = {}
        self._variables_log_probs: dict[str, Any] = {}
        # Prime the parser
        self._advance(None)
        self.consume_bytes(prompt)

    def matched(self) -> bool:
        if self.pos < len(self.bytes):
            return False
        return self.token_parser.is_accepting()

    def valid_next_bytes(self) -> set[bytes]:
        if self.pos < len(self.bytes):
            return {self.bytes[self.pos : self.pos + 1]}
        if self.gen_data is None:
            return set()
        return {
            bytes([t]) for t in self.gen_data.valid_next_tokens if t != self.tokenizer.eos_token_id
        }

    def next_byte_mask(self) -> NDArray[np.uint8]:
        mask = np.zeros(256, dtype=np.uint8)
        for t in self.valid_next_bytes():
            mask[t[0]] = 1
        return mask

    def _advance(self, engine_output: Optional[EngineOutput]) -> None:
        tokens, compute_mask_future, _ = self.token_parser.advance(engine_output)
        mask, ll_response = compute_mask_future.result()
        if ll_response.stop:
            assert mask is None
            self.token_parser.cleanup()
            self.gen_data = None
        else:
            assert mask is not None
            assert ll_response.temperature is not None
            self.gen_data = GenData(
                tokens=tokens,
                mask=mask,
                temperature=ll_response.temperature,
            )
        response = ll_response.progress.to_engine_call_response()
        self._update_capture(response)
        self.bytes += response.new_bytes

    def consume_bytes(self, bts: bytes) -> None:
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
                assert self.token_parser.done()
                assert not self.valid_next_bytes()
                raise ByteParserException(
                    f"Expected end of input, got {bytes([b])!r}",
                    current_byte=bytes([b]),
                    allowed_bytes=set(),
                    consumed_bytes=self.bytes[: self.pos],
                )
            # We're in generation mode. Assure that the byte is one of the valid next bytes
            if b not in self.gen_data.valid_next_tokens:
                valid_next_bytes = self.valid_next_bytes()
                raise ByteParserException(
                    f"Expected one of the following bytes: {valid_next_bytes!r}, got {bytes([b])!r}",
                    current_byte=bytes([b]),
                    allowed_bytes=valid_next_bytes,
                    consumed_bytes=self.bytes[: self.pos],
                )
            # Byte was good, have ll_parser consume it so we can advance further
            fake_engine_output = self.fake_engine_output(b)
            self._advance(fake_engine_output)

            # Run consume_bytes to advance ll_parser and consume the next byte
            self.consume_bytes(bts)

    def force_done(self):
        if not self.matched():
            raise ByteParserException("Hit end of input before reaching a valid state")
        if self.token_parser.done():
            return

        fake_engine_output = self.fake_engine_output(self.tokenizer.eos_token_id)
        self._advance(fake_engine_output)
        if not self.token_parser.done() or not self.matched():
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
                        inner_v = inner_v.decode("utf8") if isinstance(inner_v, bytes) else inner_v
                    except UnicodeDecodeError:
                        pass

                    if k not in self._variables or not isinstance(self._variables[k], list):
                        self._variables[k] = []
                        self._variables_log_probs[k] = []
                    self._variables[k].append(inner_v)
                    self._variables_log_probs[k].append(response.capture_group_log_probs[k][i])

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

    def fake_engine_output(self, token_id: int) -> EngineOutput:
        fake_issued_token = GenToken(
            token_id=token_id,
            prob=1.0,
            text=self.tokenizer.decode([token_id]).decode("utf-8"),
            latency_ms=0,
            is_generated=True,
        )
        fake_engine_output = EngineOutput(
            issued_token=fake_issued_token,
            top_k=None,
            masked_top_k=None,
        )
        return fake_engine_output
