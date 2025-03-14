# TODO(nopdive): This module requires a memory review.

import logging
import time
import weakref
from abc import ABC
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Sequence

import numpy as np

from ..._parser import TokenParser
from ..._schema import (
    EngineCallResponse,
    EngineOutput,
    GenToken,
    GuidanceEngineMetrics,
    LLGrammar,
)
from ...registry import get_exchange
from ..._utils import log_cleanup, log_init, softmax, to_utf8_or_bytes_string
from ...visual import (
    ExecutionCompletedMessage,
    ExecutionStartedMessage,
    GuidanceMessage,
    OutputRequestMessage,
    MetricMessage,
)
from ._state import EngineState
from ._tokenizer import Tokenizer

if TYPE_CHECKING:
    from .._base._model import Model

logger = logging.getLogger(__name__)


def _engine_cleanup(msg_recv: Callable[[GuidanceMessage], None], log_msg: str):
    get_exchange().unsubscribe(msg_recv)
    log_cleanup(log_msg)


def _wrapped_msg_recv(engine_weak_ref: weakref.ref) -> Callable[[GuidanceMessage], None]:
    def closure(message):
        return _msg_recv(engine_weak_ref, message)

    return closure


def _msg_recv(engine_weakref: weakref.ReferenceType, message: GuidanceMessage) -> None:
    # NOTE(nopdive): This is run on a background thread.

    engine = engine_weakref()
    if engine is None:
        return

    if isinstance(message, MetricMessage):
        return

    # logger.debug(f"ENGINE({id(engine)}):msg_recv:{message}")
    if isinstance(message, ExecutionStartedMessage):
        pass
    elif isinstance(message, ExecutionCompletedMessage) and message.is_err:
        pass
    elif isinstance(message, ExecutionCompletedMessage):
        pass
    elif isinstance(message, OutputRequestMessage):
        pass


class Engine(ABC):
    """The engine owns the inference computation and is used/created by the Model class.

    Engine objects represent the expensive parts of inference. While Model objects are cheap and do not
    need to know about the tokenizer or the model parameters, Engine objects know about both. Many
    Model objects can reference a single Engine object. Engine objects can also be hidden behind a
    Server so a single server can serve many clients' model objects through a single Engine object.
    """

    def __init__(self, tokenizer: Tokenizer, compute_log_probs=False, enable_backtrack=True, enable_ff_tokens=True,
                 enable_monitoring=True, **kwargs):
        from ...registry import get_monitor

        self.tokenizer = tokenizer
        self.compute_log_probs = compute_log_probs
        self._enable_backtrack = enable_backtrack
        self._enable_ff_tokens = enable_ff_tokens
        self._enable_monitoring = enable_monitoring
        self._top_k = kwargs.get("top_k", 5)

        # TODO(nopdive): Remove on refactor.
        self.metrics = GuidanceEngineMetrics()

        if self._enable_monitoring:
            # Idempotent start
            _ = get_monitor()

        msg_recv = _wrapped_msg_recv(weakref.ref(self))
        get_exchange().subscribe(msg_recv)

        weakref.finalize(
            self,
            _engine_cleanup,
            msg_recv,
            f"engine({id(self)})",
        )
        log_init(f"engine({id(self)})")

    # These need to be properties because once an Engine is started, you can't change their behavior.
    @property
    def enable_backtrack(self):
        return self._enable_backtrack

    @property
    def enable_ff_tokens(self):
        return self._enable_ff_tokens

    @property
    def enable_monitoring(self):
        return self._enable_monitoring

    def get_chat_template(
        self,
    ):  # TODO [HN]: Add more logic here...should we instantiate class here? do we even need to?
        return (
            self.tokenizer.chat_template()
        )  # Instantiate the class before returning to client for now

    def reset_metrics(self):
        self.metrics = GuidanceEngineMetrics()

    def __call__(
        self,
        state: EngineState,
        grammar: LLGrammar,
        ensure_bos_token: bool = True,
        echo: bool = True,
    ) -> Iterator[EngineCallResponse]:
        """Main entry point for the inference-parser loop. Yields EngineCallResponse objects as
        the parser advances through the grammar.

        Parameters
        ----------
        prompt : str or Parser
            This is represents the current state of a guidance parser that will be extended
            using the passed grammar. If a string is given then we assume the previous parser
            state is just a fixed string prompt, if a full Parser is given then we extend that
            parser by appending the new grammar to the parser's current grammar and then
            inferencing the model. (TODO: implement full parser extension support)
        grammar: Function
            Grammar (RawFunction or GrammarFunction) used to extend the prompt.
        ensure_bos_token: bool
            Ensures that the prompt ends with the BOS token.
        """
        # TODO: Pass these to get_logits
        # images = state.images
        # audio = state.audio
        # videos = state.videos

        parser = TokenParser(
            grammar,
            tokenizer=self.tokenizer,
            prompt=state.prompt.encode("utf-8"),
            ensure_bos_token=ensure_bos_token,
            enable_backtrack=self.enable_backtrack,
            enable_ff_tokens=self.enable_ff_tokens,
        )

        has_get_logits = True
        engine_output = None
        logits_lat_ms = 0
        delayed_bytes = b""
        delayed_engine_outputs: list[EngineOutput] = []
        while not parser.done():
            t0 = time.time()

            tokens, mask_fut, backtrack = parser.advance(engine_output)

            # Note that has_pending_stop implies that the response is a stop response,
            # but the converse is not true. We can therefore avoid some (but not all)
            # unnecessary calls to get_logits on the final iteration.
            has_pending_stop = parser.has_pending_stop()

            if has_get_logits and not has_pending_stop:
                try:
                    t0 = time.time()
                    logits = self.get_logits(token_ids=tokens)
                    logits_lat_ms = (time.time() - t0) * 1000
                except NotImplementedError:
                    # Permanently fall-back to get_next_token if get_logits is not implemented
                    has_get_logits = False
                    logits = None
                    logits_lat_ms = 0
            else:
                logits = None

            # Important: don't wait on this future until after getting the logits;
            # this allows the mask to be built concurrently with model inference
            mask, ll_response = mask_fut.result()

            engine_response = ll_response.progress.to_engine_call_response()
            engine_response.backtrack = backtrack
            if engine_output:
                engine_response.engine_outputs.append(engine_output)

            # NOTE (loc): Temporary solution to quickly check which segments are generated and which are force-forwarded to animate visualizations on the UI
            # These tokens in chunk will not be used for final visualization
            # TODO: This should be handled by the interpreter
            if echo and engine_response.new_bytes:
                try:
                    _new_bytes = delayed_bytes + engine_response.new_bytes
                    _tokens = parser.tokenizer.encode(_new_bytes)
                    delayed_bytes = b""
                except UnicodeDecodeError:
                    # similar to what we did in _run_stateless function, if we could not decode current bytes
                    # we will delay until we can decode them
                    delayed_bytes += engine_response.new_bytes
                    if engine_output:
                        engine_response.engine_outputs.pop()
                        delayed_engine_outputs.append(engine_output)

                if not delayed_bytes:
                    ff_token_start_idx = 1
                    if engine_output is None and len(delayed_engine_outputs) == 0:
                        ff_token_start_idx = 0
                    elif (
                        engine_output.issued_token.token_id == _tokens[0]
                        and len(delayed_engine_outputs) == 0
                    ):
                        # this is generated
                        engine_response.generated_bytes = parser.tokenizer.decode([_tokens[0]])
                        engine_output.issued_token.is_generated = True
                        engine_response.generated_tokens.append(engine_output.issued_token)
                    else:
                        # handle delayed bytes
                        engine_outputs = (
                            delayed_engine_outputs + [engine_output] if engine_output else []
                        )
                        engine_output_tokens = [e.issued_token.token_id for e in engine_outputs]

                        generated = to_utf8_or_bytes_string(
                            parser.tokenizer.decode(engine_output_tokens)
                        )
                        force_forwarded = _new_bytes.decode("utf-8")

                        if force_forwarded.startswith(generated):
                            engine_output_tokens = np.array(engine_output_tokens)
                            ff_tokens = np.array(_tokens)

                            # check if engine_output_tokens in ff_tokens
                            _idx = -1
                            for _i in range(0, len(ff_tokens) - len(engine_output_tokens) + 1):
                                if np.array_equal(
                                    engine_output_tokens,
                                    ff_tokens[_i : _i + len(engine_output_tokens)],
                                ):
                                    _idx = _i + len(engine_output_tokens)
                                    break

                            if _idx < 0:
                                ff_token_start_idx = 0
                            else:
                                # all previous tokens before _idx are generated
                                engine_response.generated_bytes = parser.tokenizer.decode(
                                    ff_tokens[:_idx]
                                )
                                idx_in_engine_output_tokens = 0
                                for _i in range(_idx):
                                    matching_engine_output = None
                                    if (
                                        _tokens[_i]
                                        == engine_output_tokens[idx_in_engine_output_tokens]
                                    ):
                                        matching_engine_output = engine_outputs[
                                            idx_in_engine_output_tokens
                                        ]
                                        idx_in_engine_output_tokens += 1
                                    engine_response.generated_tokens.append(
                                        GenToken(
                                            token_id=_tokens[_i],
                                            prob=(
                                                1.0
                                                if not matching_engine_output
                                                else matching_engine_output.issued_token.prob
                                            ),
                                            text=(
                                                parser.tokenizer.decode([_tokens[_i]])
                                                if not matching_engine_output
                                                else matching_engine_output.issued_token.text
                                            ),
                                            latency_ms=(
                                                0.0
                                                if not matching_engine_output
                                                else matching_engine_output.issued_token.latency_ms
                                            ),
                                            is_generated=True,
                                        )
                                    )
                                ff_token_start_idx = _idx
                        else:
                            ff_token_start_idx = 0

                    if len(_tokens[ff_token_start_idx:]):
                        engine_response.force_forwarded_bytes = parser.tokenizer.decode(
                            _tokens[ff_token_start_idx:]
                        )
                        for _token in _tokens[ff_token_start_idx:]:
                            engine_response.force_forwarded_tokens.append(
                                GenToken(
                                    token_id=_token,
                                    prob=1.0,
                                    text=to_utf8_or_bytes_string(
                                        parser.tokenizer.decode([_token])
                                    ),
                                    latency_ms=0,
                                    is_force_forwarded=True,
                                )
                            )

                    delayed_engine_outputs = []
            elif not echo and engine_response.new_bytes:
                # do not collect tokens-metrics if echo is disabled
                engine_response.generated_bytes = engine_response.new_bytes
                engine_response.generated_tokens.clear()

            # process engine_response
            # NOTE (loc): We should not yield the engine_response if new_bytes are invalid utf-8 bytes
            # delayed bytes should be handled here in the engine
            yield engine_response

            if ll_response.stop:
                assert mask is None
                # May raise an exception if the parser is in an bad state!
                parser.cleanup()
                # Ensure we break AFTER yielding the final response
                break

            # If there was a pending stop, we should have broken out of the loop
            assert not has_pending_stop

            # Help the type checker: assert that everything we need to get the next token is not None
            assert mask is not None
            assert ll_response.temperature is not None

            can_finish_early = parser.is_accepting() and self.tokenizer.eos_token_id is not None

            if can_finish_early:
                # Type checker needs some help
                assert self.tokenizer.eos_token_id is not None
                # Should be equivalent to parser.is_accepting()
                assert mask[self.tokenizer.eos_token_id]
                # Whenever we are in an accepting state, we will allow the model to generate whatever it wants
                # but we will treat any "illegal" tokens as EOS, allowing the model to finish gracefully.
                # Hence, mask must be None
                mask_for_sampling = None
            else:
                mask_for_sampling = mask

            engine_output = self.get_next_token_with_top_k(
                logits=logits,
                logits_lat_ms=logits_lat_ms,
                token_ids=tokens,
                mask=mask_for_sampling,
                temperature=ll_response.temperature,
                k=self._top_k,
                force_return_unmasked_probs=echo,
            )

            if can_finish_early and not mask[engine_output.issued_token.token_id]:
                # Type checker needs some help
                assert self.tokenizer.eos_token_id is not None
                engine_output.issued_token.token_id = self.tokenizer.eos_token_id

    def get_next_token_with_top_k(
        self,
        logits: Optional[np.ndarray],
        logits_lat_ms: Optional[float],
        token_ids: list[int],
        mask: Optional[bytes],
        temperature: float,
        k: int = 5,
        force_return_unmasked_probs: bool = False,
    ) -> EngineOutput:
        """Get the next token and associated top-k tokens from the engine.

        Parameters
        -------
        logits : Optional[np.ndarray]
            The logits for the current token ids in the sequence.
            If None, the model will call get_logits to get the logits.
        logits_lat_ms: Optional[float]
            The time taken to compute the logits.
            If logits is None, the model will call get_logits to measure the time.
        token_ids : list[int]
            The current token ids in the sequence.
        mask : Optional[bytes]
            The mask to apply to the logits.
        temperature : float
            The temperature to apply to the logits.
        k : int
            The number of top-k tokens to return.
        force_return_unmasked_probs: bool
            If True, the top-k unmasked probabilities will be returned.

        Returns
        -------
        EngineOutput
            The output from the model.
        """

        if logits is None:
            t0 = time.time()
            try:
                logits = self.get_logits(token_ids)
            except NotImplementedError:
                # fallback to orignal get_next_token method
                _t0 = time.time()
                token_id = self.get_next_token(
                    token_ids=token_ids,
                    mask=mask,
                    temperature=temperature,
                )
                _lat = (time.time() - _t0) * 1000

                _issued_token = GenToken(
                    token_id=token_id,
                    prob=1.0,
                    text=to_utf8_or_bytes_string(self.tokenizer.decode([token_id])),
                    latency_ms=_lat,
                    is_generated=True,
                )

                return EngineOutput(
                    issued_token=_issued_token,
                    top_k=[_issued_token],
                    masked_top_k=[_issued_token] if mask is not None else None,
                )
            lat_ms = (time.time() - t0) * 1000
        else:
            lat_ms = logits_lat_ms

        def get_top_k(_probs: np.ndarray, _k: int = 5) -> list[GenToken]:
            top_k_indices = np.argsort(_probs)[::-1][:_k]
            top_k_probs = _probs[top_k_indices]

            return [
                GenToken(
                    token_id=token,
                    prob=prob,
                    text=to_utf8_or_bytes_string(self.tokenizer.decode([token])),
                    latency_ms=lat_ms,
                    is_generated=True,
                )
                for token, prob in zip(top_k_indices, top_k_probs)
                if prob > 0
            ]

        # compute top-k without masking
        probs = (
            softmax(np.array(logits))
            if temperature < 0.0001
            else softmax(np.array(logits) / temperature)
        )

        top_k: list[GenToken] = []
        if force_return_unmasked_probs:
            top_k = get_top_k(probs, k)

        # compute top-k with masking
        masked_top_k: list[GenToken] = []
        if mask is not None:
            # shift logits to [0 - max] range first and apply mask
            masked_logits = (logits - np.min(logits)) * np.frombuffer(mask, dtype=np.uint8)
            masked_probs = (
                softmax(masked_logits)
                if temperature < 0.0001
                else softmax(masked_logits / temperature)
            )
            masked_top_k = get_top_k(masked_probs, k)

        if temperature < 0.0001:
            if len(masked_top_k) > 0:
                issued_token = masked_top_k[0]
            else:
                if len(top_k) == 0:
                    top_k = get_top_k(probs, k)
                issued_token = top_k[0]
        else:
            # we need to sample from the probabilities
            if mask is None:
                sampled_index = np.random.choice(len(probs), p=probs)
                sampled_prob = probs[sampled_index]
            else:
                sampled_index = np.random.choice(len(masked_probs), p=masked_probs)
                sampled_prob = masked_probs[sampled_index]

            issued_token = GenToken(
                token_id=sampled_index,
                prob=sampled_prob,
                text=to_utf8_or_bytes_string(self.tokenizer.decode([sampled_index])),
                latency_ms=lat_ms,
                is_generated=True,
            )

        output = EngineOutput(
            issued_token=issued_token,
            top_k=top_k,
            masked_top_k=None if not masked_top_k else masked_top_k,
            is_backtracked=False,
        )

        return output

    def get_next_token(
        self, token_ids: list[int], mask: Optional[bytes], temperature: float
    ) -> int:
        # Prefer to implement get_logits over get_next_token as it allows for concurrent mask computation
        raise NotImplementedError

    def get_logits(self, token_ids: list[int]) -> np.ndarray:
        # Prefer to implement get_logits over get_next_token as it allows for concurrent mask computation
        raise NotImplementedError

    def get_per_token_topk_probs(self, token_ids: list[int], top_k: int = 5) -> list[GenToken]:
        """Get the top-k probabilities for each token in the sequence."""
        raise NotImplementedError

    def sample_with_temperature(
        self, logits: np.ndarray, mask: Optional[bytes], temperature: float
    ) -> int:
        if mask is not None:
            logits += np.frombuffer(mask, dtype=np.uint8)
        if temperature < 0.0001:
            return int(np.argmax(logits))
        # Get probabilities from softmax
        probabilities = softmax(logits / temperature)
        # Sample an index based on the probabilities
        sampled_index = np.random.choice(len(logits), p=probabilities)
        return sampled_index

    def _report_failed_match(self, prompt):
        """Note that this can be overridden by subclasses that have more likely reasons than a bug in the token set (like remote models)."""
        return Exception(
            "We can't consume any more tokens, but we are not yet done! Perhaps your model's token set is incomplete? This happened after the prompt:"
            + str(prompt[-40:])
        )


class ConstraintException(Exception):
    def __init__(self, *args, **kwargs):
        self.prompt = kwargs.pop("prompt", None)
        self.data = kwargs.pop("data", None)
        super().__init__(*args, **kwargs)


