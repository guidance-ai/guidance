# TODO(nopdive): This module requires a memory review.

import logging
import time
import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Iterator, Optional

import numpy as np

from ..._parser import TokenParser
from ..._schema import (
    EngineOutput,
    EngineResponse,
    GenToken,
    GuidanceEngineMetrics,
)

from ...registry import get_exchange
from ..._utils import log_cleanup, log_init, softmax
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
        grammar: str,
        ensure_bos_token: bool = True,
        echo: bool = True,
    ) -> Iterator[EngineResponse]:
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

        tokens = self.tokenizer.encode(state.prompt.encode("utf-8"))

        parser = TokenParser(
            grammar,
            tokenizer=self.tokenizer,
            enable_backtrack=self.enable_backtrack,
            enable_ff_tokens=self.enable_ff_tokens,
        )

        engine_output = None
        logits_lat_ms = 0
        while not parser.done():
            t0 = time.time()

            recode = False
            if engine_output is None:
                prefix_tokens, backtrack, ff_tokens, mask_fut = parser.process_prompt(
                    prompt_tokens=tokens,
                    ensure_bos_token=ensure_bos_token,
                )
                if prefix_tokens:
                    tokens = prefix_tokens + tokens
                    recode = True
            else:
                backtrack, ff_tokens, mask_fut = parser.advance(
                    token_id=engine_output.issued_token.token_id
                )

            if backtrack:
                backtracked_bytes = self.tokenizer.decode(tokens[-backtrack:])
                tokens = tokens[:-backtrack]
            else:
                backtracked_bytes = b""
            tokens += ff_tokens

            if recode:
                # Only necessary when we add a prefix (bos token), which can only happen once
                # per loop. Needs to happen after adding ff_tokens to maintain associativity of
                # (model + prompt) + grammar == model + (prompt + grammar)
                tokens = self.tokenizer.recode(tokens)

            # Note that has_pending_stop implies that the response is a stop response,
            # but the converse is not true. We can therefore avoid some (but not all)
            # unnecessary calls to get_logits on the final iteration.
            has_pending_stop = parser.has_pending_stop()

            if not has_pending_stop:
                t1 = time.time()
                logits = self.get_logits(token_ids=tokens)
                logits_lat_ms = (time.time() - t1) * 1000
            else:
                # Avoid calling get_logits if we know we won't use it
                logits = None

            # Important: don't wait on this future until after getting the logits;
            # this allows the mask to be built concurrently with model inference
            mask, ll_response = mask_fut.result()
            legacy_engine_response = ll_response.progress.to_engine_call_response()

            ff_lat_ms = (time.time() - t0) * 1000
            if not ll_response.stop:
                # Logit latency will go into the NEXT token
                # if it exists
                ff_lat_ms -= logits_lat_ms

            gen_tokens = []
            if engine_output is None:
                for token_id in ff_tokens:
                    gen_tokens.append(
                        GenToken(
                            token_id=token_id,
                            bytes=self.tokenizer.decode([token_id]),
                            # amortize latency
                            latency_ms=ff_lat_ms/len(ff_tokens),
                            is_input=True,
                        )
                    )
            else:
                gen_tokens.append(engine_output.issued_token)
                if backtrack or ff_tokens[:1] != [engine_output.issued_token.token_id]:
                    engine_output.issued_token.is_backtracked = True
                    ff_start_index = 0
                else:
                    ff_start_index = 1
                for token_id in ff_tokens[ff_start_index:]:
                    gen_tokens.append(
                        GenToken(
                            token_id=token_id,
                            bytes=self.tokenizer.decode([token_id]),
                            # amortize latency
                            latency_ms=ff_lat_ms/len(ff_tokens[ff_start_index:]),
                            is_force_forwarded=True,
                        )
                    )

            engine_response = EngineResponse(
                new_bytes=legacy_engine_response.new_bytes,
                backtrack_bytes=backtracked_bytes,
                capture_groups=legacy_engine_response.capture_groups,
                capture_group_log_probs=legacy_engine_response.capture_group_log_probs,
                backtrack=backtrack,
                tokens=gen_tokens,
            )

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
                logits = self.get_logits(token_ids=token_ids)
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
                    bytes=self.tokenizer.decode([token_id]),
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
                    bytes=self.tokenizer.decode([token]),
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
                bytes=self.tokenizer.decode([sampled_index]),
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

    @abstractmethod
    def get_logits(self, token_ids: list[int]) -> np.ndarray:
        pass

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


