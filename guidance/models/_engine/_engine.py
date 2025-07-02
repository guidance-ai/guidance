# TODO(nopdive): This module requires a memory review.

import logging
import time
from abc import ABC, abstractmethod
from typing import Generator, Optional, TypedDict

import numpy as np
from numpy.typing import NDArray

from ..._parser import TokenParser
from ..._schema import EngineOutput, EngineResponse, GenToken, SamplingParams, TokenUsage
from ..._utils import apply_min_p_filter, apply_top_k_and_top_p_filter, log_init, softmax, apply_repetition_penalty
from ._state import EngineState
from ._tokenizer import Tokenizer

logger = logging.getLogger(__name__)

_TEMPERATURE_EPSILON = 0.0001


class LogitsOutput(TypedDict):
    logits: NDArray
    n_tokens: int
    n_cached: int


class Engine(ABC):
    """The engine owns the inference computation and is used/created by the Model class.

    Engine objects represent the expensive parts of inference. While Model objects are cheap and do not
    need to know about the tokenizer or the model parameters, Engine objects know about both. Many
    Model objects can reference a single Engine object. Engine objects can also be hidden behind a
    Server so a single server can serve many clients' model objects through a single Engine object.
    """

    def __init__(
        self, tokenizer: Tokenizer, enable_backtrack=True, enable_ff_tokens=True, enable_monitoring=True, **kwargs
    ):
        from ...registry import get_monitor

        self.tokenizer = tokenizer
        self._enable_backtrack = enable_backtrack
        self._enable_ff_tokens = enable_ff_tokens
        self._enable_monitoring = enable_monitoring
        self._top_k = kwargs.get("top_k", 5)

        if self._enable_monitoring:
            # Idempotent start
            _ = get_monitor()

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
        return self.tokenizer.chat_template()  # Instantiate the class before returning to client for now

    def __call__(
        self,
        state: EngineState,
        grammar: str,
        ensure_bos_token: bool = True,
        echo: bool = True,
        sampling_params: Optional[SamplingParams] = None,
    ) -> Generator[EngineResponse, None, TokenUsage]:
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
        sampling_params: Optional[SamplingParams]
            Additional sampling parameters to apply to the logits.
        """
        # Note: t0 will get reset further down in the loop, just after the break condition
        _t0 = time.time()
        t0 = _t0

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

        last_temperature = 1.0
        engine_output = None
        usage = TokenUsage(round_trips=1, ff_tokens=0)

        while not parser.done():
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
                backtrack, ff_tokens, mask_fut = parser.advance(token_id=engine_output.issued_token.token_id)

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

            # We can avoid a final get_logits call in the case that:
            # 1. The parser has a pending stop
            # 2. There are no ff_tokens (except for our last generated token)
            # TODO: allow avoiding final forward pass if metrics are disabled
            # and we have a pending stop
            if parser.has_pending_stop() and (
                (not ff_tokens)
                or (
                    len(ff_tokens) == 1
                    and engine_output is not None
                    and ff_tokens[0] == engine_output.issued_token.token_id
                )
            ):
                logits = None
                logits_lat_ms = 0.0
            else:
                t1 = time.time()
                logits_output = self.get_logits(token_ids=tokens, include_all_uncached_tokens=True)
                logits = logits_output["logits"]
                usage.input_tokens += logits_output["n_tokens"]
                usage.cached_input_tokens += logits_output["n_cached"]
                if logits_output["n_tokens"] > logits_output["n_cached"]:
                    usage.forward_passes += 1
                else:
                    usage.cached_output_tokens += 1
                logits_lat_ms = (time.time() - t1) * 1000

            # Important: don't wait on this future until after getting the logits;
            # this allows the mask to be built concurrently with model inference
            mask, ll_response = mask_fut.result()
            legacy_engine_response = ll_response.progress.to_engine_call_response()

            if logits is not None:
                # Not the last one -- that's for the *next* token.
                ff_logits = logits[-len(ff_tokens) - 1 : -1]
                ff_probs = (
                    softmax(ff_logits)
                    if last_temperature < _TEMPERATURE_EPSILON
                    else softmax(ff_logits / last_temperature)
                )

                if len(ff_tokens) == len(tokens) and ff_probs.shape[0] == len(ff_tokens) - 1:
                    # We didn't have a BOS token, so we need to fake the first token prob (say... 1?)
                    ff_probs = np.pad(ff_probs, [(1, 0), (0, 0)], mode="constant", constant_values=1.0)
                elif ff_probs.shape[0] < len(ff_tokens):
                    # Not enough logits were returned despite include_all_uncached_tokens=True, probably due to
                    # using a mock model that doesn't bother to return logits for uncached tokens (all are uncached...)
                    ff_probs = np.pad(
                        ff_probs,
                        [(len(ff_tokens) - ff_probs.shape[0], 0), (0, 0)],
                        mode="constant",
                        constant_values=np.nan,
                    )
            else:
                # really just for mypy -- we shouldn't need this
                ff_probs = np.empty(shape=())

            ff_lat_ms = (time.time() - t0) * 1000
            if not ll_response.stop:
                # If we're not stopping, the logit latency will go into the next generated token
                ff_lat_ms -= logits_lat_ms

            gen_tokens = []
            if engine_output is None:
                # Note: intentionally not counting ff_tokens here, as we're counting these
                # all just as input tokens
                for i, token_id in enumerate(ff_tokens):
                    gen_tokens.append(
                        GenToken(
                            token_id=token_id,
                            bytes=self.tokenizer.decode([token_id]),
                            prob=ff_probs[i, token_id],
                            latency_ms=ff_lat_ms / len(ff_tokens),
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
                ff_tokens = ff_tokens[ff_start_index:]

                # Just update ff tokens here -- usage for engine_output has already been
                # handled where we got logits above
                usage.ff_tokens += len(ff_tokens)
                for i, token_id in enumerate(ff_tokens, start=ff_start_index):
                    gen_tokens.append(
                        GenToken(
                            token_id=token_id,
                            bytes=self.tokenizer.decode([token_id]),
                            prob=ff_probs[i, token_id],
                            latency_ms=ff_lat_ms / len(ff_tokens),
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
            # Reset time down here instead of at the top of the loop in order to make sure
            # we take sampling time into account
            t0 = time.time()

            # Help the type checker: assert that everything we need to get the next token is not None
            assert logits is not None
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
                logits=logits[-1, :],
                logits_lat_ms=logits_lat_ms,
                token_ids=tokens,
                mask=mask_for_sampling,
                temperature=ll_response.temperature,
                k=self._top_k,
                force_return_unmasked_probs=echo,
                sampling_params=sampling_params,
            )
            last_temperature = ll_response.temperature

            if can_finish_early and not mask[engine_output.issued_token.token_id]:
                # Type checker needs some help
                assert self.tokenizer.eos_token_id is not None
                engine_output.issued_token.token_id = self.tokenizer.eos_token_id

        usage.total_latency_ms += (time.time() - _t0) * 1000
        return usage

    def get_next_token_with_top_k(
        self,
        logits: NDArray,
        logits_lat_ms: float,
        token_ids: list[int],
        mask: Optional[bytes],
        temperature: float,
        k: int = 5,
        force_return_unmasked_probs: bool = False,
        sampling_params: Optional[SamplingParams] = None,
    ) -> EngineOutput:
        """Get the next token and associated top-k tokens from the engine.

        Parameters
        -------
        logits : Optional[NDArray]
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
        sampling_params : Optional[SamplingParams]
            Additional sampling parameters to apply to the logits.

        Returns
        -------
        EngineOutput
            The output from the model.
        """

        def get_top_k(_probs: NDArray, _k: int = 5) -> list[GenToken]:
            top_k_indices = _probs.argpartition(-_k)[-_k:]
            top_k_probs = _probs[top_k_indices]

            top_k_tokens = [
                GenToken(
                    token_id=token,
                    prob=prob,
                    bytes=self.tokenizer.decode([token]),
                    latency_ms=logits_lat_ms,
                    is_generated=True,
                )
                for token, prob in zip(top_k_indices, top_k_probs)
                if prob > 0
            ]
            # Sort by probability in descending order, as above argpartition
            # does not guarantee order. Sorting the smaller array is faster.
            return sorted(top_k_tokens, key=lambda x: x.prob, reverse=True)

        # compute top-k without masking
        filtered_logits = np.array(logits)
        filtered_logits = apply_repetition_penalty(token_ids, filtered_logits, sampling_params)
        filtered_logits = filtered_logits if temperature < _TEMPERATURE_EPSILON else filtered_logits / temperature
        filtered_logits = apply_min_p_filter(filtered_logits, sampling_params)
        filtered_logits = apply_top_k_and_top_p_filter(filtered_logits, sampling_params)
        probs = softmax(filtered_logits)

        top_k: list[GenToken] = []
        if force_return_unmasked_probs:
            top_k = get_top_k(probs, k)

        # compute top-k with masking
        masked_top_k: list[GenToken] = []
        if mask is not None:
            mask = np.frombuffer(mask, dtype=np.uint8)
            masked_logits = np.where(mask != 0, logits, -np.inf)

            if temperature < _TEMPERATURE_EPSILON:
                masked_logits = np.where(masked_logits == np.max(masked_logits), 0, -np.inf)
            else:
                masked_logits /= temperature

            masked_logits = apply_min_p_filter(masked_logits, sampling_params)
            masked_logits = apply_top_k_and_top_p_filter(masked_logits, sampling_params)

            masked_probs = softmax(masked_logits)
            masked_top_k = get_top_k(masked_probs, k)

        if temperature < _TEMPERATURE_EPSILON:
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
                latency_ms=logits_lat_ms,
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
    def get_logits(self, token_ids: list[int], include_all_uncached_tokens: bool = False) -> LogitsOutput:
        """
        Get the logits for the given token ids.
        If include_all_uncached_tokens is True:
            logits for all uncached tokens will be returned, i.e.
            the return value's shape will be `(len(tokens)-num_cached, vocab_size)`.
        If include_all_uncached_tokens is False:
            logits for the last token will be returned, i.e.
            the return value's shape will be `(1, vocab_size)`.
        """
        pass
