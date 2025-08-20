# TODO(nopdive): This module requires a memory review.

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Generator, Optional, TypedDict, Union

import numpy as np
from jinja2 import BaseLoader, Environment
from numpy.typing import NDArray

from ..._parser import TokenParser
from ..._schema import EngineResponse, GenToken, GenTokenExtra, SamplingParams, TokenUsage
from ..._utils import apply_min_p_filter, apply_repetition_penalty, apply_top_k_and_top_p_filter, log_init, softmax
from ._state import EngineState
from ._tokenizer import Tokenizer

logger = logging.getLogger(__name__)

_TEMPERATURE_EPSILON = 1e-5


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
        self,
        tokenizer: Tokenizer,
        enable_backtrack=True,
        enable_ff_tokens=True,
        enable_monitoring=True,
        enable_token_probabilities=False,
        enable_top_k=False,
        top_k: int = 5,
    ):
        from ...registry import get_monitor

        self.tokenizer = tokenizer
        self._enable_backtrack = enable_backtrack
        self._enable_ff_tokens = enable_ff_tokens
        self._enable_monitoring = enable_monitoring
        self._enable_token_probabilities = enable_token_probabilities
        self._enable_top_k = enable_top_k
        self._top_k = top_k

        if enable_top_k and not enable_token_probabilities:
            raise ValueError("enable_top_k requires enable_token_probabilities to be True.")

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
        sampling_params: Optional[SamplingParams] = None,
    ) -> Generator[EngineResponse, None, TokenUsage]:
        """Main entry point for the inference-parser loop. Yields EngineCallResponse objects as
        the parser advances through the grammar.

        Parameters
        ----------
        state: EngineState
            The current state of the engine, including the prompt.
        grammar: Function
            Grammar (RawFunction or GrammarFunction) used to extend the prompt.
        ensure_bos_token: bool
            Ensures that the prompt ends with the BOS token.
        sampling_params: Optional[SamplingParams]
            Additional sampling parameters to apply to the logits.
        """
        t0 = time.monotonic()

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
        issued_token: Optional[GenToken] = None
        usage = TokenUsage(round_trips=1, ff_tokens=0)

        while not parser.done():
            t1 = time.monotonic()
            recode = False
            if issued_token is None:
                prefix_tokens, backtrack, ff_tokens, mask_fut = parser.process_prompt(
                    prompt_tokens=tokens,
                    ensure_bos_token=ensure_bos_token,
                )
                if prefix_tokens:
                    tokens = prefix_tokens + tokens
                    recode = True
            else:
                backtrack, ff_tokens, mask_fut = parser.advance(token_id=issued_token.token_id)

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

            if issued_token is not None:
                if backtrack or ff_tokens[:1] != [issued_token.token_id]:
                    issued_token.is_backtracked = True
                else:
                    # Remove the issued token from ff_tokens
                    ff_tokens = ff_tokens[1:]
                # Note: only need to update usage in this branch (issued_token is not None), as these ff_tokens
                # will otherwise just be counted as "input_tokens" when we call get_logits below
                usage.ff_tokens += len(ff_tokens)

            t2 = time.monotonic()
            parser_lat_ms = (t2 - t1) * 1000

            if parser.has_pending_stop() and (
                # There are no ff_tokens
                not ff_tokens
                # Monitoring is disabled
                or not self._enable_token_probabilities
            ):
                # We can skip the logits computation because it would only be used to enrich
                # the fast-forwarded tokens with probabilities for the sake of monitoring
                logits = None
            else:
                logits_output = self.get_logits(
                    token_ids=tokens, include_all_uncached_tokens=self._enable_token_probabilities
                )
                logits = logits_output["logits"]
                usage.input_tokens += logits_output["n_tokens"]
                usage.cached_input_tokens += logits_output["n_cached"]
                if logits_output["n_tokens"] > logits_output["n_cached"]:
                    usage.forward_passes += 1
                else:
                    usage.cached_output_tokens += 1

            t3 = time.monotonic()
            logits_lat_ms = (t3 - t2) * 1000

            # Important: don't wait on this future until after getting the logits;
            # this allows the mask to be built concurrently with model inference
            mask, ll_response, mask_compute_ms = mask_fut.result()
            # Mask time is the time it took to advance the parser plus the total time spent computing mask
            usage.mask_times_ms.append(parser_lat_ms + mask_compute_ms)
            if usage.ttfm_ms == 0:
                usage.ttfm_ms = (time.monotonic() - t0) * 1000
            # Mask overhead time is the time it took to advance the parser plus the total time spent waiting
            # on the mask future (i.e. time spent computing mask LESS the portion of that time parallelized with logits)
            t4 = time.monotonic()
            usage.mask_overheads_ms.append(parser_lat_ms + (t4 - t3) * 1000)

            legacy_engine_response = ll_response.progress.to_engine_call_response()

            ff_probs: Optional[NDArray] = None
            if logits is not None and self._enable_token_probabilities:
                # Exclude the "next token" logits
                # Note: may not have logits for all ff tokens if some prefix of them hit cache
                # Note: may have some extra here if something caused us to miss cache
                ff_logits = logits[-len(ff_tokens) - 1 : -1, :]
                # Avoid mutation of the original logits
                ff_logits = ff_logits.copy()

                if ff_logits.shape[0] > 0:
                    ff_logits_list: list[NDArray] = []
                    for i in range(ff_logits.shape[0]):
                        ff_logits_list.append(
                            apply_temp_and_sampling_params(
                                ff_logits[i, :],
                                tokens[: len(tokens) - ff_logits.shape[0] + i],
                                last_temperature,
                                sampling_params,
                            )
                        )
                    ff_logits = np.stack(ff_logits_list, axis=0)
                    ff_probs = softmax(ff_logits)

            # Note: ff_lat_ms includes parser_lat_ms (t2 - t1)
            ff_lat_ms = (time.monotonic() - t1) * 1000
            if not ll_response.stop:
                # If we're not stopping, the logit latency will go into the next generated token
                ff_lat_ms -= logits_lat_ms

            gen_tokens: list[GenTokenExtra] = []
            if issued_token is not None:
                gen_tokens.append(issued_token)

            for i, token_id in enumerate(ff_tokens):
                prob = float("nan")
                top_k: list[GenToken] = []
                if ff_probs is not None:
                    prob_ix = i + (ff_probs.shape[0] - len(ff_tokens))
                    if prob_ix >= 0:
                        prob = float(ff_probs[prob_ix, token_id])
                        top_k_ixs = get_top_k(ff_probs[prob_ix], self._top_k if self._enable_top_k else 0)
                        if token_id not in top_k_ixs:
                            top_k_ixs.append(token_id)
                        for top_k_token_id in top_k_ixs:
                            top_k.append(
                                GenToken(
                                    token_id=top_k_token_id,
                                    prob=float(ff_probs[prob_ix, top_k_token_id]),
                                    bytes=self.tokenizer.decode([top_k_token_id]),
                                    latency_ms=ff_lat_ms / len(ff_tokens),
                                    is_input=issued_token is None,
                                    is_force_forwarded=issued_token is not None,
                                    is_masked=top_k_token_id != token_id,
                                )
                            )
                gen_tokens.append(
                    GenTokenExtra(
                        token_id=token_id,
                        bytes=self.tokenizer.decode([token_id]),
                        prob=prob,
                        latency_ms=ff_lat_ms / len(ff_tokens),
                        is_input=issued_token is None,
                        is_force_forwarded=issued_token is not None,
                        is_masked=False,
                        top_k=top_k,
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

            issued_token = self.get_next_token_with_top_k(
                logits=logits[-1, :],
                logits_lat_ms=logits_lat_ms,
                token_ids=tokens,
                mask=mask_for_sampling,
                temperature=ll_response.temperature,
                k=self._top_k if self._enable_top_k else 0,
                compute_unmasked_probs=self._enable_token_probabilities,
                sampling_params=sampling_params,
            )
            last_temperature = ll_response.temperature

            if can_finish_early and not mask[issued_token.token_id]:
                # Type checker needs some help
                assert self.tokenizer.eos_token_id is not None
                issued_token.token_id = self.tokenizer.eos_token_id

            if usage.ttft_ms == 0:
                usage.ttft_ms += (time.monotonic() - t0) * 1000

        usage.total_latency_ms += (time.monotonic() - t0) * 1000
        return usage

    def get_next_token_with_top_k(
        self,
        logits: NDArray,
        logits_lat_ms: float,
        token_ids: list[int],
        mask: Optional[bytes],
        temperature: float,
        k: int,
        compute_unmasked_probs: bool,
        sampling_params: Optional[SamplingParams],
    ) -> GenTokenExtra:
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
        t0 = time.monotonic()

        if k > 0 and not compute_unmasked_probs:
            raise ValueError("If k > 0, compute_unmasked_probs must be True to get the top-k tokens.")

        probs: Optional[NDArray] = None
        top_k: list[int] = []
        if compute_unmasked_probs or mask is None:
            # NOTE: we clone logits here to avoid modifying the original logits twice
            filtered_logits = apply_temp_and_sampling_params(
                np.array(logits, copy=True), token_ids, temperature, sampling_params
            )
            probs = softmax(filtered_logits)
            # Get the top-k tokens from the unmasked logits
            top_k = get_top_k(probs, k)

        masked_probs: Optional[NDArray] = None
        if mask is not None:
            np_mask = np.frombuffer(mask, dtype=np.uint8)
            masked_logits = np.where(np_mask != 0, logits, -np.inf)
            # TODO: if temp is 0, we only need to apply the params that affect argmax, e.g. repetition penalty
            filtered_masked_logits = apply_temp_and_sampling_params(
                masked_logits, token_ids, temperature, sampling_params
            )
            masked_probs = softmax(filtered_masked_logits)

        if temperature < _TEMPERATURE_EPSILON:
            # Greedy sampling
            if mask is None:
                assert probs is not None, "Probs should not be None when mask is None"
                if len(top_k) == 0:
                    issued_token = np.argmax(probs)
                else:
                    # If we have top_k, we can just return the first one
                    issued_token = top_k[0]
            else:
                assert masked_probs is not None, "Masked probabilities should not be None when mask is provided"
                issued_token = np.argmax(masked_probs)
        else:
            # We need to sample from the probabilities
            if mask is None:
                assert probs is not None, "Probs should not be None when mask is None"
                issued_token = np.random.choice(len(probs), p=probs)
            else:
                assert masked_probs is not None, "Masked probabilities should not be None when mask is provided"
                issued_token = np.random.choice(len(masked_probs), p=masked_probs)

        if issued_token not in top_k:
            # This ensures that the issued token is always included in the top-k tokens
            # Note: needs to be added to the end in order to maintain sorted order
            top_k.append(issued_token)

        issued_token_bytes = self.tokenizer.decode([issued_token])
        top_k_token_bytes = [self.tokenizer.decode([token_id]) for token_id in top_k]

        sampling_lat_ms = (time.monotonic() - t0) * 1000

        top_k_tokens = [
            GenToken(
                token_id=token_id,
                prob=float("nan") if probs is None else float(probs[token_id]),
                bytes=token_bytes,
                latency_ms=logits_lat_ms + sampling_lat_ms,
                is_generated=True,
                is_masked=mask is not None and bool(mask[token_id] == 0),
            )
            for token_id, token_bytes in zip(top_k, top_k_token_bytes)
        ]

        return GenTokenExtra(
            token_id=issued_token,
            prob=float("nan") if probs is None else float(probs[issued_token]),
            bytes=issued_token_bytes,
            latency_ms=logits_lat_ms + sampling_lat_ms,
            is_generated=True,
            top_k=top_k_tokens,
        )

    def chat_completion_streaming(
        self, messages: dict[str, str], grammar: str, tools: Union[list[dict[str, Any]], None] = None
    ) -> Generator[tuple[bytes, dict[str, str]], None, None]:
        """Generate a single streaming chat completion, constrained by a Lark grammar.

        This function provides low level access to Guidance, similar to calling an Azure OpenAI endpoint
        with a Lark grammar.
        It is very much experimental in nature, and the API is subject to change.
        """
        # Get the tokens which might be needed by the chat template
        tokens = {
            "eos_token": self.tokenizer.eos_token.decode("utf-8"),
            "bos_token": self.tokenizer.bos_token.decode("utf-8"),
        }

        # Render the messages
        chat_template = self.get_chat_template().template_str
        rtemplate = Environment(loader=BaseLoader).from_string(chat_template)
        rendered_prompt = rtemplate.render(messages=messages, tools=tools, **tokens)
        rendered_prompt += self.get_chat_template().get_role_start("assistant")

        # Load into a State object
        state = EngineState()
        state.prompt = rendered_prompt

        for nxt in self(state, grammar):
            nxt_tokens = [x.token_id for x in nxt.tokens]
            nxt_bytes = self.tokenizer.decode(nxt_tokens)
            nxt_captures = {}
            for k, v in nxt.capture_groups.items():
                nxt_captures[k] = v.decode("utf-8")
            yield nxt_bytes, nxt_captures

    def chat_completion(
        self, messages: dict[str, str], grammar: str, tools: Union[list[dict[str, Any]], None] = None
    ) -> tuple[str, dict[str, str]]:
        """Generate a single chat completion, constrained by a Lark grammar.

        This function provides low level access to Guidance, similar to calling an Azure OpenAI endpoint
        with a Lark grammar.
        It is very much experimental in nature, and the API is subject to change.
        """

        full_response = bytearray()
        captures: dict[str, str] = {}
        for nxt_bytes, nxt_captures in self.chat_completion_streaming(messages, grammar, tools):
            full_response += nxt_bytes
            captures.update(nxt_captures)

        return full_response.decode("utf-8"), captures

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


def get_top_k(_probs: NDArray, _k: int = 5) -> list[int]:
    if _k <= 0:
        return []
    top_k_indices = _probs.argpartition(-_k)[-_k:].tolist()
    # Sort by probability in descending order, as above argpartition
    # does not guarantee order. Sorting the smaller array is faster.
    return sorted(top_k_indices, key=lambda idx: _probs[idx], reverse=True)


def apply_temp_and_sampling_params(
    logits: NDArray,
    token_ids: list[int],
    temperature: float,
    sampling_params: Optional[SamplingParams],
) -> NDArray:
    """Apply the sampling parameters to the logits."""
    if sampling_params is None:
        return logits
    logits = apply_repetition_penalty(token_ids, logits, sampling_params)
    if temperature >= _TEMPERATURE_EPSILON:
        # https://github.com/vllm-project/vllm/blob/e17a4d3bf9cffe32ec308a5979790732818e4919/vllm/sampling_params.py#L355
        # follow vllm sampling strategy for low sampling temperature
        logits = logits / temperature
        logits = apply_min_p_filter(logits, sampling_params)
        logits = apply_top_k_and_top_p_filter(logits, sampling_params)
    return logits
