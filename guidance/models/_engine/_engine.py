# TODO(nopdive): This module requires a memory review.

import logging
import time
import weakref
from abc import ABC
from asyncio import CancelledError
from enum import Enum
from multiprocessing import Manager, Process
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Sequence, Union

import numpy as np
import psutil

from ..._parser import TokenParser
from ..._schema import (
    EngineCallResponse,
    EngineOutput,
    GenToken,
    GenTokenExtra,
    GuidanceEngineMetrics,
    LLGrammar,
)
from ..._singleton import get_renderer, get_trace_handler
from ..._utils import log_cleanup, log_init, softmax, to_utf8_or_bytes_string
from ...visual import (
    AutoRenderer,
    ExecutionCompletedMessage,
    ExecutionStartedMessage,
    GuidanceMessage,
    JupyterWidgetRenderer,
    MetricMessage,
    OutputRequestMessage,
    Renderer,
    TokensMessage,
)
from ...visual._async import async_task, run_async_coroutine
from ._state import EngineState
from ._tokenizer import Tokenizer

if TYPE_CHECKING:
    from .._base._model import Model

logger = logging.getLogger(__name__)


class PeriodicMetricsGenerator:
    def __init__(self, renderer: Renderer, monitor: "Monitor", sleep_sec=0.5):
        self._renderer = renderer
        self._monitor = monitor
        self._sleep_sec = sleep_sec
        self._task = None
        self._task_cancelled = False
        self._is_paused = False
        run_async_coroutine(self._emit())

    def start(self):
        self._task = run_async_coroutine(async_task(self._emit())).result()

    def stop(self):
        if self._task is not None:
            self._task.cancel()
            # TODO: it seems _task.cancel() is not working, use a flag to stop the task
            self._task_cancelled = True

    def pause(self):
        """
        Pauses the model by setting the internal _is_paused flag to True.

        This method can be used to temporarily halt the model's operations.
        """
        self._is_paused = True

    def resume(self):
        """
        Resume the model's operation by setting the paused state to False.

        This method changes the internal state of the model to indicate that it is no longer paused.
        """
        self._is_paused = False

    async def _emit(self):
        import asyncio
        import time

        time_start = time.time()
        while not self._task_cancelled:
            try:
                await asyncio.sleep(self._sleep_sec)

                cpu_percent = self._monitor.get_metric(MonitoringMetric.CPU_USAGE)
                used_ram = self._monitor.get_metric(MonitoringMetric.MEM_USAGE)
                gpu_percent = self._monitor.get_metric(MonitoringMetric.GPU_USAGE)
                gpu_used_vram = self._monitor.get_metric(MonitoringMetric.GPU_USED_MEM)

                if gpu_percent:
                    gpu_percent = max(gpu_percent)
                else:
                    gpu_percent = 0

                if gpu_used_vram:
                    gpu_used_vram = max(gpu_used_vram)
                else:
                    gpu_used_vram = 0

                if not cpu_percent:
                    cpu_percent = 0

                if not used_ram:
                    used_ram = 0

                time_end = time.time()
                time_elapsed = time_end - time_start

                if not self._is_paused:
                    self._renderer.update(MetricMessage(name="wall time", value=time_elapsed))
                    self._renderer.update(MetricMessage(name="cpu", value=cpu_percent))
                    self._renderer.update(MetricMessage(name="ram", value=used_ram))
                    self._renderer.update(MetricMessage(name="gpu", value=gpu_percent))
                    self._renderer.update(MetricMessage(name="vram", value=gpu_used_vram))
            except CancelledError:
                logger.debug("METRICGEN:canceling")
                break
            except Exception as e:
                logger.debug(f"METRICGEN: {repr(e)}")
                break

        logger.debug("METRICGEN:exiting")


class PostExecMetrics:
    def __init__(self, renderer: Renderer, monitor: "Monitor"):
        self._renderer = renderer
        self._monitor = monitor

    def emit_messages(self, lm: "Model"):
        token_reduction = self._monitor.get_metric(MonitoringMetric.TOKEN_REDUCTION, lm)
        if token_reduction is not None:
            self._renderer.update(
                MetricMessage(
                    name="token reduction",
                    value=token_reduction * 100,
                )
            )

        output_tokens = self._monitor.get_metric(MonitoringMetric.OUTPUT_TOKENS, lm)
        if output_tokens is not None:
            self._renderer.update(MetricMessage(name="consumed", value=output_tokens))

        avg_latency = self._monitor.get_metric(MonitoringMetric.AVG_LATENCY, lm)
        if avg_latency is not None:
            self._renderer.update(MetricMessage(name="avg latency", value=avg_latency))


def _engine_cleanup(
    renderer: Renderer,
    msg_recv: Callable[[GuidanceMessage], None],
    monitor: Optional["Monitor"],
    periodic_metrics_generator: Optional[PeriodicMetricsGenerator],
    log_msg: str,
):
    renderer.unsubscribe(msg_recv)

    try:
        # force renderer cleanup
        # TODO: figure out why in some cases _recv_task and _send_task are not stopped
        from ...visual._renderer import _cleanup

        if isinstance(renderer, AutoRenderer) and isinstance(
            renderer._renderer, JupyterWidgetRenderer
        ):
            _cleanup(
                renderer._renderer._recv_queue,
                renderer._renderer._send_queue,
                f"renderer({id(renderer)})",
            )
    except Exception as e:
        logger.error(f"Failed to force-cleanup renderer: {e}")

    if periodic_metrics_generator is not None:
        try:
            periodic_metrics_generator.stop()
        except Exception as e:
            logger.error(f"Failed to stop periodic metrics generator: {e}")

    if monitor is not None:
        try:
            monitor.stop()
        except Exception as e:
            logger.error(f"Failed to stop monitor: {e}")

    log_cleanup(log_msg)


def _wrapped_msg_recv(engine_weak_ref: weakref.ref) -> Callable[[GuidanceMessage], None]:
    def closure(message):
        return _msg_recv(engine_weak_ref, message)

    return closure


def _msg_recv(engine_weakref: weakref.ReferenceType, message: GuidanceMessage) -> None:
    engine = engine_weakref()
    if engine is None:
        return

    # NOTE(nopdive): This is usually run on a background thread.
    logger.debug(f"ENGINE:msg_recv:{message}")
    if isinstance(message, ExecutionStartedMessage):
        # TODO(nopdive): Start execution logic here.
        if engine.periodic_metrics_generator is not None:
            engine.periodic_metrics_generator.resume()
    elif isinstance(message, ExecutionCompletedMessage) and message.is_err:
        pass
    elif isinstance(message, (ExecutionCompletedMessage, OutputRequestMessage, TokensMessage)):
        failed = False
        processed_gen_tokens: list[GenTokenExtra] = []  # suppress IDE warnings by definition
        try:
            last_trace_id = message.last_trace_id
            last_model: "Model" = engine.model_dict[message.last_trace_id]
            processed_gen_tokens = last_model.get_per_token_stats()
        except Exception as e:
            logger.error(f"Failed to get per token stats: {e}")
            failed = True

        if not failed:
            final_text = "".join([gen_token.text for gen_token in processed_gen_tokens])
            logger.debug(f"ENGINE:final_text:{final_text}")
            logger.debug(f"ENGINE:model_state:{last_model._state}")
            logger.debug(f"ENGINE:final_text == _state:{final_text == last_model._state}")

            tokens = [gen_token.token_id for gen_token in processed_gen_tokens]
            engine.renderer.update(
                TokensMessage(
                    trace_id=last_trace_id,
                    text=engine.tokenizer.decode(tokens).decode("utf-8"),
                    tokens=processed_gen_tokens,
                )
            )
            engine.renderer.update(MetricMessage(name="status", value="Done"))
        else:
            engine.renderer.update(MetricMessage(name="status", value="Error"))

        if engine.periodic_metrics_generator is not None:
            engine.periodic_metrics_generator.pause()

        try:
            # send stats to the renderer
            engine.post_exec_metrics.emit_messages(last_model)
        except:
            pass


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
        compute_log_probs=False,
        enable_backtrack=True,
        enable_ff_tokens=True,
        renderer=None,
        enable_monitoring=True,
        **kwargs,
    ):
        # TODO(nopdive): Hook up renderer keyword to all engines.
        self.tokenizer = tokenizer
        self.compute_log_probs = compute_log_probs
        self._enable_backtrack = enable_backtrack
        self._enable_ff_tokens = enable_ff_tokens
        self._enable_monitoring = enable_monitoring
        self._top_k = kwargs.get("top_k", 5)
        self.metrics = GuidanceEngineMetrics()

        if renderer is None:
            self.trace_handler = get_trace_handler()
            self.renderer = get_renderer(self.trace_handler)
        else:
            self.renderer = renderer
            self.trace_handler = renderer._trace_handler

        msg_recv = _wrapped_msg_recv(weakref.ref(self))
        self.renderer.subscribe(msg_recv)

        self.model_dict: weakref.WeakValueDictionary[int, "Model"] = weakref.WeakValueDictionary()

        self.monitor = None
        self.periodic_metrics_generator = None
        self.post_exec_metrics = None
        if self._enable_monitoring:
            self.monitor = Monitor(self.metrics)
            self.monitor.start()

            self.periodic_metrics_generator = PeriodicMetricsGenerator(self.renderer, self.monitor)
            self.periodic_metrics_generator.start()
            self.post_exec_metrics = PostExecMetrics(self.renderer, self.monitor)

        weakref.finalize(
            self,
            _engine_cleanup,
            self.renderer,
            msg_recv,
            self.monitor,
            self.periodic_metrics_generator,
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

        tokens = self.tokenizer.encode(state.prompt.encode("utf-8"))
        # add the beginning of sequence token if needed
        if (
            ensure_bos_token
            and self.tokenizer.bos_token is not None
            and tokens[:1] != [self.tokenizer.bos_token_id]
        ):
            tokens = [self.tokenizer.bos_token_id] + tokens
            tokens = self.tokenizer.recode(tokens)

        parser = TokenParser(
            grammar,
            tokenizer=self.tokenizer,
            prompt_tokens=tokens,
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

            if engine_output is not None:
                token_id = engine_output.issued_token.token_id
            else:
                token_id = None
            backtrack, ff_tokens, mask_fut = parser.advance(token_id)
            if backtrack:
                tokens = tokens[:-backtrack]
            tokens += ff_tokens

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


class MonitoringMetric(str, Enum):
    CPU_USAGE = "cpu_usage"
    MEM_USAGE = "mem_usage"
    GPU_USAGE = "gpu_usage"
    GPU_USED_MEM = "gpu_used_mem"
    GPU_TOTAL_MEM = "gpu_total_mem"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    BACKTRACK_TOKENS = "backtrack_tokens"
    TOKEN_COUNT = "token_count"
    TOKEN_REDUCTION = "token_reduction"
    AVG_LATENCY = "avg_latency"


ALL_METRICS = [
    MonitoringMetric.CPU_USAGE,
    MonitoringMetric.MEM_USAGE,
    MonitoringMetric.GPU_USAGE,
    MonitoringMetric.GPU_USED_MEM,
    MonitoringMetric.GPU_TOTAL_MEM,
    MonitoringMetric.INPUT_TOKENS,
    MonitoringMetric.OUTPUT_TOKENS,
    MonitoringMetric.BACKTRACK_TOKENS,
    MonitoringMetric.TOKEN_COUNT,
    MonitoringMetric.TOKEN_REDUCTION,
    MonitoringMetric.AVG_LATENCY,
]


def _monitor_fn(
    stop_flag,
    metrics_dict: dict[MonitoringMetric, list],
    max_size: int = 100,
    interval_ms: float = 1000,
):
    # print("Monitoring started")

    to_collect_gpu_stats = False
    has_gpustat = False
    try:
        import gpustat

        has_gpustat = True
    except:
        logger.warning("gpustat is not installed, run `pip install gpustat` to collect GPU stats.")

    if has_gpustat:
        try:
            gpu_stats = gpustat.GPUStatCollection.new_query()
            if len(gpu_stats) > 0:
                # only collect GPU stats if there is at least one GPU
                to_collect_gpu_stats = True
        except:
            logger.warning("Non-Nvidia GPU monitoring is not supported in this version.")

    try:
        while not stop_flag.value:
            t0 = time.time()

            # cpu_percent = psutil.cpu_percent(interval=1)
            cpu_percent = psutil.cpu_percent() / 100.0
            memory_usage = psutil.virtual_memory()

            metrics_dict[MonitoringMetric.CPU_USAGE].append(cpu_percent)
            metrics_dict[MonitoringMetric.MEM_USAGE].append(memory_usage.used / (1024**3))

            if to_collect_gpu_stats:
                gpu_stats = gpustat.GPUStatCollection.new_query()

                usage = [gpu.utilization / 100.0 for gpu in gpu_stats.gpus]
                mem_usage = [gpu.memory_used for gpu in gpu_stats.gpus]
                mem_total = [gpu.memory_total for gpu in gpu_stats.gpus]

                metrics_dict[MonitoringMetric.GPU_USAGE].append(usage)
                metrics_dict[MonitoringMetric.GPU_USED_MEM].append(mem_usage)
                metrics_dict[MonitoringMetric.GPU_TOTAL_MEM].append(mem_total)

            for metrics in metrics_dict.values():
                if len(metrics) > max_size:
                    metrics.pop(0)

            lat = time.time() - t0

            # sleep for the remaining time of the interval
            sleep_time = interval_ms / 1000.0 - lat
            if sleep_time < 0:
                time.sleep(sleep_time)
    except Exception as e:
        # print(f"Error in monitoring: {e}")
        pass

    # print("Monitoring stopped")


class Monitor:
    """Monitoring service to collect necessary metrics for visualization"""

    def __init__(self, engine_metrics: GuidanceEngineMetrics, **kwargs):
        self.engine_metrics = engine_metrics
        self.mp_manager = Manager()

        # use list instead of queue for easily accessing each item, e.g., last item
        self.max_size = kwargs.get("max_size", 100)

        self.metrics_dict = {
            MonitoringMetric.CPU_USAGE: self.mp_manager.list(),
            MonitoringMetric.MEM_USAGE: self.mp_manager.list(),
            MonitoringMetric.GPU_USAGE: self.mp_manager.list(),
            MonitoringMetric.GPU_USED_MEM: self.mp_manager.list(),
            MonitoringMetric.GPU_TOTAL_MEM: self.mp_manager.list(),
        }

        self.stop_flag = self.mp_manager.Value("b", False)
        self.process = None

        self.per_token_metrics = []  # store metrics per token in token list

    def start(self):
        self.process = Process(
            target=_monitor_fn, args=(self.stop_flag, self.metrics_dict, self.max_size)
        )
        self.process.start()
        logger.debug("Monitor:start")

    def stop(self):
        if self.process:
            self.stop_flag.value = True
            self.process.terminate()

            for metrics in self.metrics_dict.values():
                metrics[:] = []  # NOTE(nopdive): ListProxy does not have .clear method.
        logger.debug("Monitor:stop")

    def reset(self):
        self.stop()

        for metrics in self.metrics_dict.values():
            metrics.clear()

        self.start()
        logger.debug("Monitor:reset")

    def get_metrics(
        self, metrics=None, lm: Union["Model", None] = None
    ) -> dict[MonitoringMetric, Any]:
        if metrics is None:
            metrics = ALL_METRICS
        result = {}

        for metric in metrics:
            if metric in [
                MonitoringMetric.CPU_USAGE,
                MonitoringMetric.MEM_USAGE,
                MonitoringMetric.GPU_USAGE,
                MonitoringMetric.GPU_USED_MEM,
                MonitoringMetric.GPU_TOTAL_MEM,
            ]:
                result[metric] = (
                    self.metrics_dict[metric][-1] if len(self.metrics_dict[metric]) > 0 else None
                )
            elif metric == MonitoringMetric.INPUT_TOKENS:
                result[metric] = self.engine_metrics.engine_input_tokens
            elif metric == MonitoringMetric.OUTPUT_TOKENS:
                result[metric] = self.engine_metrics.engine_output_tokens
            elif metric == MonitoringMetric.BACKTRACK_TOKENS:
                result[metric] = self.engine_metrics.engine_backtrack_tokens
            elif metric == MonitoringMetric.TOKEN_COUNT:
                result[metric] = lm.token_count if lm is not None else None
            elif metric == MonitoringMetric.TOKEN_REDUCTION:
                if lm is not None and lm.token_count > 0:
                    result[metric] = 1 - min(1, (lm.metrics.engine_output_tokens / lm.token_count))
                else:
                    result[metric] = None
            elif metric == MonitoringMetric.AVG_LATENCY:
                if lm is None:
                    result[metric] = None
                else:
                    lats = []
                    model = lm
                    while model._parent is not None:
                        if model.vis_chunk:
                            for token in model.vis_chunk.generated_tokens:
                                lats.append(token.latency_ms)
                            for token in model.vis_chunk.force_forwarded_tokens:
                                lats.append(token.latency_ms)
                        model = model._parent

                    if len(lats) == 0:
                        result[metric] = None
                    else:
                        result[metric] = np.mean(lats)

        return result

    def get_metric(self, metric: MonitoringMetric, lm: Union["Model", None] = None) -> Any:
        return self.get_metrics([metric], lm)[metric]
