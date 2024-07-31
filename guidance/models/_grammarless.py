import os
import logging
import queue
import threading
import time

from typing import Iterator, List, Optional, Sequence

import numpy as np
import tiktoken

from guidance._schema import EngineCallResponse
from guidance.models._byte_tokenizer import ByteTokenizer

from ..chat import ChatMLTemplate
from ._model import ConstraintException, Engine, Modality, Model, PromptPart
from ._tokenizer import Tokenizer

logger = logging.getLogger(__name__)

try:
    from .. import cpp  # type: ignore[attr-defined]
except ImportError:
    logger.warn(
        "Failed to load guidance.cpp, falling back to Python mirror implementations..."
    )
    from .. import _cpp as cpp


class GrammarlessEngine(Engine):
    def __init__(
        self, tokenizer, max_streaming_tokens: int, timeout, compute_log_probs: bool
    ):
        self.max_streaming_tokens = max_streaming_tokens
        self.timeout = timeout

        # this is where the streaming thread puts results
        self._data_queue: queue.Queue = queue.Queue()
        # TODO: use tokens here, not bytes. We don't need to re-tokenize, but we do need
        # to know how the model tokenized the response in order to send the EngineCallResponse.
        self._data = b""  # these are the bytes we are ready to use in the main thread

        # this is phrased negatively so we can wait for the stop event
        self._not_running_stream: threading.Event = threading.Event()
        self._last_call_time = 0.0
        self._num_calls_made = 0 # Might be able to get rid of this except for when retrying due to API failure
        self._current_temp = 0.0 # Probably don't need this
        self._last_stream_start = b""

        self._not_running_stream.set()

        self.max_repeated_calls = 10
        self.timeout = timeout

        # # Fall back to ByteTokenizer if no tokenizer is provided
        # if not isinstance(tokenizer, Tokenizer):
        #     tokenizer = ByteTokenizer()

        # build the Engine
        super().__init__(tokenizer=tokenizer, compute_log_probs=compute_log_probs)

    def _generator(self, prompt: bytes, temperature: float) -> Iterator[bytes]:
        raise NotImplementedError("Child classes must implement _generator()")

    def __call__(self, prompt: List[PromptPart], grammar, ensure_bos_token=True) -> Iterator[EngineCallResponse]:
        self._num_calls_made = 0  # reset the number of calls count so we only limit the number of calls within a single grammar execution
        response = EngineCallResponse()
        yield response

    def _running_stream(self):
        return not self._not_running_stream.is_set()  # wrap double negation (which)

    def _start_generator_stream(self, generator):
        logger.debug(f"start Grammarless._start_generator_stream")
        dqueue = self._data_queue
        first_iteration = True
        try:
            for chunk in generator:
                logger.debug(f"Got chunk: " + str(chunk))
                if len(chunk) > 0:
                    dqueue.put(chunk)
                if (
                    self._not_running_stream.is_set()
                    or not first_iteration
                    and time.time() - self._last_call_time > self.timeout
                ):
                    break
                first_iteration = False

        # we pass any exceptions back to the main thread
        except Exception as e:
            self._not_running_stream.set()
            while not dqueue.empty():
                dqueue.get()
            dqueue.put(e)

        self._not_running_stream.set()
        dqueue.put(b"")  # so we never get stuck waiting for a running stream to return something

    def _start_new_stream(self, prompt: bytes, temperature: float) -> None:
        assert isinstance(prompt, bytes)
        # make sure the display is up to date (since we are about to delay for a while)
        # TODO: how can we handle this better since the engine is now separate from the client?
        #       we could use a timeout for the GUI update throttling, those were just kind of slow... (but would be best)
        # self._update_display(throttle=False)

        if self._num_calls_made > self.max_repeated_calls:
            raise Exception(
                f"We have exceeded the maximum number of repeat calls ({self.max_repeated_calls}) per grammar execution!"
            )

        # stop any running stream
        if self._running_stream():
            # Stop stream and wait for thread to complete
            self._not_running_stream.set()
            self._model_interaction_thread.join()  # type: ignore # mypy being strange

        # clear the data queue
        while not self._data_queue.empty():
            self._data_queue.get()

        # start the new stream
        self._used_bytes_len = 0
        self._current_temp = temperature
        self._last_call_time = time.time()
        generator = self._generator(prompt, temperature)
        self._not_running_stream.clear()  # so we know we are running
        self._num_calls_made += 1
        self._model_interaction_thread = threading.Thread(
            target=self._start_generator_stream, args=(generator,)
        )
        self._model_interaction_thread.start()

    def _reset_shared_data(self, new_data: bytes, temperature: float):
        """Should be called by _generator calls to reset the shared data state."""
        assert isinstance(new_data, bytes)
        if temperature == 0 and self._last_stream_start == new_data:
            raise self._report_failed_match(new_data)
        self._data = new_data
        self._last_stream_start = self._data

    def get_next_token(
        self, prompt: list[PromptPart], token_ids: list[list[int]], mask: Optional[bytes], temperature: float) -> int:

        logger.debug(
            f"Start Grammarless.get_next_token({token_ids=}, {mask=}, {temperature=})"
        )
        if len(token_ids) == 0:
            raise ValueError("token_ids must contain some tokens.")

        # compute the prompt bytes
        # TODO: we need to get the forced bytes from the mask -- should streamline this?
        if mask is not None:
            forced_bytes = os.path.commonprefix([self.tokenizer.tokens[i] for i, b in enumerate(mask) if b != 0])
        else:
            forced_bytes = b""

        # whole_token_prompt = self.tokenizer.decode(token_ids)
        prompt.append(PromptPart(modality=Modality.TEXT, content=forced_bytes))
        logger.debug(f"Grammarless.get_next_token: {prompt=}")

        self._last_call_time = time.time()

        # keep looping until we have at least one more byte past our prompt
        token_id = None
        restarted = False  # track if we have restarted the data stream during this call
        while True:
            logger.debug(f"Grammarless.get_next_token: Starting main loop")

            # if the generation temperature changes we have to restart
            if self._current_temp != temperature:
                logger.debug(f"Grammarless.get_next_token: Starting new stream")
                self._start_new_stream(prompt, temperature)
                continue

            # try and get the next token id
            elif self._data.startswith(prompt):
                logger.debug(f"Grammarless.get_next_token: Getting next token id")
                # This used to walk the token byte trie
                token_id = self._get_next_token(len(prompt) - len(forced_bytes))
                logger.debug(f"Grammarless.get_next_token: {token_id=}")
                if token_id is not None:

                    # if we have a non-zero sampling temperature we can't reuse bytes
                    new_used_len = len(whole_token_prompt) + len(
                        self.tokenizer.tokens[token_id]
                    )
                    logger.debug(f"Grammarless.get_next_token: {new_used_len=}")
                    if temperature > 0 and self._used_bytes_len >= new_used_len:
                        logger.debug(f"Grammarless.get_next_token: Need to restart stream")
                        token_id = None
                        self._start_new_stream(prompt, temperature)
                        continue

                    # ...otherwise we have found the token id we want to emit
                    else:
                        logger.debug(f"Grammarless.get_next_token: Found token id")
                        self._used_bytes_len = len(whole_token_prompt) + len(
                            self.tokenizer.tokens[token_id]
                        )
                        break

            # restart if extending our data will never lead to matching our prompt
            elif not self._data.startswith(prompt) and len(self._data) >= len(
                prompt
            ):  # not prompt.startswith(self._data): # len(self._data) >= len(prompt) or
                logger.debug(f"Grammarless.get_next_token: Data will not match prompt")
                # check if we have already restarted once and so retrying by default is not likely to be helpful
                if restarted:
                    raise self._report_failed_match(prompt)

                # check the length of the prefix match
                match_len = 0
                found_mismatch = False
                data = self._data
                for match_len, v in enumerate(prompt):
                    if v != data[match_len]:
                        found_mismatch = True
                        break
                if not found_mismatch:
                    match_len = len(prompt)
                leftover = prompt[match_len:]
                logger.debug(f"Grammarless.get_next_token: {leftover=}")

                # record any active non-empty role ends. Ignore role ends that are spaces
                parts: Sequence[Optional[bytes]] = [
                    b"<|im_end|>",
                    self.tokenizer.eos_token,
                ]  # note we assume we are role tags that end with <|im_end|>

                # see if adding an end token would work here (if so we avoid recalling the server and just produce an end token)
                found_match = False
                for p in parts:
                    logger.debug(f"Grammarless.get_next_token: Considering part {str(p)}")
                    if p is not None:
                        if p.startswith(leftover):
                            self._data = self._data[:match_len] + p
                            logger.debug(
                                f"automatically adding an end token since it fits the forcing of the grammar"
                            )
                            found_match = True
                            break
                if found_match:
                    continue  # start our loop over again

                logger.debug(
                    f"restarting a stream because the data we have does not match the ids. We have {str(self._data)} but the prompt is {str(prompt)}"
                )
                restarted = True
                self._start_new_stream(prompt, temperature)

            # extend our data with a chunk from the model stream
            if not self._data_queue.empty():
                new_bytes = self._data_queue.get_nowait()
                logger.debug(f"Got {new_bytes} from _data_queue")
                if isinstance(new_bytes, Exception):
                    raise new_bytes

                # if we are at the end of the generation then we try again allowing for early token stopping
                if len(new_bytes) == 0:
                    token_id = self._get_next_token(len(prompt), allow_early_stop=True)
                    if token_id is not None:
                        break
                self._data += new_bytes

            # but if there is nothing and we are not running then we start a stream
            elif self._not_running_stream.is_set():
                if (self.tokenizer.eos_token_id is not None) and (
                    mask is None or mask[self.tokenizer.eos_token_id] != 0
                ):
                    return self.tokenizer.eos_token_id
                logger.debug(
                    "starting a new stream because there is no data to read and no stream running..."
                )
                restarted = True
                self._start_new_stream(prompt, temperature)

            # we wait for the running stream to put something in the queue
            else:
                # Set to essentialy infinity so we don't stop the data stream while we are waiting for it
                self._last_call_time = 1e11

                new_bytes = self._data_queue.get()
                logger.debug(f"Got {new_bytes} from _data_queue")
                if isinstance(new_bytes, Exception):
                    raise new_bytes
                self._data += new_bytes
                # reset out call time to allow the data stream to time out if we happen to be done with it
                self._last_call_time = time.time()

        return token_id


class Grammarless(Model):
    """The base class for all models that do not support Guidance grammars."""

    pass

    # def __init__(self, model, tokenizer=None, echo=True, compute_log_probs=False, max_streaming_tokens=None, timeout=0.5):
    #     '''Build a new remote model object that represents a model in a given state.

    #     This is an abstract class. To instantiate it use a specific subclass like guidance.models.OpenAI.
    #     '''
    #     super().__init__(
    #         engine=GrammarlessEngine(model, tokenizer, max_streaming_tokens, timeout, compute_log_probs),
    #         echo=echo
    #     )
