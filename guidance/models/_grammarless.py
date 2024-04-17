import threading
import numpy as np
import queue
import time
import tiktoken
import re
import logging
from ._model import Tokenizer, Engine, Model, format_pattern, ConstraintException

logger = logging.getLogger(__name__)


class GrammarlessTokenizer(Tokenizer):
    def __init__(self, tokenizer):

        # Grammarless models don't always have public tokenizations, so when not provided we pretend they tokenize like gpt2...
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("gpt2")

        # tiktoken tokenizer was given
        if hasattr(tokenizer, "decode_tokens_bytes"):
            special_map = {v: k for k, v in tokenizer._special_tokens.items()}
            first_special = tokenizer.n_vocab
            for k in special_map:
                if k < first_special:
                    first_special = k

            # consume one-by-one until we have passed all the special tokens AND gotten a valid token
            i = tokenizer.n_vocab - 1
            byte_tokens = []
            while True:
                try:
                    bval = tokenizer.decode_single_token_bytes(i)
                    found = True
                except KeyError:
                    bval = special_map.get(i, b"<|invalid_special_token|>")
                    found = False
                byte_tokens.append(bval)

                if i < first_special and found:
                    break
                i -= 1

            # do the rest of the tokens as a batch
            byte_tokens = tokenizer.decode_tokens_bytes(np.arange(i + 1)) + byte_tokens

            if hasattr(tokenizer, "bos_token_id"):
                bos_token_id = tokenizer.bos_token_id
            else:
                bos_token_id = None
            if hasattr(tokenizer, "eos_token_id"):
                eos_token_id = tokenizer.eos_token_id
            else:
                eos_token_id = tokenizer._special_tokens["<|endoftext|>"]

        # a transformer tokenizer was given that has a byte_decoder
        elif hasattr(tokenizer, "byte_decoder"):
            byte_tokens = []
            for i in range(tokenizer.vocab_size):
                byte_coded = bytes(
                    [
                        tokenizer.byte_decoder[c]
                        for c in tokenizer.convert_ids_to_tokens(i)
                    ]
                )
                byte_tokens.append(byte_coded)
            bos_token_id = tokenizer.bos_token_id
            eos_token_id = tokenizer.eos_token_id

        # a transformer tokenizer was given with byte_decoder
        elif hasattr(tokenizer, "convert_ids_to_tokens"):
            byte_tokens = [
                bytes(
                    tokenizer.convert_tokens_to_string(
                        ["a", tokenizer.convert_ids_to_tokens(i)]
                    )[1:],
                    encoding="utf8",
                )
                for i in range(tokenizer.vocab_size)
            ]
            bos_token_id = tokenizer.bos_token_id
            eos_token_id = tokenizer.eos_token_id

        # a HuggingFace tokenizers tokenizer was given with id_to_token
        elif hasattr(tokenizer, "id_to_token"):
            a_token_ids = tokenizer.encode("a").ids
            if len(a_token_ids) == 3:
                bos_token_id = a_token_ids[0]
                a_id = a_token_ids[1]
                eos_token_id = a_token_ids[2]
            else:
                raise Exception(
                    "This tokenizer does not seem to have a BOS and EOS, support for this need to be implemented still."
                )

            byte_tokens = [
                bytes(tokenizer.decode([a_id, i])[1:], encoding="utf8")
                for i in range(tokenizer.get_vocab_size())
            ]
            for i, b in enumerate(byte_tokens):
                if b == b"":
                    byte_tokens[i] = bytes(tokenizer.id_to_token(i), encoding="utf8")

        else:
            raise Exception("The tokenizer given was not of a recognized type!")

        super().__init__(byte_tokens, bos_token_id, eos_token_id)


class GrammarlessEngine(Engine):
    def __init__(self, tokenizer, max_streaming_tokens, timeout, compute_log_probs):
        self.max_streaming_tokens = max_streaming_tokens
        self.timeout = timeout

        self._data_queue = (
            queue.Queue()
        )  # this is where the streaming thread puts results
        self._data = b""  # these are the bytes we are ready to use in the main thread
        self._not_running_stream = (
            threading.Event()
        )  # this is phrased negatively so we can wait for the stop event
        self._last_call = 0
        self._num_calls_made = 0
        self._current_temp = 0
        self._last_stream_start = None

        self._not_running_stream.set()

        self.max_repeated_calls = 10
        self.timeout = timeout

        # If tokenizer is not already an instance of Tokenizer, then instantiate it as a GrammarlessTokenizer
        if not isinstance(tokenizer, Tokenizer):
            tokenizer = GrammarlessTokenizer(tokenizer)

        # build the
        super().__init__(tokenizer=tokenizer, compute_log_probs=compute_log_probs)

    def __call__(self, *args, **kwargs):
        self._num_calls_made = 0  # reset the number of calls count so we only limit the number of calls within a single grammar execution
        return super().__call__(*args, **kwargs)

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
                    and time.time() - self._last_call > self.timeout
                ):
                    break
                first_iteration = False

        # we pass any exceptions back to the main thread
        except Exception as e:
            self._not_running_stream.set()
            while not dqueue.empty():
                dqueue.get()
            dqueue.put(e)

        if self._running_stream():
            dqueue.put(self.tokenizer.eos_token)
        self._not_running_stream.set()
        dqueue.put(
            b""
        )  # so we never get stuck waiting for a running stream to return something

    def _start_new_stream(self, prompt, temperature):

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
            self._not_running_stream.set()  # stop the generator
            self._remote_thread.join()  # wait for the thread to finish

        # clear the data queue
        while not self._data_queue.empty():
            self._data_queue.get()

        # start the new stream
        self._used_bytes_len = 0
        self._current_temp = temperature
        self._last_call = time.time()
        generator = self._generator(prompt, temperature)
        self._not_running_stream.clear()  # so we know we are running
        self._num_calls_made += 1
        self._remote_thread = threading.Thread(
            target=self._start_generator_stream, args=(generator,)
        )
        self._remote_thread.start()

    def _reset_shared_data(self, new_data, temperature):
        """Should be called by _generator calls to reset the shared data state."""
        if temperature == 0 and self._last_stream_start == new_data:
            raise self._report_failed_match(new_data)
        self._data = new_data
        self._last_stream_start = self._data

    def get_logits(self, token_ids, forced_bytes, current_temp):
        """Computes the logits for the given token state.

        This overrides a method from the Local class that is used to get
        inference results from the model.
        """

        logger.debug(f"start Grammarless._get_logits(token_ids={token_ids})")

        if len(token_ids) == 0:
            raise ValueError("token_ids must contain some tokens.")

        # compute the prompt bytes
        whole_token_prompt = b"".join([self.tokenizer.tokens[i] for i in token_ids])
        prompt = whole_token_prompt + forced_bytes

        self._last_call = time.time()

        # keep looping until we have at least one more byte past our prompt
        token_id = None
        restarted = False  # track if we have restarted the data stream during this call
        while True:

            # if the generation temperature changes we have to restart
            if self._current_temp != current_temp:
                self._start_new_stream(prompt, current_temp)
                continue

            # try and get the next token id
            elif self._data.startswith(prompt):
                token_id = self._get_next_token(len(prompt) - len(forced_bytes))
                if token_id is not None:

                    # if we have a non-zero sampling temperature we can't reuse bytes
                    new_used_len = len(whole_token_prompt) + len(
                        self.tokenizer.tokens[token_id]
                    )
                    if current_temp > 0 and self._used_bytes_len >= new_used_len:
                        token_id = None
                        self._start_new_stream(prompt, current_temp)
                        continue

                    # ...otherwise we have found the token id we want to emit
                    else:
                        self._used_bytes_len = len(whole_token_prompt) + len(
                            self.tokenizer.tokens[token_id]
                        )
                        break

            # restart if extending our data will never lead to matching our prompt
            elif not self._data.startswith(prompt) and len(self._data) >= len(
                prompt
            ):  # not prompt.startswith(self._data): # len(self._data) >= len(prompt) or

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

                # record any active non-empty role ends. Ignore role ends that are spaces
                parts = [
                    b"<|im_end|>",
                    self.tokenizer.eos_token,
                ]  # note we assume we are role tags that end with <|im_end|>

                # for _,role_end_str in self.opened_blocks.values():
                #     role_end_str = format_pattern.sub("", role_end_str)
                #     if len(role_end_str) > 0 and not re.fullmatch(r'\s+', role_end_str):
                #         parts.append(role_end_str.encode("utf8"))

                # # record the eos token
                # parts.append(self.eos_token)

                # see if adding an end token would work here (if so we avoid recalling the server and just produce an end token)
                found_match = False
                for p in parts:
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
                self._start_new_stream(prompt, current_temp)

            # extend our data with a chunk from the model stream
            if not self._data_queue.empty():
                new_bytes = self._data_queue.get_nowait()
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
                logger.debug(
                    "starting a new stream because there is no data to read and no stream running..."
                )
                restarted = True
                self._start_new_stream(prompt, current_temp)

            # we wait for the running stream to put something in the queue
            else:
                self._last_call = 10e9  # set to essentialy infinity so we don't stop the data stream while we are waiting for it
                new_bytes = self._data_queue.get()
                if isinstance(new_bytes, Exception):
                    raise new_bytes
                self._data += new_bytes
                self._last_call = (
                    time.time()
                )  # reset out call time to allow the data stream to time out if we happen to be done with it

        # # if we don't have the next byte of data yet then we wait for it (from the streaming thread)
        # if len(self._data) == len(prompt):
        #     self._data += self._data_queue.get()

        # token_id = self._get_next_token(len(prompt))

        # set the logits to the next byte the model picked
        logits = np.ones(len(self.tokenizer.tokens)) * -np.inf
        logits[token_id] = 100
        if token_id != self.tokenizer.eos_token:
            logits[self.tokenizer.eos_token_id] = (
                0  # we always allow the model to use EOS if that is the only way forward
            )

        return logits

    def _report_failed_match(self, prompt):

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

        # compute the mismatch parts
        data_after_prompt = self._data[match_len:]
        if len(data_after_prompt) > 40:
            data_after_prompt = data_after_prompt[:40] + b"..."
        prompt_tail = prompt[:match_len]
        if len(prompt_tail) > 40:
            prompt_tail = b"..." + prompt_tail[-40:]

        # show in the model output where and how we diverged from the grammar
        try:
            # just for display when echo is on
            already_shown = len(self._current_prompt().encode())
            self += (
                self._data[already_shown:match_len].decode()
                + f"<||_html:<span style='color: rgba(165,0,0,1);' title='{leftover}'><span style='text-decoration: underline;'>{data_after_prompt.decode()}</span></span>_||>"
            )
        except:
            pass  # could not decode the data the model generated into a string...

        # create an exception for users to deal with (that our caller can throw)
        return ConstraintException(
            f"The model attempted to generate {str(data_after_prompt)} after the prompt `{prompt_tail}`, but that does\n"
            + "not match the given grammar constraints! Since your model is a remote API that does not support full guidance\n"
            + "integration we cannot force the model to follow the grammar, only flag an error when it fails to match.\n"
            + "You can try to address this by improving the prompt, making your grammar more flexible, rerunning with\n"
            + "a non-zero temperature, or using a model that supports full guidance grammar constraints."
        )

    def _get_next_token(self, pos, allow_early_stop=False):
        data = self._data
        trie = self._token_trie
        token_id = None
        while True:

            # see if we have run out of data
            if pos >= len(data):
                if allow_early_stop:
                    return token_id
                else:
                    return None

            # try and walk down the trie
            next_byte = data[pos : pos + 1]
            if trie.has_child(next_byte):
                trie = trie.child(next_byte)
                pos += 1
                if trie.value >= 0:
                    token_id = trie.value
            else:
                return token_id  # this is the longest greedy token match we can make


class Grammarless(Model):
    """The base class for all remote models (hosted behind a remote API)."""

    pass

    # def __init__(self, model, tokenizer=None, echo=True, compute_log_probs=False, max_streaming_tokens=None, timeout=0.5):
    #     '''Build a new remote model object that represents a model in a given state.

    #     This is an abstract class. To instantiate it use a specific subclass like guidance.models.OpenAI.
    #     '''
    #     super().__init__(
    #         engine=GrammarlessEngine(model, tokenizer, max_streaming_tokens, timeout, compute_log_probs),
    #         echo=echo
    #     )
